/************************************************************************
 *  Program: cfd_exp3d_cuda (native CUDA 3D backend, spec v0.2)         *
 *  Description: 3D hydrostatic model in native CUDA, built to test the *
 *               hypothesis of docs/21 S3.2: that the GPU loses on the  *
 *               semi-implicit elliptic solve because every PCG inner   *
 *               product is a device-to-host synchronisation, and that  *
 *               keeping the scalars resident on the device removes it. *
 *               Both synchronisation strategies are in this one binary *
 *               and selected at runtime, so the comparison holds       *
 *               everything else fixed.                                 *
 *  Pipeline: toml2nml.py -> cfd_exp3d_cuda -> compare_backends3d.py    *
 ************************************************************************/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "config.hpp"
#include "pcg_device.cuh"
#include "mg_device.cuh"
#include "kernels3d.cuh"

static constexpr int BLK = 256;
static constexpr double PI = 3.14159265358979323846;

struct Grid3 {
    int nx, ny, nz, n2, n3;
    double dx, dy, dz;
    dim3 b2, g2, b3, g3, b1, g1, gcol;

    explicit Grid3(const RunConfig& c)
        : nx(c.nx), ny(c.ny), nz(c.nz), n2(c.nx * c.ny), n3(c.nx * c.ny * c.nz),
          dx(c.lx / c.nx), dy(c.ly / c.ny), dz(c.h0 / c.nz) {
        b2 = dim3(32, 8);
        g2 = dim3((nx + 31) / 32, (ny + 7) / 8);
        b3 = dim3(32, 8);
        g3 = dim3((nx + 31) / 32, (ny + 7) / 8, nz);
        b1 = dim3(BLK);
        g1 = dim3((n3 + BLK - 1) / BLK);
        gcol = g2;
    }
};

/* ------------------------------------------------- host-side case fields */
struct Host3 {
    std::vector<double> u, v, b, eta;
    Host3(int n3, int n2) : u(n3), v(n3), b(n3), eta(n2) {}
};

static double zt_centre(const RunConfig& c, int k) {
    return (double(c.nz - k) - 0.5) * (c.h0 / c.nz);
}

static void case_fields3d(const RunConfig& c, const Grid3& g, double t, Host3& s) {
    std::fill(s.u.begin(), s.u.end(), 0.0);
    std::fill(s.v.begin(), s.v.end(), 0.0);
    std::fill(s.b.begin(), s.b.end(), 0.0);
    std::fill(s.eta.begin(), s.eta.end(), 0.0);

    const double kx = 2.0 * PI * c.mode_x / c.lx;
    const double ly = 2.0 * PI * c.mode_y / c.ly;
    const double kap2 = kx * kx + ly * ly;

    if (c.case_name == "barotropic3d") {
        const double om = std::sqrt(c.f0 * c.f0 + c.g * c.h0 * kap2);
        const double amp = c.eta0 / (c.h0 * kap2);
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                const double xe = (i + 0.5) * g.dx, ye = (j + 0.5) * g.dy;
                const double xu = (i + 1.0) * g.dx, yv = (j + 1.0) * g.dy;
                const double pu = kx * xu + ly * ye - om * t;
                const double pv = kx * xe + ly * yv - om * t;
                const double pe = kx * xe + ly * ye - om * t;
                s.eta[j * g.nx + i] = c.eta0 * std::cos(pe);
                for (int k = 0; k < g.nz; ++k) {
                    const int cc = (k * g.ny + j) * g.nx + i;
                    s.u[cc] = amp * (om * kx * std::cos(pu) - c.f0 * ly * std::sin(pu));
                    s.v[cc] = amp * (om * ly * std::cos(pv) + c.f0 * kx * std::sin(pv));
                }
            }
        }
    } else if (c.case_name == "vdiffusion") {
        const double m = PI * c.mode_z / c.h0;
        const double lam = (2.0 * std::cos(PI * c.mode_z / c.nz) - 2.0) / (g.dz * g.dz);
        const double decay = std::exp(c.nu * lam * t);
        for (int k = 0; k < g.nz; ++k) {
            const double cz = std::cos(m * zt_centre(c, k));
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    s.u[(k * g.ny + j) * g.nx + i] = c.u0 * cz * decay;
        }
    } else if (c.case_name == "baroclinic_igw") {
        const double m = PI * c.mode_z / c.h0;
        const double c2 = c.n2 / (m * m);
        const double om = std::sqrt(c.f0 * c.f0 + c2 * kap2);
        const double amp = c.g * c.eta0 / (c2 * kap2);
        for (int k = 0; k < g.nz; ++k) {
            const double cz = std::cos(m * zt_centre(c, k));
            const double sz = std::sin(m * zt_centre(c, k));
            for (int j = 0; j < g.ny; ++j) {
                for (int i = 0; i < g.nx; ++i) {
                    const double xe = (i + 0.5) * g.dx, ye = (j + 0.5) * g.dy;
                    const double xu = (i + 1.0) * g.dx, yv = (j + 1.0) * g.dy;
                    const double pu = kx * xu + ly * ye - om * t;
                    const double pv = kx * xe + ly * yv - om * t;
                    const double pe = kx * xe + ly * ye - om * t;
                    const int cc = (k * g.ny + j) * g.nx + i;
                    s.u[cc] = amp * (om * kx * std::cos(pu) - c.f0 * ly * std::sin(pu)) * cz;
                    s.v[cc] = amp * (om * ly * std::cos(pv) + c.f0 * kx * std::sin(pv)) * cz;
                    s.b[cc] = -c.g * c.eta0 * m * std::cos(pe) * sz;
                }
            }
        }
    } else {
        std::fprintf(stderr, "FATAL: unknown 3D case %s\n", c.case_name.c_str());
        std::exit(1);
    }
}

/* --------------------------------------------------------------- stepper */
struct Stepper3 {
    const RunConfig& c;
    const Grid3& g;
    PcgDevice pcg;
    HelmholtzMG mg;
    bool use_mg = false;
    double h_eff = 0.0;
    long tridiag_solves = 0;
    double *nu = nullptr, *kap = nullptr;
    double *msub = nullptr, *mdiag = nullptr, *msup = nullptr;
    double *bsub = nullptr, *bdiag = nullptr, *bsup = nullptr;
    double *q = nullptr, *cstar = nullptr, *ones = nullptr;
    double *phi = nullptr, *pgx = nullptr, *pgy = nullptr;
    double *avpv = nullptr, *avpu = nullptr, *vcor = nullptr, *ucor = nullptr;
    double *gu = nullptr, *gv = nullptr, *ghu = nullptr, *ghv = nullptr;
    double *dzu = nullptr, *dzv = nullptr, *dzb = nullptr;
    double *uit = nullptr, *vit = nullptr, *brhs = nullptr;
    double *div3 = nullptr, *w = nullptr, *wc = nullptr;
    double *gx2 = nullptr, *gy2 = nullptr, *uint2 = nullptr, *vint2 = nullptr;
    double *divold = nullptr, *divg = nullptr, *rhs2 = nullptr, *etanew = nullptr;
    double *tau_u = nullptr, *tau_v = nullptr, *zero2 = nullptr;
    /* split-explicit barotropic workspace (spec S8) */
    int n_split = 0;
    long baro_substeps = 0;
    double *bu = nullptr, *bv = nullptr, *bfx = nullptr, *bfy = nullptr;
    double *bavv = nullptr, *bavu = nullptr, *bvcor = nullptr, *bucor = nullptr;
    double *bui = nullptr, *bvi = nullptr, *bgxe = nullptr, *bgye = nullptr;
    double *corx = nullptr, *cory = nullptr, *umean = nullptr, *vmean = nullptr;

    Stepper3(const RunConfig& cfg, const Grid3& grid) : c(cfg), g(grid) {
        auto a3 = [&](double** p) { CUDA_CHECK(cudaMalloc(p, g.n3 * sizeof(double))); };
        auto a2 = [&](double** p) { CUDA_CHECK(cudaMalloc(p, g.n2 * sizeof(double))); };
        CUDA_CHECK(cudaMalloc(&nu, (size_t)g.n2 * (g.nz + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&kap, (size_t)g.n2 * (g.nz + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w, (size_t)g.n2 * (g.nz + 1) * sizeof(double)));
        for (double** p : {&msub, &mdiag, &msup, &bsub, &bdiag, &bsup, &q, &cstar,
                           &ones, &phi, &pgx, &pgy, &avpv, &avpu, &vcor, &ucor,
                           &gu, &gv, &ghu, &ghv, &dzu, &dzv, &dzb, &uit, &vit,
                           &brhs, &div3, &wc}) a3(p);
        for (double** p : {&gx2, &gy2, &uint2, &vint2, &divold, &divg, &rhs2,
                           &etanew, &tau_u, &tau_v, &zero2, &bu, &bv, &bfx, &bfy,
                           &bavv, &bavu, &bvcor, &bucor, &bui, &bvi, &bgxe, &bgye,
                           &corx, &cory, &umean, &vmean}) a2(p);

        const int nfaces = g.n2 * (g.nz + 1);
        k3_fill<<<(nfaces + BLK - 1) / BLK, BLK>>>(nu, nfaces, c.nu);
        k3_fill<<<(nfaces + BLK - 1) / BLK, BLK>>>(kap, nfaces, c.kappa);
        k3_fill<<<g.g1, g.b1>>>(ones, g.n3, 1.0);
        k3_fill<<<(g.n2 + BLK - 1) / BLK, BLK>>>(tau_u, g.n2, c.tau_x / c.rho0);
        k3_fill<<<(g.n2 + BLK - 1) / BLK, BLK>>>(tau_v, g.n2, c.tau_y / c.rho0);
        k3_fill<<<(g.n2 + BLK - 1) / BLK, BLK>>>(zero2, g.n2, 0.0);

        const double fac = c.theta_v * c.dt / (g.dz * g.dz);
        const double drag_term = c.theta_v * c.dt * c.bottom_drag / g.dz;
        k3_diffusion_coeffs<<<g.g3, g.b3>>>(nu, msub, mdiag, msup, g.nx, g.ny, g.nz,
                                            fac, drag_term);
        k3_diffusion_coeffs<<<g.g3, g.b3>>>(kap, bsub, bdiag, bsup, g.nx, g.ny, g.nz,
                                            fac, 0.0);
        k3_thomas_column<<<g.gcol, g.b2>>>(msub, mdiag, msup, ones, q, cstar,
                                           g.nx, g.ny, g.nz);
        k3_depth_integral<<<g.gcol, g.b2>>>(q, uint2, g.nx, g.ny, g.nz, g.dz);
        tridiag_solves = 1;

        std::vector<double> h(g.n2);
        CUDA_CHECK(cudaMemcpy(h.data(), uint2, g.n2 * sizeof(double), cudaMemcpyDeviceToHost));
        const double lo = *std::min_element(h.begin(), h.end());
        const double hi = *std::max_element(h.begin(), h.end());
        if (hi - lo > 1e-10 * std::max(1.0, std::fabs(hi))) {
            std::fprintf(stderr, "FATAL: spec v0.2 assumes a horizontally uniform "
                                 "effective depth; spread=%.3e\n", hi - lo);
            std::exit(1);
        }
        h_eff = 0.0;
        for (double x : h) h_eff += x;
        h_eff /= g.n2;

        if (c.scheme_name == "theta") {
            const double coef = c.g * h_eff * c.theta * c.theta * c.dt * c.dt;
            use_mg = (c.solver_kind == "multigrid");
            if (use_mg) {
                if (g.nx % 2 || g.ny % 2) {
                    std::fprintf(stderr, "FATAL: multigrid needs even nx and ny for a "
                                         "consistent red-black colouring\n");
                    std::exit(1);
                }
                mg.init(g.nx, g.ny, g.dx, g.dy, coef, c.rtol, 200);
            } else {
                pcg.init(g.nx, g.ny, g.dx, g.dy, coef, c.rtol, c.max_iter,
                         c.pcg_check_every, c.pcg_sync);
            }
        } else if (c.scheme_name == "split_explicit") {
            const double cwave = std::sqrt(c.g * c.h0);
            const double dt_baro = 2.0 / (cwave * std::sqrt(4.0 / (g.dx * g.dx)
                                                          + 4.0 / (g.dy * g.dy)));
            n_split = c.n_split > 0 ? c.n_split
                                    : std::max(1, int(std::ceil(1.2 * c.dt / dt_baro)));
            if (c.dt / n_split > dt_baro) {
                std::fprintf(stderr, "FATAL: n_split=%d leaves a barotropic substep "
                                     "of %.1f s above the limit %.1f s\n",
                             n_split, c.dt / n_split, dt_baro);
                std::exit(1);
            }
        }
    }

    void barotropic(double* eta) {
        const double ddt = c.dt / n_split, f = c.f0, tc = c.theta_cor;
        const double gh = c.g * c.h0;
        const int n = g.n2;
        const dim3 gb((n + BLK - 1) / BLK), bb(BLK);
        for (int m = 0; m < n_split; ++m) {
            k2_avg_vu<<<g.g2, g.b2>>>(bv, bavv, g.nx, g.ny);
            k2_avg_uv<<<g.g2, g.b2>>>(bu, bavu, g.nx, g.ny);
            k2_gradx<<<g.g2, g.b2>>>(eta, bgxe, g.nx, g.ny, g.dx);
            k2_grady<<<g.g2, g.b2>>>(eta, bgye, g.nx, g.ny, g.dy);
            k3_copy<<<gb, bb>>>(bu, bui, n);
            k3_copy<<<gb, bb>>>(bv, bvi, n);
            for (int q = 0; q < c.n_picard; ++q) {
                k2_avg_vu<<<g.g2, g.b2>>>(bvi, bvcor, g.nx, g.ny);
                k2_avg_uv<<<g.g2, g.b2>>>(bui, bucor, g.nx, g.ny);
                k2_baro_mom<<<gb, bb>>>(bu, bv, bavv, bavu, bvcor, bucor, bgxe, bgye,
                                        bfx, bfy, bui, bvi, n, ddt, f, tc, gh);
            }
            k3_copy<<<gb, bb>>>(bui, bu, n);
            k3_copy<<<gb, bb>>>(bvi, bv, n);
            k2_div<<<g.g2, g.b2>>>(bu, bv, divg, g.nx, g.ny, g.dx, g.dy);
            k2_eta_dec<<<gb, bb>>>(eta, divg, n, ddt);
        }
        baro_substeps += n_split;
    }

    void step_split(double* u, double* v, double* b, double* eta) {
        const double f = c.f0, dt = c.dt, tc = c.theta_cor, thv = c.theta_v;
        const double dz = g.dz;
        const dim3 gb((g.n2 + BLK - 1) / BLK), bb(BLK);

        k3_phi<<<g.gcol, g.b2>>>(b, phi, g.nx, g.ny, g.nz, dz);
        k3_gradx_u<<<g.g3, g.b3>>>(phi, pgx, g.nx, g.ny, g.nz, g.dx);
        k3_grady_v<<<g.g3, g.b3>>>(phi, pgy, g.nx, g.ny, g.nz, g.dy);
        k3_avg_v_to_u<<<g.g3, g.b3>>>(v, avpv, g.nx, g.ny, g.nz);
        k3_avg_u_to_v<<<g.g3, g.b3>>>(u, avpu, g.nx, g.ny, g.nz);
        k3_depth_integral<<<g.gcol, g.b2>>>(u, bu, g.nx, g.ny, g.nz, dz);
        k3_depth_integral<<<g.gcol, g.b2>>>(v, bv, g.nx, g.ny, g.nz, dz);
        k3_apply_diffusion<<<g.g3, g.b3>>>(u, nu, tau_u, dzu, g.nx, g.ny, g.nz,
                                           dz, c.bottom_drag);
        k3_apply_diffusion<<<g.g3, g.b3>>>(v, nu, tau_v, dzv, g.nx, g.ny, g.nz,
                                           dz, c.bottom_drag);
        k3_copy<<<g.g1, g.b1>>>(u, uit, g.n3);
        k3_copy<<<g.g1, g.b1>>>(v, vit, g.n3);

        for (int m = 1; m <= c.n_picard; ++m) {
            k3_avg_v_to_u<<<g.g3, g.b3>>>(vit, vcor, g.nx, g.ny, g.nz);
            k3_avg_u_to_v<<<g.g3, g.b3>>>(uit, ucor, g.nx, g.ny, g.nz);
            /* c1 = 0 drops the barotropic pressure gradient from the baroclinic
             * predictor; the barotropic system carries it instead. */
            k3_predictor<<<g.g3, g.b3>>>(u, v, avpv, avpu, vcor, ucor, pgx, pgy,
                                         zero2, zero2, dzu, dzv, tau_u, tau_v,
                                         gu, gv, g.nx, g.ny, g.nz, dt, f, tc,
                                         0.0, thv, dz);
            k3_thomas_column<<<g.gcol, g.b2>>>(msub, mdiag, msup, gu, ghu, cstar,
                                               g.nx, g.ny, g.nz);
            k3_thomas_column<<<g.gcol, g.b2>>>(msub, mdiag, msup, gv, ghv, cstar,
                                               g.nx, g.ny, g.nz);
            tridiag_solves += 2;
            k3_copy<<<g.g1, g.b1>>>(ghu, uit, g.n3);
            k3_copy<<<g.g1, g.b1>>>(ghv, vit, g.n3);
        }

        k3_coriolis_int<<<g.gcol, g.b2>>>(avpv, avpu, vcor, ucor, corx, cory,
                                          g.nx, g.ny, g.nz, f, tc, dz);
        k3_depth_integral<<<g.gcol, g.b2>>>(ghu, uint2, g.nx, g.ny, g.nz, dz);
        k3_depth_integral<<<g.gcol, g.b2>>>(ghv, vint2, g.nx, g.ny, g.nz, dz);
        k2_baro_forcing<<<gb, bb>>>(uint2, vint2, bu, bv, corx, cory, bfx, bfy,
                                    g.n2, dt);

        barotropic(eta);

        k3_depth_integral<<<g.gcol, g.b2>>>(ghu, umean, g.nx, g.ny, g.nz, 1.0 / g.nz);
        k3_depth_integral<<<g.gcol, g.b2>>>(ghv, vmean, g.nx, g.ny, g.nz, 1.0 / g.nz);
        k3_split_correct<<<g.g3, g.b3>>>(ghu, ghv, umean, vmean, bu, bv, uit, vit,
                                         g.nx, g.ny, g.nz, 1.0 / c.h0);

        k3_div<<<g.g3, g.b3>>>(uit, vit, div3, g.nx, g.ny, g.nz, g.dx, g.dy);
        k3_w_from_div<<<g.gcol, g.b2>>>(div3, w, g.nx, g.ny, g.nz, dz);
        k3_w_centres<<<g.g3, g.b3>>>(w, wc, g.nx, g.ny, g.nz);
        k3_apply_diffusion<<<g.g3, g.b3>>>(b, kap, zero2, dzb, g.nx, g.ny, g.nz, dz, 0.0);
        k3_b_rhs<<<g.g3, g.b3>>>(b, wc, dzb, brhs, g.nx, g.ny, g.nz, dt, c.n2, thv);
        k3_thomas_column<<<g.gcol, g.b2>>>(bsub, bdiag, bsup, brhs, b, cstar,
                                           g.nx, g.ny, g.nz);
        ++tridiag_solves;
        k3_copy<<<g.g1, g.b1>>>(uit, u, g.n3);
        k3_copy<<<g.g1, g.b1>>>(vit, v, g.n3);
    }

    void step(double* u, double* v, double* b, double* eta) {
        if (c.scheme_name == "split_explicit") { step_split(u, v, b, eta); return; }
        /* fb applies the full explicit surface gradient; theta is the theta scheme's (N14) */
        const double f = c.f0, dt = c.dt, th = (c.scheme_name == "theta") ? c.theta : 0.0, tc = c.theta_cor;
        const double thv = c.theta_v, dz = g.dz;
        const double c1 = c.g * dt * (1.0 - th);
        const double c2f = c.g * dt * th;

        k3_phi<<<g.gcol, g.b2>>>(b, phi, g.nx, g.ny, g.nz, dz);
        k3_gradx_u<<<g.g3, g.b3>>>(phi, pgx, g.nx, g.ny, g.nz, g.dx);
        k3_grady_v<<<g.g3, g.b3>>>(phi, pgy, g.nx, g.ny, g.nz, g.dy);
        k2_gradx<<<g.g2, g.b2>>>(eta, gx2, g.nx, g.ny, g.dx);
        k2_grady<<<g.g2, g.b2>>>(eta, gy2, g.nx, g.ny, g.dy);
        k3_avg_v_to_u<<<g.g3, g.b3>>>(v, avpv, g.nx, g.ny, g.nz);
        k3_avg_u_to_v<<<g.g3, g.b3>>>(u, avpu, g.nx, g.ny, g.nz);
        k3_depth_integral<<<g.gcol, g.b2>>>(u, uint2, g.nx, g.ny, g.nz, dz);
        k3_depth_integral<<<g.gcol, g.b2>>>(v, vint2, g.nx, g.ny, g.nz, dz);
        k2_div<<<g.g2, g.b2>>>(uint2, vint2, divold, g.nx, g.ny, g.dx, g.dy);
        k3_apply_diffusion<<<g.g3, g.b3>>>(u, nu, tau_u, dzu, g.nx, g.ny, g.nz,
                                           dz, c.bottom_drag);
        k3_apply_diffusion<<<g.g3, g.b3>>>(v, nu, tau_v, dzv, g.nx, g.ny, g.nz,
                                           dz, c.bottom_drag);
        k3_copy<<<g.g1, g.b1>>>(u, uit, g.n3);
        k3_copy<<<g.g1, g.b1>>>(v, vit, g.n3);

        for (int m = 1; m <= c.n_picard; ++m) {
            k3_avg_v_to_u<<<g.g3, g.b3>>>(vit, vcor, g.nx, g.ny, g.nz);
            k3_avg_u_to_v<<<g.g3, g.b3>>>(uit, ucor, g.nx, g.ny, g.nz);
            k3_predictor<<<g.g3, g.b3>>>(u, v, avpv, avpu, vcor, ucor, pgx, pgy,
                                         gx2, gy2, dzu, dzv, tau_u, tau_v, gu, gv,
                                         g.nx, g.ny, g.nz, dt, f, tc, c1, thv, dz);
            k3_thomas_column<<<g.gcol, g.b2>>>(msub, mdiag, msup, gu, ghu, cstar,
                                               g.nx, g.ny, g.nz);
            k3_thomas_column<<<g.gcol, g.b2>>>(msub, mdiag, msup, gv, ghv, cstar,
                                               g.nx, g.ny, g.nz);
            tridiag_solves += 2;

            if (c.scheme_name == "theta") {
                k3_depth_integral<<<g.gcol, g.b2>>>(ghu, uint2, g.nx, g.ny, g.nz, dz);
                k3_depth_integral<<<g.gcol, g.b2>>>(ghv, vint2, g.nx, g.ny, g.nz, dz);
                k2_div<<<g.g2, g.b2>>>(uint2, vint2, divg, g.nx, g.ny, g.dx, g.dy);
                k2_rhs_eta<<<(g.n2 + BLK - 1) / BLK, BLK>>>(eta, divold, divg, rhs2,
                                                           g.n2, dt, th);
                if (use_mg) mg.solve(rhs2, etanew); else pcg.solve(rhs2, etanew);
                k2_gradx<<<g.g2, g.b2>>>(etanew, gx2, g.nx, g.ny, g.dx);
                k2_grady<<<g.g2, g.b2>>>(etanew, gy2, g.nx, g.ny, g.dy);
                k3_backsub<<<g.g3, g.b3>>>(ghu, ghv, q, gx2, gy2, uit, vit,
                                           g.nx, g.ny, g.nz, c2f);
                if (m < c.n_picard) {
                    k2_gradx<<<g.g2, g.b2>>>(eta, gx2, g.nx, g.ny, g.dx);
                    k2_grady<<<g.g2, g.b2>>>(eta, gy2, g.nx, g.ny, g.dy);
                }
            } else {
                k3_copy<<<g.g1, g.b1>>>(ghu, uit, g.n3);
                k3_copy<<<g.g1, g.b1>>>(ghv, vit, g.n3);
            }
        }

        if (c.scheme_name != "theta") {
            k3_depth_integral<<<g.gcol, g.b2>>>(uit, uint2, g.nx, g.ny, g.nz, dz);
            k3_depth_integral<<<g.gcol, g.b2>>>(vit, vint2, g.nx, g.ny, g.nz, dz);
            k2_div<<<g.g2, g.b2>>>(uint2, vint2, divg, g.nx, g.ny, g.dx, g.dy);
            k2_eta_fb<<<(g.n2 + BLK - 1) / BLK, BLK>>>(eta, divg, etanew, g.n2, dt);
        }

        k3_div<<<g.g3, g.b3>>>(uit, vit, div3, g.nx, g.ny, g.nz, g.dx, g.dy);
        k3_w_from_div<<<g.gcol, g.b2>>>(div3, w, g.nx, g.ny, g.nz, dz);
        k3_w_centres<<<g.g3, g.b3>>>(w, wc, g.nx, g.ny, g.nz);
        k3_apply_diffusion<<<g.g3, g.b3>>>(b, kap, zero2, dzb, g.nx, g.ny, g.nz, dz, 0.0);
        k3_b_rhs<<<g.g3, g.b3>>>(b, wc, dzb, brhs, g.nx, g.ny, g.nz, dt, c.n2, thv);
        k3_thomas_column<<<g.gcol, g.b2>>>(bsub, bdiag, bsup, brhs, b, cstar,
                                           g.nx, g.ny, g.nz);
        ++tridiag_solves;

        k3_copy<<<g.g1, g.b1>>>(uit, u, g.n3);
        k3_copy<<<g.g1, g.b1>>>(vit, v, g.n3);
        k3_copy<<<(g.n2 + BLK - 1) / BLK, BLK>>>(etanew, eta, g.n2);
    }
};

/* ---------------------------------------------------------- diagnostics */
static double rel_l2(const std::vector<double>& a, const std::vector<double>& b,
                     double state_scale) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    num = std::sqrt(num);
    den = std::sqrt(den);
    /* Where the reference field is identically zero (eta in baroclinic_igw)
     * only round-off remains; normalise by the state instead (docs/21 S1.1). */
    if (den > 1e-11 * state_scale) return num / den;
    return state_scale > 0.0 ? num / state_scale : num;
}

static double norm2(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x * x;
    return std::sqrt(s);
}

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static std::string jnum(double x) {
    if (!std::isfinite(x)) return "null";
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.16E", x);
    return std::string(buf);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: cfd_exp3d_cuda <namelist>\n");
        return 1;
    }
    const RunConfig c = read_config(argv[1]);
    if (c.nz < 1) {
        std::fprintf(stderr, "FATAL: cfd_exp3d_cuda needs nz >= 1\n");
        return 1;
    }
    const Grid3 g(c);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    Host3 h0(g.n3, g.n2), hf(g.n3, g.n2), he(g.n3, g.n2);
    case_fields3d(c, g, 0.0, h0);
    case_fields3d(c, g, c.t_final, he);

    double *d_u, *d_v, *d_b, *d_eta;
    CUDA_CHECK(cudaMalloc(&d_u, g.n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v, g.n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, g.n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eta, g.n2 * sizeof(double)));
    size_t free_before = 0, total_dev = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_dev));
    Stepper3 st(c, g);
    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_dev));
    /* Measured device footprint, not a count of declared arrays: what decides
     * the largest grid that actually fits. */
    const double dev_bytes = double(free_before - free_after);
    const double bytes_per_cell = dev_bytes / double(g.n3);

    auto reset = [&]() {
        CUDA_CHECK(cudaMemcpy(d_u, h0.u.data(), g.n3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h0.v.data(), g.n3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h0.b.data(), g.n3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_eta, h0.eta.data(), g.n2 * sizeof(double), cudaMemcpyHostToDevice));
    };

    std::printf("cuda3d: case=%s nx=%d nz=%d scheme=%s dt=%12.5E steps=%d "
                "pcg_sync=%s check_every=%d gpu=%s\n",
                c.case_name.c_str(), c.nx, c.nz, c.scheme_name.c_str(), c.dt,
                c.n_steps, c.pcg_sync.c_str(), c.pcg_check_every, prop.name);

    for (int rep = 0; rep < c.n_warmup; ++rep) {
        reset();
        for (int s = 0; s < c.n_steps; ++s) st.step(d_u, d_v, d_b, d_eta);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    st.pcg.total_iterations = 0;
    st.pcg.failures = 0;
    st.mg.total_iterations = 0;
    st.baro_substeps = 0;

    std::vector<double> samples;
    for (int rep = 0; rep < c.n_repeat; ++rep) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < c.n_steps; ++s) st.step(d_u, d_v, d_b, d_eta);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    const long iters = (st.use_mg ? st.mg.total_iterations
                                  : st.pcg.total_iterations) / std::max(1, c.n_repeat);
    const long subs = st.baro_substeps / std::max(1, c.n_repeat);

    CUDA_CHECK(cudaMemcpy(hf.u.data(), d_u, g.n3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hf.v.data(), d_v, g.n3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hf.b.data(), d_b, g.n3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hf.eta.data(), d_eta, g.n2 * sizeof(double), cudaMemcpyDeviceToHost));

    const double scale = std::max(std::max(norm2(he.u), norm2(he.v)),
                                  std::max(norm2(he.b), norm2(he.eta)));
    const double l2u = rel_l2(hf.u, he.u, scale);
    const double l2b = rel_l2(hf.b, he.b, scale);
    const double l2e = rel_l2(hf.eta, he.eta, scale);
    const double t_med = median_of(samples);
    std::vector<double> dev;
    for (double s : samples) dev.push_back(std::fabs(s - t_med));
    const double t_mad = median_of(dev);

    std::printf("  L2(u)=%12.5E  L2(b)=%12.5E  L2(eta)=%12.5E\n", l2u, l2b, l2e);
    std::printf("  wall median=%12.5E s  mad=%12.5E s  solver_iters/run=%ld\n",
                t_med, t_mad, iters);
    std::printf("  device memory=%.1f MiB  (%.0f B/cell)  free after=%.1f GiB\n",
                dev_bytes / 1048576.0, bytes_per_cell, free_after / 1073741824.0);

    const std::string bin = c.out_prefix + "_state3d.bin";
    if (FILE* f = std::fopen(bin.c_str(), "wb")) {
        std::fwrite(hf.eta.data(), sizeof(double), g.n2, f);
        std::fwrite(hf.u.data(), sizeof(double), g.n3, f);
        std::fwrite(hf.v.data(), sizeof(double), g.n3, f);
        std::fwrite(hf.b.data(), sizeof(double), g.n3, f);
        std::fclose(f);
    }
    const std::string mj = c.out_prefix + "_metrics.json";
    if (FILE* f = std::fopen(mj.c_str(), "w")) {
        std::fprintf(f,
            "{\n  \"backend\": \"cuda3d\",\n  \"gpu\": \"%s\",\n"
            "  \"nx\": %d,\n  \"ny\": %d,\n  \"nz\": %d,\n  \"cells\": %d,\n"
            "  \"n_steps\": %d,\n  \"scheme\": \"%s\",\n  \"theta\": %s,\n"
            "  \"theta_v\": %s,\n  \"dt\": %s,\n  \"h_eff\": %s,\n"
            "  \"pcg_sync\": \"%s\",\n  \"pcg_check_every\": %d,\n"
            "  \"n_split\": %d,\n  \"barotropic_substeps\": %ld,\n"
            "  \"solver\": \"%s\",\n  \"device_bytes\": %.0f,\n"
            "  \"bytes_per_cell\": %s,\n"
            "  \"solver_iterations\": %ld,\n  \"tridiagonal_solves\": %ld,\n"
            "  \"l2_rel_u\": %s,\n  \"l2_rel_b\": %s,\n  \"l2_rel_eta\": %s,\n"
            "  \"wall_s\": %s,\n  \"wall_mad_s\": %s,\n"
            "  \"n_repeat\": %d,\n  \"omp_num_threads\": 0\n}\n",
            prop.name, g.nx, g.ny, g.nz, g.n3, c.n_steps, c.scheme_name.c_str(),
            jnum(c.theta).c_str(), jnum(c.theta_v).c_str(), jnum(c.dt).c_str(),
            jnum(st.h_eff).c_str(), c.pcg_sync.c_str(), c.pcg_check_every,
            st.n_split, subs, c.solver_kind.c_str(), dev_bytes,
            jnum(bytes_per_cell).c_str(), iters, st.tridiag_solves, jnum(l2u).c_str(), jnum(l2b).c_str(),
            jnum(l2e).c_str(), jnum(t_med).c_str(), jnum(t_mad).c_str(), c.n_repeat);
        std::fclose(f);
    }
    return 0;
}
