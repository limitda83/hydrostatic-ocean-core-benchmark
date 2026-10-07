/************************************************************************
 *  Program: cfd_exp_cuda (cuda backend)                                *
 *  Description: Native CUDA C++ implementation of the 2D linear        *
 *               rotating shallow water solver of                       *
 *               docs/03_discretization_spec.md. Reads the same         *
 *               namelist and writes the same state dump and metrics    *
 *               as the Fortran backends, so tools/compare_backends.py  *
 *               gates it unchanged.                                    *
 *  Pipeline: toml2nml.py -> cfd_exp_cuda -> compare_backends.py        *
 ************************************************************************/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "config.hpp"
#include "kernels.cuh"

static constexpr int BLOCK = 256;
static constexpr double PI = 3.14159265358979323846;

/* ------------------------------------------------------------- geometry */
struct Grid {
    int nx, ny, n;
    double dx, dy, cell_area;
    dim3 block2d, grid2d;
    dim3 block1d, grid1d;

    Grid(const RunConfig& c)
        : nx(c.nx), ny(c.ny), n(c.nx * c.ny),
          dx(c.lx / c.nx), dy(c.ly / c.ny), cell_area(dx * dy),
          block2d(32, 8), block1d(BLOCK) {
        grid2d = dim3((nx + block2d.x - 1) / block2d.x, (ny + block2d.y - 1) / block2d.y);
        grid1d = dim3((n + BLOCK - 1) / BLOCK);
    }
};

/* Host-side initial and exact fields. Built once, so they stay on the host
 * and keep the device code focused on the time loop being benchmarked. */
struct HostState {
    std::vector<double> u, v, eta;
    explicit HostState(int n) : u(n), v(n), eta(n) {}
};

static void case_fields(const RunConfig& c, const Grid& g, double t, HostState& s) {
    const int nx = g.nx, ny = g.ny;
    auto coord = [&](int i, int j, double ox, double oy, double& x, double& y) {
        x = (i + ox) * g.dx;
        y = (j + oy) * g.dy;
    };
    std::fill(s.u.begin(), s.u.end(), 0.0);
    std::fill(s.v.begin(), s.v.end(), 0.0);
    std::fill(s.eta.begin(), s.eta.end(), 0.0);

    if (c.case_name == "igw" || c.case_name == "igw_broadband") {
        struct Mode { double k, l, kappa2, omega, amp, phase; };
        std::vector<Mode> modes;
        if (c.case_name == "igw") {
            const double k = 2.0 * PI * c.mode_x / c.lx;
            const double l = 2.0 * PI * c.mode_y / c.ly;
            const double kap2 = k * k + l * l;
            modes.push_back({k, l, kap2,
                             std::sqrt(c.f0 * c.f0 + c.g * c.h0 * kap2), c.eta0, 0.0});
        } else {
            double raw_sq = 0.0;
            for (int m = 1; m <= c.n_modes; ++m) {
                for (int nn = 1; nn <= c.n_modes; ++nn) {
                    const double k = 2.0 * PI * m / c.lx;
                    const double l = 2.0 * PI * nn / c.ly;
                    const double kap2 = k * k + l * l;
                    const double raw = std::pow(double(m * m + nn * nn), -0.5 * c.slope);
                    raw_sq += raw * raw;
                    modes.push_back({k, l, kap2,
                                     std::sqrt(c.f0 * c.f0 + c.g * c.h0 * kap2), raw,
                                     2.0 * PI * ((m * 37 + nn * 17) % 101) / 101.0});
                }
            }
            const double norm = c.eta0 / std::sqrt(raw_sq / 2.0);
            for (auto& m : modes) m.amp *= norm;
        }
        for (const auto& m : modes) {
            const double scale = m.amp / (c.h0 * m.kappa2);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double xe, ye, xu, yu, xv, yv;
                    coord(i, j, 0.5, 0.5, xe, ye);
                    coord(i, j, 1.0, 0.5, xu, yu);
                    coord(i, j, 0.5, 1.0, xv, yv);
                    const double pe = m.k * xe + m.l * ye - m.omega * t + m.phase;
                    const double pu = m.k * xu + m.l * yu - m.omega * t + m.phase;
                    const double pv = m.k * xv + m.l * yv - m.omega * t + m.phase;
                    const int cix = j * nx + i;
                    s.eta[cix] += m.amp * std::cos(pe);
                    s.u[cix] += scale * (m.omega * m.k * std::cos(pu) -
                                         c.f0 * m.l * std::sin(pu));
                    s.v[cix] += scale * (m.omega * m.l * std::cos(pv) +
                                         c.f0 * m.k * std::sin(pv));
                }
            }
        }
    } else if (c.case_name == "geo_balance") {
        /* Exact steady state of the DISCRETE operators, from the Fourier
         * symbols in docs/03_discretization_spec.md S5-V2. */
        const double tx = 2.0 * PI * c.mode_x / nx;
        const double ty = 2.0 * PI * c.mode_y / ny;
        const double fac = 4.0 * c.g / c.f0;
        const double cx = std::cos(tx), sx = std::sin(tx);
        const double cy = std::cos(ty), sy = std::sin(ty);
        /* v_hat =  fac (e^{i tx} - 1) / (dx (1 + e^{-i ty})(1 + e^{i tx})) eta0
         * u_hat = -fac (e^{i ty} - 1) / (dy (1 + e^{-i tx})(1 + e^{i ty})) eta0 */
        auto cmul = [](double ar, double ai, double br, double bi, double& r, double& im) {
            r = ar * br - ai * bi;
            im = ar * bi + ai * br;
        };
        auto cdiv = [](double ar, double ai, double br, double bi, double& r, double& im) {
            const double d = br * br + bi * bi;
            r = (ar * br + ai * bi) / d;
            im = (ai * br - ar * bi) / d;
        };
        double dr, di, nr, ni, vr, vi, ur, ui;
        cmul(1.0 + cy, -sy, 1.0 + cx, sx, dr, di);          /* (1+e^{-ity})(1+e^{itx}) */
        cdiv(fac * (cx - 1.0) * c.eta0, fac * sx * c.eta0, g.dx * dr, g.dx * di, vr, vi);
        cmul(1.0 + cx, -sx, 1.0 + cy, sy, nr, ni);          /* (1+e^{-itx})(1+e^{ity}) */
        cdiv(-fac * (cy - 1.0) * c.eta0, -fac * sy * c.eta0, g.dy * nr, g.dy * ni, ur, ui);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double ph = tx * i + ty * j;
                const double cph = std::cos(ph), sph = std::sin(ph);
                const int cix = j * nx + i;
                s.eta[cix] = c.eta0 * cph;
                s.u[cix] = ur * cph - ui * sph;
                s.v[cix] = vr * cph - vi * sph;
            }
        }
    } else {
        std::fprintf(stderr, "FATAL: unknown case %s "
                             "(implemented: igw|igw_broadband|geo_balance)\n",
                     c.case_name.c_str());
        std::exit(1);
    }
}

/* --------------------------------------------------------------- solver */
struct Pcg {
    double coef = 0.0, inv_diag = 1.0, rtol = 1e-12;
    int max_iter = 2000;
    long total_iterations = 0;
    int failures = 0;
    double *r = nullptr, *z = nullptr, *p = nullptr, *ap = nullptr;
    double *wx = nullptr, *wy = nullptr, *lap = nullptr;
    double *partial = nullptr, *d_scalar = nullptr;
    int n_partial = 0;

    void init(const Grid& g, const RunConfig& c, double dt) {
        coef = c.g * c.h0 * c.theta * c.theta * dt * dt;
        inv_diag = 1.0 / (1.0 + coef * (2.0 / (g.dx * g.dx) + 2.0 / (g.dy * g.dy)));
        rtol = c.rtol;
        max_iter = c.max_iter;
        n_partial = std::min(1024, int(g.grid1d.x));
        for (double** q : {&r, &z, &p, &ap, &wx, &wy, &lap}) {
            CUDA_CHECK(cudaMalloc(q, g.n * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&partial, n_partial * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(double)));
    }

    double dot(const double* a, const double* b, int n) {
        k_dot_partial<BLOCK><<<n_partial, BLOCK>>>(a, b, partial, n);
        k_dot_final<BLOCK><<<1, BLOCK>>>(partial, d_scalar, n_partial);
        double h = 0.0;
        CUDA_CHECK(cudaMemcpy(&h, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }

    void apply(const Grid& g, const double* x, double* ax) {
        k_gradx_u<<<g.grid2d, g.block2d>>>(x, wx, g.nx, g.ny, g.dx);
        k_grady_v<<<g.grid2d, g.block2d>>>(x, wy, g.nx, g.ny, g.dy);
        k_div<<<g.grid2d, g.block2d>>>(wx, wy, lap, g.nx, g.ny, g.dx, g.dy);
        k_apply<<<g.grid1d, g.block1d>>>(x, lap, ax, g.n, coef);
    }

    void solve(const Grid& g, const double* rhs, double* x) {
        k_zero<<<g.grid1d, g.block1d>>>(x, g.n);
        apply(g, x, ap);
        const double norm_b = std::sqrt(dot(rhs, rhs, g.n));
        if (norm_b == 0.0) return;

        k_pcg_init<<<g.grid1d, g.block1d>>>(rhs, ap, r, z, p, g.n, inv_diag);
        double rz = dot(r, z, g.n);
        double residual = 0.0;
        int used = max_iter;
        bool converged = false;

        for (int it = 1; it <= max_iter; ++it) {
            apply(g, p, ap);
            const double alpha = rz / dot(p, ap, g.n);
            k_pcg_update_x_r<<<g.grid1d, g.block1d>>>(x, r, p, ap, g.n, alpha);
            residual = std::sqrt(dot(r, r, g.n)) / norm_b;
            if (residual < rtol) {
                converged = true;
                used = it;
                break;
            }
            k_pcg_precondition<<<g.grid1d, g.block1d>>>(r, z, g.n, inv_diag);
            const double rz_new = dot(r, z, g.n);
            k_pcg_update_p<<<g.grid1d, g.block1d>>>(p, z, g.n, rz_new / rz);
            rz = rz_new;
        }
        total_iterations += used;
        if (!converged) {
            ++failures;
            std::fprintf(stderr, "ERROR: PCG failed to converge: %d iterations, "
                                 "residual=%.5e > rtol=%.5e\n", max_iter, residual, rtol);
        }
    }
};

/* -------------------------------------------------------------- stepper */
struct Stepper {
    const RunConfig& c;
    const Grid& g;
    double dt;
    bool is_theta;
    Pcg pcg;
    double *gx = nullptr, *gy = nullptr, *div_old = nullptr, *tmp = nullptr;
    double *av_prev_v = nullptr, *av_prev_u = nullptr;
    double *v_cor = nullptr, *u_cor = nullptr;
    double *gu = nullptr, *gv = nullptr, *rhs = nullptr;
    double *u_it = nullptr, *v_it = nullptr, *eta_new = nullptr;

    Stepper(const RunConfig& cfg, const Grid& grid, double step)
        : c(cfg), g(grid), dt(step), is_theta(cfg.scheme_name == "theta") {
        for (double** q : {&gx, &gy, &div_old, &tmp, &av_prev_v, &av_prev_u,
                           &v_cor, &u_cor, &gu, &gv, &rhs, &u_it, &v_it, &eta_new}) {
            CUDA_CHECK(cudaMalloc(q, g.n * sizeof(double)));
        }
        if (is_theta) {
            if (c.solver_kind != "pcg_jacobi") {
                std::fprintf(stderr, "FATAL: the CUDA backend implements "
                                     "solver_kind=pcg_jacobi only; got %s\n",
                             c.solver_kind.c_str());
                std::exit(1);
            }
            pcg.init(g, c, dt);
        }
    }

    void step(double* u, double* v, double* eta) {
        const double f = c.f0, h = c.h0, th = c.theta, tc = c.theta_cor;
        const double c1 = c.g * dt * (1.0 - th);
        const double c2 = c.g * dt * th;

        k_gradx_u<<<g.grid2d, g.block2d>>>(eta, gx, g.nx, g.ny, g.dx);
        k_grady_v<<<g.grid2d, g.block2d>>>(eta, gy, g.nx, g.ny, g.dy);
        /* Depends only on the level-n state, so hoisting it out of the Picard
         * loop leaves the per-element arithmetic identical to the reference. */
        k_avg_v_to_u<<<g.grid2d, g.block2d>>>(v, av_prev_v, g.nx, g.ny);
        k_avg_u_to_v<<<g.grid2d, g.block2d>>>(u, av_prev_u, g.nx, g.ny);
        CUDA_CHECK(cudaMemcpy(u_it, u, g.n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(v_it, v, g.n * sizeof(double), cudaMemcpyDeviceToDevice));

        if (is_theta) {
            k_div<<<g.grid2d, g.block2d>>>(u, v, div_old, g.nx, g.ny, g.dx, g.dy);
            for (int m = 1; m <= c.n_picard; ++m) {
                k_avg_v_to_u<<<g.grid2d, g.block2d>>>(v_it, v_cor, g.nx, g.ny);
                k_avg_u_to_v<<<g.grid2d, g.block2d>>>(u_it, u_cor, g.nx, g.ny);
                k_predictor<<<g.grid2d, g.block2d>>>(u, v, av_prev_v, av_prev_u, v_cor,
                                                     u_cor, gx, gy, gu, gv,
                                                     g.nx, g.ny, dt, f, tc, c1);
                k_div<<<g.grid2d, g.block2d>>>(gu, gv, tmp, g.nx, g.ny, g.dx, g.dy);
                k_rhs<<<g.grid1d, g.block1d>>>(eta, div_old, tmp, rhs, g.n, dt, h, th);
                pcg.solve(g, rhs, eta_new);
                k_gradx_u<<<g.grid2d, g.block2d>>>(eta_new, gx, g.nx, g.ny, g.dx);
                k_grady_v<<<g.grid2d, g.block2d>>>(eta_new, gy, g.nx, g.ny, g.dy);
                k_back_substitute<<<g.grid2d, g.block2d>>>(gu, gv, gx, gy, u_it, v_it,
                                                           g.nx, g.ny, c2);
                if (m < c.n_picard) {
                    k_gradx_u<<<g.grid2d, g.block2d>>>(eta, gx, g.nx, g.ny, g.dx);
                    k_grady_v<<<g.grid2d, g.block2d>>>(eta, gy, g.nx, g.ny, g.dy);
                }
            }
            k_commit<<<g.grid1d, g.block1d>>>(u_it, v_it, eta_new, u, v, eta, g.n);
        } else {
            for (int m = 1; m <= c.n_picard; ++m) {
                k_avg_v_to_u<<<g.grid2d, g.block2d>>>(v_it, v_cor, g.nx, g.ny);
                k_avg_u_to_v<<<g.grid2d, g.block2d>>>(u_it, u_cor, g.nx, g.ny);
                k_fb_momentum<<<g.grid2d, g.block2d>>>(u, v, av_prev_v, av_prev_u, v_cor,
                                                       u_cor, gx, gy, u_it, v_it,
                                                       g.nx, g.ny, dt, f, tc, c.g);
            }
            k_div<<<g.grid2d, g.block2d>>>(u_it, v_it, tmp, g.nx, g.ny, g.dx, g.dy);
            k_fb_continuity<<<g.grid1d, g.block1d>>>(eta, tmp, g.n, dt, h);
            CUDA_CHECK(cudaMemcpy(u, u_it, g.n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(v, v_it, g.n * sizeof(double), cudaMemcpyDeviceToDevice));
        }
    }
};

/* ---------------------------------------------------------- diagnostics */
static double l2_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num);
}

static double linf_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::fabs(a[i] - b[i]));
        den = std::max(den, std::fabs(b[i]));
    }
    return den > 0.0 ? num / den : num;
}

static double total_energy(const HostState& s, const Grid& g, double h, double gg) {
    double e = 0.0;
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            const int c = j * g.nx + i;
            const int im = j * g.nx + (i - 1 + g.nx) % g.nx;
            const int jm = ((j - 1 + g.ny) % g.ny) * g.nx + i;
            const double u2 = 0.5 * (s.u[c] * s.u[c] + s.u[im] * s.u[im]);
            const double v2 = 0.5 * (s.v[c] * s.v[c] + s.v[jm] * s.v[jm]);
            e += h * (u2 + v2) + gg * s.eta[c] * s.eta[c];
        }
    }
    return 0.5 * e * g.cell_area;
}

/* JSON has no NaN or Infinity literals, so a diverged run must serialise
 * non-finite values as null rather than corrupting the whole record. */
static std::string jnum(double x) {
    if (!std::isfinite(x)) return "null";
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.16E", x);
    return std::string(buf);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ----------------------------------------------------------------- main */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: cfd_exp_cuda <namelist>\n");
        return 1;
    }
    const RunConfig c = read_config(argv[1]);
    const Grid g(c);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    HostState h0(g.n), hf(g.n), he(g.n);
    case_fields(c, g, 0.0, h0);
    case_fields(c, g, c.t_final, he);

    double *d_u, *d_v, *d_eta;
    CUDA_CHECK(cudaMalloc(&d_u, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eta, g.n * sizeof(double)));
    Stepper stepper(c, g, c.dt);

    auto reset = [&]() {
        CUDA_CHECK(cudaMemcpy(d_u, h0.u.data(), g.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h0.v.data(), g.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_eta, h0.eta.data(), g.n * sizeof(double), cudaMemcpyHostToDevice));
    };

    std::printf("cuda: case=%s nx=%d scheme=%s theta=%6.3f dt=%12.5E steps=%d gpu=%s\n",
                c.case_name.c_str(), c.nx, c.scheme_name.c_str(), c.theta, c.dt,
                c.n_steps, prop.name);

    /* Timing protocol R7: discard warm-ups, n_repeat measured runs, device
     * synchronise before stopping the clock (R7-4). */
    for (int rep = 0; rep < c.n_warmup; ++rep) {
        reset();
        for (int s = 0; s < c.n_steps; ++s) stepper.step(d_u, d_v, d_eta);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    stepper.pcg.total_iterations = 0;
    stepper.pcg.failures = 0;

    std::vector<double> samples;
    for (int rep = 0; rep < c.n_repeat; ++rep) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < c.n_steps; ++s) stepper.step(d_u, d_v, d_eta);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    const long iters_per_run = stepper.pcg.total_iterations / std::max(1, c.n_repeat);

    CUDA_CHECK(cudaMemcpy(hf.u.data(), d_u, g.n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hf.v.data(), d_v, g.n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hf.eta.data(), d_eta, g.n * sizeof(double), cudaMemcpyDeviceToHost));

    const double t_med = median(samples);
    std::vector<double> dev;
    for (double s : samples) dev.push_back(std::fabs(s - t_med));
    const double t_mad = median(dev);

    double mass0 = 0.0, mass1 = 0.0, rms_e = 0.0;
    for (int i = 0; i < g.n; ++i) {
        mass0 += h0.eta[i];
        mass1 += hf.eta[i];
        rms_e += he.eta[i] * he.eta[i];
    }
    mass0 *= g.cell_area;
    mass1 *= g.cell_area;
    const double domain_area = g.cell_area * g.n;
    double mass_scale = std::sqrt(rms_e / g.n) * domain_area;
    if (mass_scale == 0.0) mass_scale = 1.0;
    const double e0 = total_energy(h0, g, c.h0, c.g);
    const double e1 = total_energy(hf, g, c.h0, c.g);

    std::printf("  L2(eta)=%12.5E  mass drift=%12.5E  energy drift=%12.5E\n",
                l2_rel(hf.eta, he.eta), (mass1 - mass0) / mass_scale,
                (e1 - e0) / std::fabs(e0));
    std::printf("  wall median=%12.5E s  mad=%12.5E s  pcg_iters/run=%ld\n",
                t_med, t_mad, iters_per_run);

    /* Same stream layout as the Fortran backend: eta, u, v, each n doubles. */
    const std::string bin = c.out_prefix + "_state.bin";
    if (FILE* f = std::fopen(bin.c_str(), "wb")) {
        std::fwrite(hf.eta.data(), sizeof(double), g.n, f);
        std::fwrite(hf.u.data(), sizeof(double), g.n, f);
        std::fwrite(hf.v.data(), sizeof(double), g.n, f);
        std::fclose(f);
    }
    const std::string mj = c.out_prefix + "_metrics.json";
    if (FILE* f = std::fopen(mj.c_str(), "w")) {
        std::fprintf(f,
            "{\n  \"backend\": \"cuda\",\n  \"gpu\": \"%s\",\n"
            "  \"nx\": %d,\n  \"ny\": %d,\n  \"dx\": %s,\n  \"dt\": %s,\n"
            "  \"n_steps\": %d,\n  \"scheme\": \"%s\",\n  \"theta\": %s,\n"
            "  \"n_picard\": %d,\n  \"solver\": \"%s\",\n"
            "  \"solver_iterations\": %ld,\n  \"solver_failures\": %d,\n"
            "  \"l2_rel_eta\": %s,\n  \"l2_rel_u\": %s,\n  \"l2_rel_v\": %s,\n"
            "  \"linf_rel_eta\": %s,\n  \"mass\": %s,\n  \"energy\": %s,\n"
            "  \"mass_drift\": %s,\n  \"energy_drift\": %s,\n"
            "  \"wall_s\": %s,\n  \"wall_mad_s\": %s,\n"
            "  \"n_repeat\": %d,\n  \"n_warmup\": %d,\n  \"omp_num_threads\": 0\n}\n",
            prop.name, g.nx, g.ny, jnum(g.dx).c_str(), jnum(c.dt).c_str(),
            c.n_steps, c.scheme_name.c_str(), jnum(c.theta).c_str(),
            c.n_picard, c.solver_kind.c_str(), iters_per_run,
            stepper.pcg.failures, jnum(l2_rel(hf.eta, he.eta)).c_str(),
            jnum(l2_rel(hf.u, he.u)).c_str(), jnum(l2_rel(hf.v, he.v)).c_str(),
            jnum(linf_rel(hf.eta, he.eta)).c_str(), jnum(mass1).c_str(),
            jnum(e1).c_str(), jnum((mass1 - mass0) / mass_scale).c_str(),
            jnum((e1 - e0) / std::fabs(e0)).c_str(), jnum(t_med).c_str(),
            jnum(t_mad).c_str(), c.n_repeat, c.n_warmup);
        std::fclose(f);
    }
    return 0;
}
