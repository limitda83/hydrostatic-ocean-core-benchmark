/************************************************************************
 *  Module: kernels3d                                                   *
 *  Description: Device kernels for the 3D hydrostatic model of         *
 *               docs/03_discretization_spec.md S7. Layout is           *
 *               [k][j][i] with i contiguous - the same byte order as   *
 *               the NumPy [nz,ny,nx] reference and the Fortran         *
 *               (nx,ny,nz) backend, so the gate reads dumps directly.  *
 *  Pipeline: cfd_exp3d_cuda -> kernels3d -> device                     *
 ************************************************************************/
#pragma once
#include <cuda_runtime.h>

#define IDX3(i, j, k, nx, ny) ((((k) * (ny)) + (j)) * (nx) + (i))
#define IDX2(i, j, nx) ((j) * (nx) + (i))

__device__ __forceinline__ int wrap3(int i, int n) { return (i + n) % n; }

#define GRID3(nx, ny, nz)                                                      \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                       \
    const int j = blockIdx.y * blockDim.y + threadIdx.y;                       \
    const int k = blockIdx.z;                                                  \
    if (i >= (nx) || j >= (ny) || k >= (nz)) return;                           \
    const int c = IDX3(i, j, k, nx, ny);

#define COL2(nx, ny)                                                           \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                       \
    const int j = blockIdx.y * blockDim.y + threadIdx.y;                       \
    if (i >= (nx) || j >= (ny)) return;

/* ------------------------------------------------- horizontal, per level */
__global__ void k3_gradx_u(const double* __restrict__ a, double* __restrict__ o,
                           int nx, int ny, int nz, double dx) {
    GRID3(nx, ny, nz);
    o[c] = (a[IDX3(wrap3(i + 1, nx), j, k, nx, ny)] - a[c]) / dx;
}

__global__ void k3_grady_v(const double* __restrict__ a, double* __restrict__ o,
                           int nx, int ny, int nz, double dy) {
    GRID3(nx, ny, nz);
    o[c] = (a[IDX3(i, wrap3(j + 1, ny), k, nx, ny)] - a[c]) / dy;
}

__global__ void k3_div(const double* __restrict__ u, const double* __restrict__ v,
                       double* __restrict__ o, int nx, int ny, int nz,
                       double dx, double dy) {
    GRID3(nx, ny, nz);
    o[c] = (u[c] - u[IDX3(wrap3(i - 1, nx), j, k, nx, ny)]) / dx +
           (v[c] - v[IDX3(i, wrap3(j - 1, ny), k, nx, ny)]) / dy;
}

__global__ void k3_avg_v_to_u(const double* __restrict__ v, double* __restrict__ o,
                              int nx, int ny, int nz) {
    GRID3(nx, ny, nz);
    const int jm = wrap3(j - 1, ny), ip = wrap3(i + 1, nx);
    o[c] = 0.25 * (v[c] + v[IDX3(i, jm, k, nx, ny)] +
                   v[IDX3(ip, j, k, nx, ny)] + v[IDX3(ip, jm, k, nx, ny)]);
}

__global__ void k3_avg_u_to_v(const double* __restrict__ u, double* __restrict__ o,
                              int nx, int ny, int nz) {
    GRID3(nx, ny, nz);
    const int im = wrap3(i - 1, nx), jp = wrap3(j + 1, ny);
    o[c] = 0.25 * (u[c] + u[IDX3(im, j, k, nx, ny)] +
                   u[IDX3(i, jp, k, nx, ny)] + u[IDX3(im, jp, k, nx, ny)]);
}

/* ------------------------------------------------------------- vertical */
/* Thread-per-column Thomas: the whole sequential k recursion in ONE kernel
 * launch instead of 2*nz of them (spec S7.4). k is the slowest axis, so
 * neighbouring threads still read adjacent memory and stay coalesced. */
__global__ void k3_thomas_column(const double* __restrict__ sub,
                                 const double* __restrict__ diag,
                                 const double* __restrict__ sup,
                                 const double* __restrict__ rhs,
                                 double* __restrict__ x, double* __restrict__ cstar,
                                 int nx, int ny, int nz) {
    COL2(nx, ny);
    int c = IDX3(i, j, 0, nx, ny);
    const int stride = nx * ny;
    cstar[c] = sup[c] / diag[c];
    x[c] = rhs[c] / diag[c];
    for (int k = 1; k < nz; ++k) {
        const int cp = c;
        c += stride;
        const double denom = diag[c] - sub[c] * cstar[cp];
        cstar[c] = sup[c] / denom;
        x[c] = (rhs[c] - sub[c] * x[cp]) / denom;
    }
    for (int k = nz - 2; k >= 0; --k) {
        c -= stride;
        x[c] -= cstar[c] * x[c + stride];
    }
}

__global__ void k3_diffusion_coeffs(const double* __restrict__ nu,
                                    double* __restrict__ sub, double* __restrict__ diag,
                                    double* __restrict__ sup, int nx, int ny, int nz,
                                    double fac, double drag_term) {
    GRID3(nx, ny, nz);
    const int ci = IDX3(i, j, k, nx, ny);              /* interface above k */
    const int cb = IDX3(i, j, k + 1, nx, ny);          /* interface below k */
    const double a_top = fac * nu[ci];
    const double a_bot = fac * nu[cb];
    sub[c] = -a_top;
    sup[c] = -a_bot;
    diag[c] = 1.0 + a_top + a_bot;
    if (k == 0) {                                       /* prescribed surface flux */
        sub[c] = 0.0;
        diag[c] = 1.0 + a_bot;
    }
    if (k == nz - 1) {                                  /* prescribed bottom flux */
        sup[c] = 0.0;
        diag[c] = 1.0 + a_top + drag_term;
    }
}

__global__ void k3_apply_diffusion(const double* __restrict__ f,
                                   const double* __restrict__ nu,
                                   const double* __restrict__ surf_flux,
                                   double* __restrict__ o, int nx, int ny, int nz,
                                   double dz, double drag) {
    GRID3(nx, ny, nz);
    double ftop, fbot;
    if (k == 0) {
        ftop = surf_flux[IDX2(i, j, nx)];
    } else {
        ftop = nu[c] * (f[c - nx * ny] - f[c]) / dz;
    }
    if (k == nz - 1) {
        fbot = drag * f[c];
    } else {
        fbot = nu[IDX3(i, j, k + 1, nx, ny)] * (f[c] - f[c + nx * ny]) / dz;
    }
    o[c] = (ftop - fbot) / dz;
}

__global__ void k3_w_from_div(const double* __restrict__ div, double* __restrict__ w,
                              int nx, int ny, int nz, double dz) {
    COL2(nx, ny);
    const int stride = nx * ny;
    int cw = IDX3(i, j, nz, nx, ny);
    w[cw] = 0.0;
    for (int k = nz - 1; k >= 0; --k) {
        const int cd = IDX3(i, j, k, nx, ny);
        w[cw - stride] = w[cw] - dz * div[cd];
        cw -= stride;
    }
}

__global__ void k3_w_centres(const double* __restrict__ w, double* __restrict__ wc,
                             int nx, int ny, int nz) {
    GRID3(nx, ny, nz);
    wc[c] = 0.5 * (w[c] + w[c + nx * ny]);
}

/* Phi = int_z^0 b dz', then its depth mean removed - a correctness condition
 * for the internal-wave solution, not a convenience (spec S7.2). */
__global__ void k3_phi(const double* __restrict__ b, double* __restrict__ phi,
                       int nx, int ny, int nz, double dz) {
    COL2(nx, ny);
    const int stride = nx * ny;
    int c = IDX3(i, j, 0, nx, ny);
    double acc = dz * 0.5 * b[c];
    phi[c] = acc;
    double mean = acc;
    for (int k = 1; k < nz; ++k) {
        const double prev = b[c];
        c += stride;
        acc += dz * 0.5 * (prev + b[c]);
        phi[c] = acc;
        mean += acc;
    }
    mean /= nz;
    c = IDX3(i, j, 0, nx, ny);
    for (int k = 0; k < nz; ++k, c += stride) phi[c] -= mean;
}

__global__ void k3_depth_integral(const double* __restrict__ a, double* __restrict__ o,
                                  int nx, int ny, int nz, double dz) {
    COL2(nx, ny);
    const int stride = nx * ny;
    int c = IDX3(i, j, 0, nx, ny);
    double acc = 0.0;
    for (int k = 0; k < nz; ++k, c += stride) acc += a[c];
    o[IDX2(i, j, nx)] = dz * acc;
}

/* --------------------------------------------------------- scheme steps */
__global__ void k3_predictor(const double* __restrict__ u, const double* __restrict__ v,
                             const double* __restrict__ avpv,
                             const double* __restrict__ avpu,
                             const double* __restrict__ vcor,
                             const double* __restrict__ ucor,
                             const double* __restrict__ pgx,
                             const double* __restrict__ pgy,
                             const double* __restrict__ gx2,
                             const double* __restrict__ gy2,
                             const double* __restrict__ dzu,
                             const double* __restrict__ dzv,
                             const double* __restrict__ tau_u,
                             const double* __restrict__ tau_v,
                             double* __restrict__ gu, double* __restrict__ gv,
                             int nx, int ny, int nz, double dt, double f,
                             double tc, double c1, double thv, double dz) {
    GRID3(nx, ny, nz);
    const int c2 = IDX2(i, j, nx);
    gu[c] = u[c] + dt * (f * ((1.0 - tc) * avpv[c] + tc * vcor[c]) + pgx[c])
            - c1 * gx2[c2] + (1.0 - thv) * dt * dzu[c];
    gv[c] = v[c] + dt * (-f * ((1.0 - tc) * avpu[c] + tc * ucor[c]) + pgy[c])
            - c1 * gy2[c2] + (1.0 - thv) * dt * dzv[c];
    if (k == 0) {
        gu[c] += thv * dt * tau_u[c2] / dz;
        gv[c] += thv * dt * tau_v[c2] / dz;
    }
}

__global__ void k3_backsub(const double* __restrict__ ghu, const double* __restrict__ ghv,
                           const double* __restrict__ q, const double* __restrict__ gx2,
                           const double* __restrict__ gy2, double* __restrict__ uit,
                           double* __restrict__ vit, int nx, int ny, int nz, double c2f) {
    GRID3(nx, ny, nz);
    const int c2 = IDX2(i, j, nx);
    uit[c] = ghu[c] - c2f * q[c] * gx2[c2];
    vit[c] = ghv[c] - c2f * q[c] * gy2[c2];
}

__global__ void k3_b_rhs(const double* __restrict__ b, const double* __restrict__ wc,
                         const double* __restrict__ dzb, double* __restrict__ o,
                         int nx, int ny, int nz, double dt, double n2, double thv) {
    GRID3(nx, ny, nz);
    o[c] = b[c] - dt * n2 * wc[c] + (1.0 - thv) * dt * dzb[c];
}

__global__ void k3_copy(const double* __restrict__ a, double* __restrict__ o, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) o[c] = a[c];
}

__global__ void k3_fill(double* __restrict__ a, int n, double value) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) a[c] = value;
}

/* Free-surface right-hand side. The transports are already depth-integrated,
 * so no extra depth factor appears here (spec S7.5 step 4). */
__global__ void k2_rhs_eta(const double* __restrict__ eta,
                           const double* __restrict__ divold,
                           const double* __restrict__ divg,
                           double* __restrict__ rhs, int n, double dt, double th) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) rhs[c] = eta[c] - dt * ((1.0 - th) * divold[c] + th * divg[c]);
}

__global__ void k2_eta_fb(const double* __restrict__ eta,
                          const double* __restrict__ divg,
                          double* __restrict__ out, int n, double dt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) out[c] = eta[c] - dt * divg[c];
}

/* ------------------------------- split-explicit (spec S8) ------------------
 * The barotropic substeps are ordinary 2D stencils: no elliptic solve and no
 * global reduction, which is exactly the property being measured against the
 * semi-implicit scheme's PCG. */
__global__ void k2_avg_vu(const double* __restrict__ v, double* __restrict__ o,
                          int nx, int ny) {
    COL2(nx, ny);
    const int jm = wrap3(j - 1, ny), ip = wrap3(i + 1, nx);
    o[IDX2(i, j, nx)] = 0.25 * (v[IDX2(i, j, nx)] + v[IDX2(i, jm, nx)] +
                                v[IDX2(ip, j, nx)] + v[IDX2(ip, jm, nx)]);
}

__global__ void k2_avg_uv(const double* __restrict__ u, double* __restrict__ o,
                          int nx, int ny) {
    COL2(nx, ny);
    const int im = wrap3(i - 1, nx), jp = wrap3(j + 1, ny);
    o[IDX2(i, j, nx)] = 0.25 * (u[IDX2(i, j, nx)] + u[IDX2(im, j, nx)] +
                                u[IDX2(i, jp, nx)] + u[IDX2(im, jp, nx)]);
}

__global__ void k2_baro_mom(const double* __restrict__ bu, const double* __restrict__ bv,
                            const double* __restrict__ avv, const double* __restrict__ avu,
                            const double* __restrict__ vcor, const double* __restrict__ ucor,
                            const double* __restrict__ gxe, const double* __restrict__ gye,
                            const double* __restrict__ fx, const double* __restrict__ fy,
                            double* __restrict__ bui, double* __restrict__ bvi,
                            int n, double ddt, double f, double tc, double gh) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    bui[c] = bu[c] + ddt * (f * ((1.0 - tc) * avv[c] + tc * vcor[c]) - gh * gxe[c] + fx[c]);
    bvi[c] = bv[c] + ddt * (-f * ((1.0 - tc) * avu[c] + tc * ucor[c]) - gh * gye[c] + fy[c]);
}

__global__ void k2_eta_dec(double* __restrict__ eta, const double* __restrict__ div,
                           int n, double ddt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) eta[c] -= ddt * div[c];
}

/* Depth-integrated Coriolis actually applied in the 3D predictor, so it can be
 * subtracted from the barotropic forcing instead of being counted twice. */
__global__ void k3_coriolis_int(const double* __restrict__ avpv,
                                const double* __restrict__ avpu,
                                const double* __restrict__ vcor,
                                const double* __restrict__ ucor,
                                double* __restrict__ corx, double* __restrict__ cory,
                                int nx, int ny, int nz, double f, double tc, double dz) {
    COL2(nx, ny);
    const int stride = nx * ny;
    int c = IDX3(i, j, 0, nx, ny);
    double ax = 0.0, ay = 0.0;
    for (int k = 0; k < nz; ++k, c += stride) {
        ax += f * ((1.0 - tc) * avpv[c] + tc * vcor[c]);
        ay += -f * ((1.0 - tc) * avpu[c] + tc * ucor[c]);
    }
    corx[IDX2(i, j, nx)] = dz * ax;
    cory[IDX2(i, j, nx)] = dz * ay;
}

__global__ void k2_baro_forcing(const double* __restrict__ ghint_x,
                                const double* __restrict__ ghint_y,
                                const double* __restrict__ bu,
                                const double* __restrict__ bv,
                                const double* __restrict__ corx,
                                const double* __restrict__ cory,
                                double* __restrict__ fx, double* __restrict__ fy,
                                int n, double dt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    fx[c] = (ghint_x[c] - bu[c]) / dt - corx[c];
    fy[c] = (ghint_y[c] - bv[c]) / dt - cory[c];
}

/* Replace the depth mean of the baroclinic solution with the barotropic result
 * so that the depth integral matches exactly (spec S8.3 step 5). */
__global__ void k3_split_correct(const double* __restrict__ ghu,
                                 const double* __restrict__ ghv,
                                 const double* __restrict__ umean,
                                 const double* __restrict__ vmean,
                                 const double* __restrict__ bu,
                                 const double* __restrict__ bv,
                                 double* __restrict__ uit, double* __restrict__ vit,
                                 int nx, int ny, int nz, double inv_h) {
    GRID3(nx, ny, nz);
    const int c2 = IDX2(i, j, nx);
    uit[c] = ghu[c] - umean[c2] + bu[c2] * inv_h;
    vit[c] = ghv[c] - vmean[c2] + bv[c2] * inv_h;
}
