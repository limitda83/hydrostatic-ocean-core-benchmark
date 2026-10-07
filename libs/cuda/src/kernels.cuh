/************************************************************************
 *  Module: kernels                                                     *
 *  Description: Device kernels implementing the discrete C-grid        *
 *               operators and theta/forward-backward updates of        *
 *               docs/03_discretization_spec.md S2.1 and S3.            *
 *               Layout is [j][i] with i contiguous - the same memory   *
 *               order as the NumPy reference and the Fortran backend.  *
 *  Pipeline: cfd_exp_cuda -> kernels -> device                         *
 ************************************************************************/
#pragma once
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "FATAL: CUDA error %s at %s:%d\n",            \
                         cudaGetErrorString(err_), __FILE__, __LINE__);        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

/* Periodic wrap, expressed as index arithmetic rather than halo exchange
 * (spec S6.4). Branch-free so warps never diverge on the boundary. */
__device__ __forceinline__ int wrap(int i, int n) { return (i + n) % n; }

#define IDX(i, j, nx) ((j) * (nx) + (i))

#define GRID_STRIDE_2D(nx, ny)                                                 \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                       \
    const int j = blockIdx.y * blockDim.y + threadIdx.y;                       \
    if (i >= (nx) || j >= (ny)) return;                                        \
    const int c = IDX(i, j, nx);

/* ------------------------------------------------------------ operators */
__global__ void k_gradx_u(const double* __restrict__ eta, double* __restrict__ out,
                          int nx, int ny, double dx) {
    GRID_STRIDE_2D(nx, ny);
    out[c] = (eta[IDX(wrap(i + 1, nx), j, nx)] - eta[c]) / dx;
}

__global__ void k_grady_v(const double* __restrict__ eta, double* __restrict__ out,
                          int nx, int ny, double dy) {
    GRID_STRIDE_2D(nx, ny);
    out[c] = (eta[IDX(i, wrap(j + 1, ny), nx)] - eta[c]) / dy;
}

__global__ void k_div(const double* __restrict__ u, const double* __restrict__ v,
                      double* __restrict__ out, int nx, int ny, double dx, double dy) {
    GRID_STRIDE_2D(nx, ny);
    out[c] = (u[c] - u[IDX(wrap(i - 1, nx), j, nx)]) / dx +
             (v[c] - v[IDX(i, wrap(j - 1, ny), nx)]) / dy;
}

__global__ void k_avg_v_to_u(const double* __restrict__ v, double* __restrict__ out,
                             int nx, int ny) {
    GRID_STRIDE_2D(nx, ny);
    const int jm = wrap(j - 1, ny), ip = wrap(i + 1, nx);
    out[c] = 0.25 * (v[c] + v[IDX(i, jm, nx)] + v[IDX(ip, j, nx)] + v[IDX(ip, jm, nx)]);
}

__global__ void k_avg_u_to_v(const double* __restrict__ u, double* __restrict__ out,
                             int nx, int ny) {
    GRID_STRIDE_2D(nx, ny);
    const int im = wrap(i - 1, nx), jp = wrap(j + 1, ny);
    out[c] = 0.25 * (u[c] + u[IDX(im, j, nx)] + u[IDX(i, jp, nx)] + u[IDX(im, jp, nx)]);
}

/* --------------------------------------------------------- scheme steps */
/* Explicit predictors Gu, Gv. Term order fixed by spec S6.1. */
__global__ void k_predictor(const double* __restrict__ u, const double* __restrict__ v,
                            const double* __restrict__ av_prev_v,
                            const double* __restrict__ av_prev_u,
                            const double* __restrict__ v_cor,
                            const double* __restrict__ u_cor,
                            const double* __restrict__ gx_eta,
                            const double* __restrict__ gy_eta,
                            double* __restrict__ gu, double* __restrict__ gv,
                            int nx, int ny, double dt, double f, double tc, double c1) {
    GRID_STRIDE_2D(nx, ny);
    gu[c] = u[c] + dt * f * ((1.0 - tc) * av_prev_v[c] + tc * v_cor[c]) - c1 * gx_eta[c];
    gv[c] = v[c] - dt * f * ((1.0 - tc) * av_prev_u[c] + tc * u_cor[c]) - c1 * gy_eta[c];
}

/* Forward-backward momentum update (spec S3.2). */
__global__ void k_fb_momentum(const double* __restrict__ u, const double* __restrict__ v,
                              const double* __restrict__ av_prev_v,
                              const double* __restrict__ av_prev_u,
                              const double* __restrict__ v_cor,
                              const double* __restrict__ u_cor,
                              const double* __restrict__ gx_eta,
                              const double* __restrict__ gy_eta,
                              double* __restrict__ u_it, double* __restrict__ v_it,
                              int nx, int ny, double dt, double f, double tc, double g) {
    GRID_STRIDE_2D(nx, ny);
    u_it[c] = u[c] + dt * (f * ((1.0 - tc) * av_prev_v[c] + tc * v_cor[c]) - g * gx_eta[c]);
    v_it[c] = v[c] + dt * (-f * ((1.0 - tc) * av_prev_u[c] + tc * u_cor[c]) - g * gy_eta[c]);
}

/* Element-wise, so it takes a 1D launch. Kernels using GRID_STRIDE_2D must be
 * launched with the 2D configuration; mixing the two silently computes only
 * the j = 0 row and leaves the rest of the field uninitialised. */
__global__ void k_rhs(const double* __restrict__ eta, const double* __restrict__ div_old,
                      const double* __restrict__ div_g, double* __restrict__ rhs,
                      int n, double dt, double h, double th) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    rhs[c] = eta[c] - dt * h * ((1.0 - th) * div_old[c] + th * div_g[c]);
}

__global__ void k_back_substitute(const double* __restrict__ gu,
                                  const double* __restrict__ gv,
                                  const double* __restrict__ gx_new,
                                  const double* __restrict__ gy_new,
                                  double* __restrict__ u_it, double* __restrict__ v_it,
                                  int nx, int ny, double c2) {
    GRID_STRIDE_2D(nx, ny);
    u_it[c] = gu[c] - c2 * gx_new[c];
    v_it[c] = gv[c] - c2 * gy_new[c];
}

__global__ void k_commit(const double* __restrict__ u_it, const double* __restrict__ v_it,
                         const double* __restrict__ eta_new, double* __restrict__ u,
                         double* __restrict__ v, double* __restrict__ eta, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    u[c] = u_it[c];
    v[c] = v_it[c];
    eta[c] = eta_new[c];
}

__global__ void k_fb_continuity(double* __restrict__ eta, const double* __restrict__ div,
                                int n, double dt, double h) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    eta[c] -= dt * h * div[c];
}

/* ------------------------------------------------------------ PCG pieces */
__global__ void k_apply(const double* __restrict__ x, const double* __restrict__ lap,
                        double* __restrict__ ax, int n, double coef) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    ax[c] = x[c] - coef * lap[c];
}

__global__ void k_zero(double* __restrict__ x, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) x[c] = 0.0;
}

__global__ void k_pcg_init(const double* __restrict__ rhs, const double* __restrict__ ap,
                           double* __restrict__ r, double* __restrict__ z,
                           double* __restrict__ p, int n, double inv_diag) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    r[c] = rhs[c] - ap[c];
    z[c] = inv_diag * r[c];
    p[c] = z[c];
}

__global__ void k_pcg_update_x_r(double* __restrict__ x, double* __restrict__ r,
                                 const double* __restrict__ p,
                                 const double* __restrict__ ap, int n, double alpha) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    x[c] += alpha * p[c];
    r[c] -= alpha * ap[c];
}

__global__ void k_pcg_precondition(const double* __restrict__ r, double* __restrict__ z,
                                   int n, double inv_diag) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) z[c] = inv_diag * r[c];
}

__global__ void k_pcg_update_p(double* __restrict__ p, const double* __restrict__ z,
                               int n, double beta) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) p[c] = z[c] + beta * p[c];
}

/* Two-stage dot product. The host reads the single scalar back each time,
 * which is exactly the global-reduction synchronisation RQ2 is about - it is
 * left visible rather than hidden behind an asynchronous trick. */
template <int BLOCK>
__global__ void k_dot_partial(const double* __restrict__ a, const double* __restrict__ b,
                              double* __restrict__ partial, int n) {
    __shared__ double s[BLOCK];
    const int tid = threadIdx.x;
    double acc = 0.0;
    for (int c = blockIdx.x * BLOCK + tid; c < n; c += BLOCK * gridDim.x) {
        acc += a[c] * b[c];
    }
    s[tid] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

template <int BLOCK>
__global__ void k_dot_final(const double* __restrict__ partial, double* __restrict__ out,
                            int n_partial) {
    __shared__ double s[BLOCK];
    const int tid = threadIdx.x;
    double acc = 0.0;
    for (int c = tid; c < n_partial; c += BLOCK) acc += partial[c];
    s[tid] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *out = s[0];
}
