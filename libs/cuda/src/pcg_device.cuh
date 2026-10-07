/************************************************************************
 *  Module: pcg_device                                                  *
 *  Description: Free-surface Helmholtz PCG with two synchronisation    *
 *               strategies, selectable at runtime:                     *
 *                 "host"   - every inner product is copied to the host *
 *                            (what OpenACC's reduction clause does)    *
 *                 "device" - alpha/beta stay in device memory and the  *
 *                            update kernels read them from there; only *
 *                            the convergence test copies, every        *
 *                            pcg_check_every iterations                *
 *               docs/21 S3 measured 174 us per iteration on GPU        *
 *               against 18 us on 16 CPU cores and attributed it to     *
 *               these synchronisations. Having both paths in one       *
 *               binary turns that attribution into a measurement.      *
 *  Pipeline: cfd_exp3d_cuda -> pcg_device -> device                    *
 ************************************************************************/
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "FATAL: CUDA error %s at %s:%d\n",            \
                         cudaGetErrorString(err_), __FILE__, __LINE__);        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#endif

/* ------------------------------------------------------- 2D operators */
__global__ void k2_gradx(const double* __restrict__ a, double* __restrict__ o,
                         int nx, int ny, double dx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    o[j * nx + i] = (a[j * nx + (i + 1) % nx] - a[j * nx + i]) / dx;
}

__global__ void k2_grady(const double* __restrict__ a, double* __restrict__ o,
                         int nx, int ny, double dy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    o[j * nx + i] = (a[((j + 1) % ny) * nx + i] - a[j * nx + i]) / dy;
}

__global__ void k2_div(const double* __restrict__ u, const double* __restrict__ v,
                       double* __restrict__ o, int nx, int ny, double dx, double dy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    o[j * nx + i] = (u[j * nx + i] - u[j * nx + (i - 1 + nx) % nx]) / dx +
                    (v[j * nx + i] - v[((j - 1 + ny) % ny) * nx + i]) / dy;
}

/* Fused Helmholtz apply: ax = x - coef * div(grad(x)). One kernel instead of
 * four, which removes three launches per PCG iteration. */
__global__ void k2_helmholtz(const double* __restrict__ x, double* __restrict__ ax,
                             int nx, int ny, double dx, double dy, double coef) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int c = j * nx + i;
    const double xc = x[c];
    const double lap = (x[j * nx + (i + 1) % nx] - 2.0 * xc + x[j * nx + (i - 1 + nx) % nx]) / (dx * dx)
                     + (x[((j + 1) % ny) * nx + i] - 2.0 * xc + x[((j - 1 + ny) % ny) * nx + i]) / (dy * dy);
    ax[c] = xc - coef * lap;
}

/* ----------------------------------------------------- device scalars */
__global__ void k_scalar_div(double* __restrict__ out, const double* __restrict__ a,
                             const double* __restrict__ b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = a[0] / b[0];
}

__global__ void k_scalar_copy(double* __restrict__ dst, const double* __restrict__ src) {
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[0] = src[0];
}

template <int BLOCK>
__global__ void k_dot(const double* __restrict__ a, const double* __restrict__ b,
                      double* __restrict__ partial, int n) {
    __shared__ double s[BLOCK];
    const int tid = threadIdx.x;
    double acc = 0.0;
    for (int c = blockIdx.x * BLOCK + tid; c < n; c += BLOCK * gridDim.x) acc += a[c] * b[c];
    s[tid] = acc;
    __syncthreads();
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
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
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

/* Update kernels take alpha/beta by DEVICE POINTER, so no host round-trip is
 * needed between the reduction that produced them and their use. */
__global__ void k_pcg_xr(double* __restrict__ x, double* __restrict__ r,
                         const double* __restrict__ p, const double* __restrict__ ap,
                         int n, const double* __restrict__ alpha) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const double a = alpha[0];
    x[c] += a * p[c];
    r[c] -= a * ap[c];
}

__global__ void k_pcg_p(double* __restrict__ p, const double* __restrict__ z,
                        int n, const double* __restrict__ beta) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) p[c] = z[c] + beta[0] * p[c];
}

__global__ void k_pcg_precond(const double* __restrict__ r, double* __restrict__ z,
                              int n, double inv_diag) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) z[c] = inv_diag * r[c];
}

__global__ void k_pcg_start(const double* __restrict__ rhs, const double* __restrict__ ap,
                            double* __restrict__ r, double* __restrict__ z,
                            double* __restrict__ p, int n, double inv_diag) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    r[c] = rhs[c] - ap[c];
    z[c] = inv_diag * r[c];
    p[c] = z[c];
}

__global__ void k_zero_d(double* __restrict__ a, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) a[c] = 0.0;
}

/* ------------------------------------------------------------- solver */
struct PcgDevice {
    static constexpr int BLOCK = 256;
    int nx = 0, ny = 0, n = 0, n_partial = 0;
    double dx = 0.0, dy = 0.0, coef = 0.0, inv_diag = 1.0, rtol = 1e-12;
    int max_iter = 2000, check_every = 5;
    bool device_scalars = true;
    long total_iterations = 0;
    int failures = 0;

    double *r = nullptr, *z = nullptr, *p = nullptr, *ap = nullptr, *partial = nullptr;
    double *d_rz = nullptr, *d_rz_new = nullptr, *d_pap = nullptr, *d_rr = nullptr;
    double *d_alpha = nullptr, *d_beta = nullptr, *d_nrmb = nullptr;
    dim3 block2d, grid2d, block1d, grid1d;

    void init(int nx_, int ny_, double dx_, double dy_, double coef_, double rtol_,
              int max_iter_, int check_every_, const std::string& sync_mode) {
        nx = nx_; ny = ny_; n = nx_ * ny_;
        dx = dx_; dy = dy_; coef = coef_; rtol = rtol_;
        max_iter = max_iter_; check_every = std::max(1, check_every_);
        device_scalars = (sync_mode != "host");
        inv_diag = 1.0 / (1.0 + coef * (2.0 / (dx * dx) + 2.0 / (dy * dy)));
        block2d = dim3(32, 8);
        grid2d = dim3((nx + 31) / 32, (ny + 7) / 8);
        block1d = dim3(BLOCK);
        grid1d = dim3((n + BLOCK - 1) / BLOCK);
        n_partial = std::min(1024, int(grid1d.x));
        for (double** q : {&r, &z, &p, &ap}) CUDA_CHECK(cudaMalloc(q, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&partial, n_partial * sizeof(double)));
        for (double** q : {&d_rz, &d_rz_new, &d_pap, &d_rr, &d_alpha, &d_beta, &d_nrmb})
            CUDA_CHECK(cudaMalloc(q, sizeof(double)));
    }

    void dot(const double* a, const double* b, double* out) {
        k_dot<BLOCK><<<n_partial, BLOCK>>>(a, b, partial, n);
        k_dot_final<BLOCK><<<1, BLOCK>>>(partial, out, n_partial);
    }

    double fetch(const double* d) {
        double h = 0.0;
        CUDA_CHECK(cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }

    void solve(const double* rhs, double* x) {
        k_zero_d<<<grid1d, block1d>>>(x, n);
        k2_helmholtz<<<grid2d, block2d>>>(x, ap, nx, ny, dx, dy, coef);
        dot(rhs, rhs, d_nrmb);
        const double norm_b = std::sqrt(fetch(d_nrmb));
        if (norm_b == 0.0) return;

        k_pcg_start<<<grid1d, block1d>>>(rhs, ap, r, z, p, n, inv_diag);
        dot(r, z, d_rz);
        double h_rz = device_scalars ? 0.0 : fetch(d_rz);

        int used = max_iter;
        bool converged = false;
        double residual = 0.0;
        for (int it = 1; it <= max_iter; ++it) {
            k2_helmholtz<<<grid2d, block2d>>>(p, ap, nx, ny, dx, dy, coef);
            dot(p, ap, d_pap);
            if (device_scalars) {
                k_scalar_div<<<1, 1>>>(d_alpha, d_rz, d_pap);
                k_pcg_xr<<<grid1d, block1d>>>(x, r, p, ap, n, d_alpha);
            } else {
                const double alpha = h_rz / fetch(d_pap);
                CUDA_CHECK(cudaMemcpy(d_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice));
                k_pcg_xr<<<grid1d, block1d>>>(x, r, p, ap, n, d_alpha);
            }
            dot(r, r, d_rr);

            /* The only mandatory host round-trip, and in device mode it is
             * taken every check_every iterations instead of every one. */
            const bool test = !device_scalars || (it % check_every == 0) || it == max_iter;
            if (test) {
                residual = std::sqrt(fetch(d_rr)) / norm_b;
                if (residual < rtol) { converged = true; used = it; break; }
            }

            k_pcg_precond<<<grid1d, block1d>>>(r, z, n, inv_diag);
            dot(r, z, d_rz_new);
            if (device_scalars) {
                k_scalar_div<<<1, 1>>>(d_beta, d_rz_new, d_rz);
                k_pcg_p<<<grid1d, block1d>>>(p, z, n, d_beta);
                k_scalar_copy<<<1, 1>>>(d_rz, d_rz_new);
            } else {
                const double rz_new = fetch(d_rz_new);
                const double beta = rz_new / h_rz;
                CUDA_CHECK(cudaMemcpy(d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice));
                k_pcg_p<<<grid1d, block1d>>>(p, z, n, d_beta);
                h_rz = rz_new;
            }
        }
        total_iterations += used;
        if (!converged) {
            ++failures;
            std::fprintf(stderr, "ERROR: PCG failed to converge: %d iterations, "
                                 "residual=%.5e > rtol=%.5e\n", max_iter, residual, rtol);
        }
    }
};
