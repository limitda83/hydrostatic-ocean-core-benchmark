/************************************************************************
 *  Module: mg_device                                                   *
 *  Description: Geometric multigrid for the free-surface Helmholtz with *
 *               a red-black Gauss-Seidel smoother.                     *
 *                                                                      *
 *               Multi-colouring is what makes Gauss-Seidel usable on a *
 *               GPU at all: on a 5-point stencil every neighbour of a  *
 *               red cell is black, so each colour updates in one fully  *
 *               parallel kernel. Used as a smoother inside a V-cycle it *
 *               makes the iteration count almost independent of the     *
 *               condition number - the reference measures 241 CG        *
 *               iterations against 7 V-cycles at CFL 32 - and a V-cycle *
 *               needs ONE device-to-host synchronisation (the residual  *
 *               test) where a CG iteration needs three.                 *
 *  Pipeline: cfd_exp3d_cuda -> mg_device -> device                     *
 ************************************************************************/
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

__global__ void k2_rbgs(double* __restrict__ x, const double* __restrict__ b,
                        int nx, int ny, double cx, double cy, double diag, int colour) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    if (((i + j) & 1) != colour) return;
    const int c = j * nx + i;
    const double nb = cx * (x[j * nx + (i + 1) % nx] + x[j * nx + (i - 1 + nx) % nx])
                    + cy * (x[((j + 1) % ny) * nx + i] + x[((j - 1 + ny) % ny) * nx + i]);
    x[c] = (b[c] + nb) / diag;
}

__global__ void k2_residual(const double* __restrict__ x, const double* __restrict__ b,
                            double* __restrict__ r, int nx, int ny,
                            double cx, double cy, double diag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int c = j * nx + i;
    const double nb = cx * (x[j * nx + (i + 1) % nx] + x[j * nx + (i - 1 + nx) % nx])
                    + cy * (x[((j + 1) % ny) * nx + i] + x[((j - 1 + ny) % ny) * nx + i]);
    r[c] = b[c] - (diag * x[c] - nb);
}

__global__ void k2_restrict(const double* __restrict__ rf, double* __restrict__ rc,
                            int nxc, int nyc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nxc || j >= nyc) return;
    const int nxf = nxc * 2;
    rc[j * nxc + i] = 0.25 * (rf[(2 * j) * nxf + 2 * i] + rf[(2 * j) * nxf + 2 * i + 1]
                            + rf[(2 * j + 1) * nxf + 2 * i] + rf[(2 * j + 1) * nxf + 2 * i + 1]);
}

__global__ void k2_prolong_add(const double* __restrict__ ec, double* __restrict__ xf,
                               int nxc, int nyc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nxc || j >= nyc) return;
    const int nxf = nxc * 2;
    const double e = ec[j * nxc + i];
    xf[(2 * j) * nxf + 2 * i] += e;
    xf[(2 * j) * nxf + 2 * i + 1] += e;
    xf[(2 * j + 1) * nxf + 2 * i] += e;
    xf[(2 * j + 1) * nxf + 2 * i + 1] += e;
}

struct MgLevel {
    int nx, ny, n;
    double cx, cy, diag;
    double *x = nullptr, *b = nullptr, *r = nullptr;
    dim3 grid, block;
};

struct HelmholtzMG {
    std::vector<MgLevel> lv;
    double rtol = 1e-12;
    int max_iter = 100, n_pre = 2, n_post = 2, n_coarse = 40;
    long total_iterations = 0;
    int failures = 0;
    double *partial = nullptr, *d_scalar = nullptr;
    int n_partial = 0, n_fine = 0;

    void init(int nx, int ny, double dx, double dy, double coef, double rtol_,
              int max_iter_) {
        rtol = rtol_;
        max_iter = max_iter_;
        n_fine = nx * ny;
        while (true) {
            MgLevel L;
            L.nx = nx; L.ny = ny; L.n = nx * ny;
            L.cx = coef / (dx * dx);
            L.cy = coef / (dy * dy);
            L.diag = 1.0 + 2.0 * L.cx + 2.0 * L.cy;
            L.block = dim3(32, 8);
            L.grid = dim3((nx + 31) / 32, (ny + 7) / 8);
            CUDA_CHECK(cudaMalloc(&L.x, L.n * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&L.b, L.n * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&L.r, L.n * sizeof(double)));
            lv.push_back(L);
            if (nx % 2 || ny % 2 || std::min(nx, ny) <= 4) break;
            nx /= 2; ny /= 2; dx *= 2; dy *= 2;
        }
        n_partial = std::min(1024, (n_fine + 255) / 256);
        CUDA_CHECK(cudaMalloc(&partial, n_partial * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(double)));
    }

    void smooth(int l, int n) {
        MgLevel& L = lv[l];
        for (int k = 0; k < n; ++k) {
            /* Alternating the colour order keeps the smoother symmetric,
             * which matters if it is ever used as a CG preconditioner. */
            const int first = k & 1;
            k2_rbgs<<<L.grid, L.block>>>(L.x, L.b, L.nx, L.ny, L.cx, L.cy, L.diag, first);
            k2_rbgs<<<L.grid, L.block>>>(L.x, L.b, L.nx, L.ny, L.cx, L.cy, L.diag, 1 - first);
        }
    }

    void vcycle(int l) {
        MgLevel& L = lv[l];
        if (l == int(lv.size()) - 1) { smooth(l, n_coarse); return; }
        smooth(l, n_pre);
        k2_residual<<<L.grid, L.block>>>(L.x, L.b, L.r, L.nx, L.ny, L.cx, L.cy, L.diag);
        MgLevel& C = lv[l + 1];
        k2_restrict<<<C.grid, C.block>>>(L.r, C.b, C.nx, C.ny);
        k_zero_d<<<(C.n + 255) / 256, 256>>>(C.x, C.n);
        vcycle(l + 1);
        k2_prolong_add<<<C.grid, C.block>>>(C.x, L.x, C.nx, C.ny);
        smooth(l, n_post);
    }

    void solve(const double* rhs, double* out) {
        MgLevel& F = lv[0];
        CUDA_CHECK(cudaMemcpy(F.b, rhs, F.n * sizeof(double), cudaMemcpyDeviceToDevice));
        k_zero_d<<<(F.n + 255) / 256, 256>>>(F.x, F.n);

        k_dot<256><<<n_partial, 256>>>(F.b, F.b, partial, F.n);
        k_dot_final<256><<<1, 256>>>(partial, d_scalar, n_partial);
        double nb = 0.0;
        CUDA_CHECK(cudaMemcpy(&nb, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
        nb = std::sqrt(nb);
        if (nb == 0.0) {
            CUDA_CHECK(cudaMemcpy(out, F.x, F.n * sizeof(double), cudaMemcpyDeviceToDevice));
            return;
        }

        int used = max_iter;
        bool ok = false;
        double residual = 0.0;
        for (int it = 1; it <= max_iter; ++it) {
            vcycle(0);
            k2_residual<<<F.grid, F.block>>>(F.x, F.b, F.r, F.nx, F.ny, F.cx, F.cy, F.diag);
            k_dot<256><<<n_partial, 256>>>(F.r, F.r, partial, F.n);
            k_dot_final<256><<<1, 256>>>(partial, d_scalar, n_partial);
            double rr = 0.0;
            /* The only synchronisation in a V-cycle. */
            CUDA_CHECK(cudaMemcpy(&rr, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
            residual = std::sqrt(rr) / nb;
            if (residual < rtol) { ok = true; used = it; break; }
        }
        total_iterations += used;
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "ERROR: multigrid failed to converge: %d V-cycles, "
                                 "residual=%.5e > rtol=%.5e\n", max_iter, residual, rtol);
        }
        CUDA_CHECK(cudaMemcpy(out, F.x, F.n * sizeof(double), cudaMemcpyDeviceToDevice));
    }
};
