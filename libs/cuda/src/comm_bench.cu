/************************************************************************
 *  Module: comm_bench (CUDA + MPI)                                     *
 *  Description: What multi-GPU communication costs, per operation, so  *
 *               that the semi-implicit and split-explicit schemes can  *
 *               be composed from measured constants.                   *
 *                                                                      *
 *  The two schemes reach the same time step by paying different tolls: *
 *                                                                      *
 *    semi-implicit  N_iter  x ( matvec + halo + 2 x Allreduce )        *
 *    split-explicit N_sub   x ( local update + halo )                  *
 *                                                                      *
 *  N_iter and N_sub are already measured (docs/22, docs/25). What is   *
 *  missing is the price of a halo exchange and of an Allreduce on this *
 *  machine, which is what this program measures. A single-node,        *
 *  slab-decomposed benchmark cannot stand in for a real multi-node     *
 *  solver - it is the constants that transfer, not the verdict.        *
 *  Pipeline: comm_bench -> docs/28                                     *
 ************************************************************************/

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

#define CUDA_OK(call)                                                        \
    do { cudaError_t _e = (call);                                            \
         if (_e != cudaSuccess) {                                            \
             std::fprintf(stderr, "FATAL: %s at %s:%d\n",                    \
                          cudaGetErrorString(_e), __FILE__, __LINE__);       \
             MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

#define IDX(i, j, nx) ((j) * (nx) + (i))

// The local slab carries one halo row above and below, so row j of the
// interior lives at j+1 in the local array.
__global__ void k_apply_local(int nx, int nyl, double coef, double dx, double dy,
                              const double *ku, const double *kv,
                              const double *msk, const double *x, double *ax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= nyl) return;
    int jj = j + 1;                       // interior row in the padded array
    int ip = (i == nx - 1) ? 0 : i + 1, im = (i == 0) ? nx - 1 : i - 1;
    double cx = coef / (dx * dx), cy = coef / (dy * dy);
    double c = x[IDX(i, jj, nx)];
    double lap = cx * (ku[IDX(i, jj, nx)] * (x[IDX(ip, jj, nx)] - c)
                     - ku[IDX(im, jj, nx)] * (c - x[IDX(im, jj, nx)]))
               + cy * (kv[IDX(i, jj, nx)] * (x[IDX(i, jj + 1, nx)] - c)
                     - kv[IDX(i, jj - 1, nx)] * (c - x[IDX(i, jj - 1, nx)]));
    ax[IDX(i, jj, nx)] = (msk[IDX(i, jj, nx)] > 0.0) ? c - lap : c;
}

// The explicit barotropic substep of the split scheme: a local stencil and
// nothing else. Same memory traffic class as the matvec, no reduction.
__global__ void k_substep(int nx, int nyl, double a,
                          const double *u, const double *v, double *e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= nyl) return;
    int jj = j + 1;
    int im = (i == 0) ? nx - 1 : i - 1;
    e[IDX(i, jj, nx)] -= a * ((u[IDX(i, jj, nx)] - u[IDX(im, jj, nx)])
                            + (v[IDX(i, jj, nx)] - v[IDX(i, jj - 1, nx)]));
}

__global__ void k_dot(int n, const double *a, const double *b, double *out)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    double s = 0.0;
    for (int t = blockIdx.x * blockDim.x + tid; t < n; t += blockDim.x * gridDim.x)
        s += a[t] * b[t];
    sh[tid] = s; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) sh[tid] += sh[tid + k];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

__global__ void k_axpy2(int n, double alpha, const double *p, const double *ap,
                        double *x, double *r)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) { x[t] += alpha * p[t]; r[t] -= alpha * ap[t]; }
}

static double wall()
{
    return MPI_Wtime();
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int nx = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int ny = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int reps = (argc > 3) ? std::atoi(argv[3]) : 200;
    bool host_stage = (argc > 4) && std::string(argv[4]) == "host";

    if (ny % nranks != 0) {
        if (!rank) std::fprintf(stderr, "FATAL: ny must divide by nranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int nyl = ny / nranks;

    int ndev = 0;
    CUDA_OK(cudaGetDeviceCount(&ndev));
    CUDA_OK(cudaSetDevice(rank % ndev));

    const size_t npad = size_t(nx) * (nyl + 2);
    double *ku, *kv, *msk, *x, *ax, *p, *r, *scal, *u, *v, *e;
    for (double **q : {&ku, &kv, &msk, &x, &ax, &p, &r, &u, &v, &e})
        CUDA_OK(cudaMalloc(q, npad * sizeof(double)));
    CUDA_OK(cudaMalloc(&scal, sizeof(double)));
    {
        std::vector<double> h(npad, 1000.0);
        for (double *q : {ku, kv, msk, x, ax, p, r, u, v, e})
            CUDA_OK(cudaMemcpy(q, h.data(), npad * sizeof(double),
                               cudaMemcpyHostToDevice));
    }

    const int up = (rank + 1) % nranks, dn = (rank - 1 + nranks) % nranks;
    std::vector<double> hs_send_u(nx), hs_send_d(nx), hs_recv_u(nx), hs_recv_d(nx);

    auto halo = [&](double *f) {
        double *top_send = f + size_t(nx) * nyl;        // last interior row
        double *bot_send = f + size_t(nx) * 1;          // first interior row
        double *top_recv = f + size_t(nx) * (nyl + 1);  // upper halo
        double *bot_recv = f;                           // lower halo
        if (!host_stage) {
            MPI_Sendrecv(top_send, nx, MPI_DOUBLE, up, 0,
                         bot_recv, nx, MPI_DOUBLE, dn, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(bot_send, nx, MPI_DOUBLE, dn, 1,
                         top_recv, nx, MPI_DOUBLE, up, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            CUDA_OK(cudaMemcpy(hs_send_u.data(), top_send, nx * sizeof(double),
                               cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(hs_send_d.data(), bot_send, nx * sizeof(double),
                               cudaMemcpyDeviceToHost));
            MPI_Sendrecv(hs_send_u.data(), nx, MPI_DOUBLE, up, 0,
                         hs_recv_d.data(), nx, MPI_DOUBLE, dn, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(hs_send_d.data(), nx, MPI_DOUBLE, dn, 1,
                         hs_recv_u.data(), nx, MPI_DOUBLE, up, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_OK(cudaMemcpy(bot_recv, hs_recv_d.data(), nx * sizeof(double),
                               cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(top_recv, hs_recv_u.data(), nx * sizeof(double),
                               cudaMemcpyHostToDevice));
        }
    };

    dim3 blk(32, 8), grd((nx + 31) / 32, (nyl + 7) / 8);
    const int TPB = 256, n = nx * nyl;
    const int nblk = (int(npad) + TPB - 1) / TPB;
    const int dot_blocks = std::min(nblk, 1024);

    auto dot_local = [&](const double *a, const double *b) {
        double zero = 0.0, out = 0.0;
        CUDA_OK(cudaMemcpy(scal, &zero, sizeof(double), cudaMemcpyHostToDevice));
        k_dot<<<dot_blocks, TPB, TPB * sizeof(double)>>>(n, a + nx, b + nx, scal);
        CUDA_OK(cudaMemcpy(&out, scal, sizeof(double), cudaMemcpyDeviceToHost));
        return out;
    };

    auto time_it = [&](const char *name, int r, void (*)(void) = nullptr) { (void)name; (void)r; };
    (void)time_it;

    auto bench = [&](const char *name, int r, const std::function<void()> &f) {
        for (int k = 0; k < 5; ++k) f();
        CUDA_OK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = wall();
        for (int k = 0; k < r; ++k) f();
        CUDA_OK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        double t = (wall() - t0) / r * 1e6;
        double tmax;
        MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank)
            std::printf("%-22s ranks=%d nx=%d ny=%d  %10.3f us\n",
                        name, nranks, nx, ny, tmax);
        return tmax;
    };

    double t_matvec = bench("matvec", reps, [&]{
        k_apply_local<<<grd, blk>>>(nx, nyl, 1.0e8, 400.0, 400.0, ku, kv, msk, x, ax);
    });
    double t_halo = bench("halo", reps, [&]{
        CUDA_OK(cudaDeviceSynchronize());
        halo(x);
    });
    double t_allred = bench("allreduce_1", reps, [&]{
        double a = 1.0, b;
        MPI_Allreduce(&a, &b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    });
    double t_dot = bench("dot_local", reps, [&]{ dot_local(x, ax); });
    double t_pcg = bench("pcg_iteration", reps, [&]{
        halo(p);
        k_apply_local<<<grd, blk>>>(nx, nyl, 1.0e8, 400.0, 400.0, ku, kv, msk, p, ax);
        double s = dot_local(p, ax), g;
        MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        k_axpy2<<<nblk, TPB>>>(int(npad), 1e-12, p, ax, x, r);
        double s2 = dot_local(r, r), g2;
        MPI_Allreduce(&s2, &g2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        (void)g; (void)g2;
    });
    double t_sub = bench("barotropic_substep", reps, [&]{
        halo(u); halo(v);
        k_substep<<<grd, blk>>>(nx, nyl, 1e-6, u, v, e);
    });

    if (!rank) {
        std::printf("\n# summary ranks=%d nx=%d ny=%d staging=%s\n", nranks, nx, ny,
                    host_stage ? "host" : "device-direct");
        std::printf("# matvec %.3f  halo %.3f  allreduce %.3f  dot %.3f"
                    "  pcg_iter %.3f  substep %.3f  [us]\n",
                    t_matvec, t_halo, t_allred, t_dot, t_pcg, t_sub);
        std::printf("CSV,%d,%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    nranks, nx, ny, host_stage ? "host" : "device",
                    t_matvec, t_halo, t_allred, t_dot, t_pcg, t_sub);
    }
    MPI_Finalize();
    return 0;
}
