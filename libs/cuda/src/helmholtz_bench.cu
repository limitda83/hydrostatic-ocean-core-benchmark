/************************************************************************
 *  Module: helmholtz_bench (CUDA)                                      *
 *  Description: Native CUDA counterpart of libs/fortran/src/           *
 *               helmholtz_bench.f90 - the variable-coefficient free-   *
 *               surface Helmholtz solve of spec S10.5 on a real        *
 *               bathymetry. Same four solvers, same stopping rule,     *
 *               same input files, so the iteration counts must match   *
 *               the Fortran and NumPy ones exactly (R2).               *
 *                                                                      *
 *  Everything stays resident on the device; the only host transfers    *
 *  are the reduction scalars CG needs, which is the cost docs/21       *
 *  measured at 30 us per iteration.                                    *
 *  Pipeline: tools/write_helmholtz_case.py -> helmholtz_bench -> JSON  *
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>

#define CUDA_OK(call)                                                        \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "FATAL: %s at %s:%d\n",                     \
                         cudaGetErrorString(_e), __FILE__, __LINE__);        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Column-major to match the Fortran layout of the binary inputs.
#define IDX(i, j, nx) ((j) * (nx) + (i))

// Working precision of the SOLVER (axis G). The problem on disk is always
// fp64 and is cast on upload, so the error against the fp64 reference is
// attributable to the arithmetic alone. Build fp32 with -DHELM_REAL=float.
#ifndef HELM_REAL
#define HELM_REAL double
#endif
typedef HELM_REAL real_t;
#define PRECISION_NAME (sizeof(real_t) == 4 ? "fp32" : "fp64")

struct Level {
    int nx = 0, ny = 0;
    real_t dx = 0.0, dy = 0.0;
    real_t *ku = nullptr, *kv = nullptr, *msk = nullptr, *dinv = nullptr;
    real_t *x = nullptr, *b = nullptr, *r = nullptr;
};

__global__ void k_setup_dinv(Level lv, real_t coef)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lv.nx || j >= lv.ny) return;
    int im = (i == 0) ? lv.nx - 1 : i - 1;
    int jm = (j == 0) ? lv.ny - 1 : j - 1;
    real_t cx = coef / (lv.dx * lv.dx), cy = coef / (lv.dy * lv.dy);
    real_t d = 1.0 + cx * (lv.ku[IDX(i, j, lv.nx)] + lv.ku[IDX(im, j, lv.nx)])
                   + cy * (lv.kv[IDX(i, j, lv.nx)] + lv.kv[IDX(i, jm, lv.nx)]);
    if (lv.msk[IDX(i, j, lv.nx)] <= 0.0) d = 1.0;
    lv.dinv[IDX(i, j, lv.nx)] = 1.0 / d;
}

__global__ void k_apply(Level lv, real_t coef, const real_t *x, real_t *ax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lv.nx || j >= lv.ny) return;
    int nx = lv.nx, ny = lv.ny;
    int ip = (i == nx - 1) ? 0 : i + 1, im = (i == 0) ? nx - 1 : i - 1;
    int jp = (j == ny - 1) ? 0 : j + 1, jm = (j == 0) ? ny - 1 : j - 1;
    real_t cx = coef / (lv.dx * lv.dx), cy = coef / (lv.dy * lv.dy);
    real_t c = x[IDX(i, j, nx)];
    real_t lap = cx * (lv.ku[IDX(i, j, nx)] * (x[IDX(ip, j, nx)] - c)
                     - lv.ku[IDX(im, j, nx)] * (c - x[IDX(im, j, nx)]))
               + cy * (lv.kv[IDX(i, j, nx)] * (x[IDX(i, jp, nx)] - c)
                     - lv.kv[IDX(i, jm, nx)] * (c - x[IDX(i, jm, nx)]));
    ax[IDX(i, j, nx)] = (lv.msk[IDX(i, j, nx)] > 0.0) ? c - lap : c;
}

__global__ void k_rbgs(Level lv, real_t coef, real_t *x, const real_t *b,
                       int colour)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lv.nx || j >= lv.ny) return;
    if (((i + j) & 1) != colour) return;
    int nx = lv.nx, ny = lv.ny;
    if (lv.msk[IDX(i, j, nx)] <= 0.0) return;
    int ip = (i == nx - 1) ? 0 : i + 1, im = (i == 0) ? nx - 1 : i - 1;
    int jp = (j == ny - 1) ? 0 : j + 1, jm = (j == 0) ? ny - 1 : j - 1;
    real_t cx = coef / (lv.dx * lv.dx), cy = coef / (lv.dy * lv.dy);
    real_t nb = cx * (lv.ku[IDX(i, j, nx)] * x[IDX(ip, j, nx)]
                    + lv.ku[IDX(im, j, nx)] * x[IDX(im, j, nx)])
              + cy * (lv.kv[IDX(i, j, nx)] * x[IDX(i, jp, nx)]
                    + lv.kv[IDX(i, jm, nx)] * x[IDX(i, jm, nx)]);
    x[IDX(i, j, nx)] = lv.dinv[IDX(i, j, nx)] * (b[IDX(i, j, nx)] + nb);
}

__global__ void k_jacobi(int n, const real_t *dinv, const real_t *r, real_t *z)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) z[t] = dinv[t] * r[t];
}

__global__ void k_fill(int n, real_t *a, real_t v)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) a[t] = v;
}

__global__ void k_copy(int n, const real_t *src, real_t *dst)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) dst[t] = src[t];
}

// x += a*p ; r -= a*ap, fused so the CG update reads each array once.
__global__ void k_axpy2(int n, real_t alpha, const real_t *p, const real_t *ap,
                        real_t *x, real_t *r)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) { x[t] += alpha * p[t]; r[t] -= alpha * ap[t]; }
}

__global__ void k_sub(int n, const real_t *a, const real_t *b, real_t *out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) out[t] = a[t] - b[t];
}

__global__ void k_pupdate(int n, real_t beta, const real_t *z, real_t *p)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) p[t] = z[t] + beta * p[t];
}

__global__ void k_dot(int n, const real_t *a, const real_t *b, real_t *out)
{
    extern __shared__ real_t sh[];
    int tid = threadIdx.x;
    real_t s = 0.0;
    for (int t = blockIdx.x * blockDim.x + tid; t < n; t += blockDim.x * gridDim.x)
        s += a[t] * b[t];
    sh[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) sh[tid] += sh[tid + k];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

__global__ void k_residual_masked(Level lv, const real_t *b, const real_t *ax,
                                  real_t *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lv.nx || j >= lv.ny) return;
    real_t d = (b[IDX(i, j, lv.nx)] - ax[IDX(i, j, lv.nx)])
             * lv.msk[IDX(i, j, lv.nx)];
    out[IDX(i, j, lv.nx)] = d;
}

__global__ void k_restrict(Level fine, Level coarse)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= coarse.nx || j >= coarse.ny) return;
    int fnx = fine.nx;
    real_t s = fine.r[IDX(2 * i, 2 * j, fnx)] + fine.r[IDX(2 * i + 1, 2 * j, fnx)]
             + fine.r[IDX(2 * i, 2 * j + 1, fnx)]
             + fine.r[IDX(2 * i + 1, 2 * j + 1, fnx)];
    coarse.b[IDX(i, j, coarse.nx)] = 0.25 * s;
    coarse.x[IDX(i, j, coarse.nx)] = 0.0;
}

__global__ void k_prolong_add(Level fine, Level coarse)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= fine.nx || j >= fine.ny) return;
    fine.x[IDX(i, j, fine.nx)] +=
        coarse.x[IDX(i / 2, j / 2, coarse.nx)] * fine.msk[IDX(i, j, fine.nx)];
}

// --------------------------------------------------------------------- host
static std::string trim_quotes(std::string s)
{
    while (!s.empty() && (s.front() == ' ' || s.front() == '\'')) s.erase(0, 1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\'' || s.back() == '\n'
                          || s.back() == '\r')) s.pop_back();
    return s;
}

struct Case {
    int nx = 128, ny = 128, max_iter = 20000, n_repeat = 5, n_warmup = 1;
    double dx = 1.0, dy = 1.0, coef = 1.0, rtol = 1e-10;
    std::string datadir = ".", solver = "pcg_jacobi", outfile;
};

static void read_nml(const char *path, Case &c)
{
    FILE *f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "FATAL: cannot open %s\n", path); std::exit(1); }
    char line[1024];
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        auto eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::string key = s.substr(0, eq), val = s.substr(eq + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        // Fortran writes 1.0d-10; C wants 1.0e-10. Only for numeric keys:
        // doing it blindly turned the path .../cfd_exp/... into .../cfe_exp/...
        if (key != "datadir" && key != "solver" && key != "outfile")
            for (auto &ch : val) if (ch == 'd' || ch == 'D') ch = 'e';
        if (key == "nx") c.nx = std::atoi(val.c_str());
        else if (key == "ny") c.ny = std::atoi(val.c_str());
        else if (key == "dx") c.dx = std::atof(val.c_str());
        else if (key == "dy") c.dy = std::atof(val.c_str());
        else if (key == "coef") c.coef = std::atof(val.c_str());
        else if (key == "rtol") c.rtol = std::atof(val.c_str());
        else if (key == "max_iter") c.max_iter = std::atoi(val.c_str());
        else if (key == "n_repeat") c.n_repeat = std::atoi(val.c_str());
        else if (key == "datadir") c.datadir = trim_quotes(val);
        else if (key == "solver") c.solver = trim_quotes(val);
        else if (key == "outfile") c.outfile = trim_quotes(val);
    }
    std::fclose(f);
}

static bool read_bin(const std::string &path, std::vector<double> &a, size_t n)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    a.resize(n);
    size_t got = std::fread(a.data(), sizeof(double), n, f);
    std::fclose(f);
    return got == n;
}

static real_t *to_device(const std::vector<double> &h)
{
    real_t *d = nullptr;
    CUDA_OK(cudaMalloc(&d, h.size() * sizeof(real_t)));
    std::vector<real_t> tmp(h.begin(), h.end());
    CUDA_OK(cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(real_t),
                       cudaMemcpyHostToDevice));
    return d;
}

static dim3 grid2d(int nx, int ny, dim3 blk)
{
    return dim3((nx + blk.x - 1) / blk.x, (ny + blk.y - 1) / blk.y);
}

int main(int argc, char **argv)
{
    if (argc < 2) { std::printf("usage: %s <case.nml> [solver]\n", argv[0]); return 1; }
    Case cs;
    read_nml(argv[1], cs);
    if (argc >= 3) cs.solver = argv[2];
    const int nx = cs.nx, ny = cs.ny, n = nx * ny;

    std::vector<double> hku, hkv, hmsk, hrhs, href;
    if (!read_bin(cs.datadir + "/ku.bin", hku, n) ||
        !read_bin(cs.datadir + "/kv.bin", hkv, n) ||
        !read_bin(cs.datadir + "/mask.bin", hmsk, n) ||
        !read_bin(cs.datadir + "/rhs.bin", hrhs, n)) {
        std::fprintf(stderr, "FATAL: missing input binaries in %s\n",
                     cs.datadir.c_str());
        return 1;
    }
    bool have_ref = read_bin(cs.datadir + "/eta_ref.bin", href, n);

    // Multigrid hierarchy, built on the host then uploaded.
    std::vector<Level> lv;
    {
        Level l0;
        l0.nx = nx; l0.ny = ny; l0.dx = cs.dx; l0.dy = cs.dy;
        l0.ku = to_device(hku); l0.kv = to_device(hkv); l0.msk = to_device(hmsk);
        lv.push_back(l0);
        std::vector<double> cku = hku, ckv = hkv, cmsk = hmsk;
        int cnx = nx, cny = ny;
        double cdx = cs.dx, cdy = cs.dy;
        while (cnx % 2 == 0 && cny % 2 == 0 && cnx / 2 >= 8 && cny / 2 >= 8) {
            int mx = cnx / 2, my = cny / 2;
            std::vector<double> nku(mx * my), nkv(mx * my), nmsk(mx * my);
            for (int j = 0; j < my; ++j)
                for (int i = 0; i < mx; ++i) {
                    nku[IDX(i, j, mx)] = 0.5 * (cku[IDX(2 * i + 1, 2 * j, cnx)]
                                              + cku[IDX(2 * i + 1, 2 * j + 1, cnx)]);
                    nkv[IDX(i, j, mx)] = 0.5 * (ckv[IDX(2 * i, 2 * j + 1, cnx)]
                                              + ckv[IDX(2 * i + 1, 2 * j + 1, cnx)]);
                    double m = cmsk[IDX(2 * i, 2 * j, cnx)];
                    m = fmax(m, cmsk[IDX(2 * i + 1, 2 * j, cnx)]);
                    m = fmax(m, cmsk[IDX(2 * i, 2 * j + 1, cnx)]);
                    m = fmax(m, cmsk[IDX(2 * i + 1, 2 * j + 1, cnx)]);
                    nmsk[IDX(i, j, mx)] = m;
                }
            Level l;
            l.nx = mx; l.ny = my; l.dx = 2.0 * cdx; l.dy = 2.0 * cdy;
            l.ku = to_device(nku); l.kv = to_device(nkv); l.msk = to_device(nmsk);
            lv.push_back(l);
            cku = nku; ckv = nkv; cmsk = nmsk;
            cnx = mx; cny = my; cdx *= 2.0; cdy *= 2.0;
        }
    }
    dim3 blk(32, 8);
    for (auto &l : lv) {
        size_t bytes = size_t(l.nx) * l.ny * sizeof(real_t);
        CUDA_OK(cudaMalloc(&l.dinv, bytes));
        CUDA_OK(cudaMalloc(&l.x, bytes));
        CUDA_OK(cudaMalloc(&l.b, bytes));
        CUDA_OK(cudaMalloc(&l.r, bytes));
        k_setup_dinv<<<grid2d(l.nx, l.ny, blk), blk>>>(l, cs.coef);
    }
    CUDA_OK(cudaDeviceSynchronize());

    real_t *d_rhs = to_device(hrhs);
    real_t *d_x, *d_r, *d_z, *d_p, *d_ap, *d_scal;
    CUDA_OK(cudaMalloc(&d_x, n * sizeof(real_t)));
    CUDA_OK(cudaMalloc(&d_r, n * sizeof(real_t)));
    CUDA_OK(cudaMalloc(&d_z, n * sizeof(real_t)));
    CUDA_OK(cudaMalloc(&d_p, n * sizeof(real_t)));
    CUDA_OK(cudaMalloc(&d_ap, n * sizeof(real_t)));
    CUDA_OK(cudaMalloc(&d_scal, sizeof(real_t)));

    const int TPB = 256;
    const int nblk = (n + TPB - 1) / TPB;
    const int dot_blocks = std::min(nblk, 1024);
    dim3 g0 = grid2d(nx, ny, blk);

    auto dot = [&](const real_t *a, const real_t *b) {
        real_t zero = 0, out = 0;
        CUDA_OK(cudaMemcpy(d_scal, &zero, sizeof(real_t), cudaMemcpyHostToDevice));
        k_dot<<<dot_blocks, TPB, TPB * sizeof(real_t)>>>(n, a, b, d_scal);
        CUDA_OK(cudaMemcpy(&out, d_scal, sizeof(real_t), cudaMemcpyDeviceToHost));
        return double(out);
    };

    std::function<void(int)> vcycle = [&](int l) {
        Level &L = lv[l];
        dim3 g = grid2d(L.nx, L.ny, blk);
        for (int s = 0; s < 2; ++s) {
            k_rbgs<<<g, blk>>>(L, cs.coef, L.x, L.b, 0);
            k_rbgs<<<g, blk>>>(L, cs.coef, L.x, L.b, 1);
        }
        if (l + 1 < (int)lv.size()) {
            k_apply<<<g, blk>>>(L, cs.coef, L.x, L.r);
            k_residual_masked<<<g, blk>>>(L, L.b, L.r, L.r);
            Level &C = lv[l + 1];
            k_restrict<<<grid2d(C.nx, C.ny, blk), blk>>>(L, C);
            vcycle(l + 1);
            k_prolong_add<<<g, blk>>>(L, C);
        }
        for (int s = 0; s < 2; ++s) {
            k_rbgs<<<g, blk>>>(L, cs.coef, L.x, L.b, 1);
            k_rbgs<<<g, blk>>>(L, cs.coef, L.x, L.b, 0);
        }
    };

    auto precondition = [&](const real_t *r, real_t *z) {
        if (cs.solver == "pcg_rbgs") {
            k_fill<<<nblk, TPB>>>(n, z, 0.0);
            k_rbgs<<<g0, blk>>>(lv[0], cs.coef, z, r, 0);
            k_rbgs<<<g0, blk>>>(lv[0], cs.coef, z, r, 1);
            k_rbgs<<<g0, blk>>>(lv[0], cs.coef, z, r, 1);
            k_rbgs<<<g0, blk>>>(lv[0], cs.coef, z, r, 0);
        } else {
            k_jacobi<<<nblk, TPB>>>(n, lv[0].dinv, r, z);
        }
    };

    int iters = 0;
    double residual = 0.0;

    auto run_pcg = [&]() {
        k_fill<<<nblk, TPB>>>(n, d_x, 0.0);
        k_copy<<<nblk, TPB>>>(n, d_rhs, d_r);
        double norm_b = std::sqrt(dot(d_rhs, d_rhs));
        iters = 0; residual = norm_b;
        if (norm_b == 0.0) return;
        double tol = cs.rtol * norm_b;
        precondition(d_r, d_z);
        k_copy<<<nblk, TPB>>>(n, d_z, d_p);
        double rz = dot(d_r, d_z);
        // Reduced precision cannot reach an arbitrary tolerance; stop when the
        // residual stops improving rather than spinning to max_iter. ONLY in
        // the fp32 build: CG's residual is not monotone and a converging fp64
        // solve can go forty iterations without beating its best (docs/90 N8).
        const bool detect_stall = sizeof(real_t) < 8;
        double best = 1e300; int stall = 0;
        for (int it = 1; it <= cs.max_iter; ++it) {
            k_apply<<<g0, blk>>>(lv[0], cs.coef, d_p, d_ap);
            double alpha = rz / dot(d_p, d_ap);
            k_axpy2<<<nblk, TPB>>>(n, alpha, d_p, d_ap, d_x, d_r);
            residual = std::sqrt(dot(d_r, d_r));
            iters = it;
            if (residual <= tol) break;
            if (detect_stall) {
                if (residual < best * 0.99) { best = residual; stall = 0; }
                else if (++stall >= 40) break;
            }
            precondition(d_r, d_z);
            double rz_new = dot(d_r, d_z);
            k_pupdate<<<nblk, TPB>>>(n, rz_new / rz, d_z, d_p);
            rz = rz_new;
        }
        residual /= norm_b;
    };

    auto run_smoother = [&](bool use_mg) {
        double norm_b = std::sqrt(dot(d_rhs, d_rhs));
        double tol = cs.rtol * norm_b;
        k_fill<<<nblk, TPB>>>(n, lv[0].x, 0.0);
        k_copy<<<nblk, TPB>>>(n, d_rhs, lv[0].b);
        iters = 0; residual = norm_b;
        const bool detect_stall = sizeof(real_t) < 8;
        double best = 1e300; int stall = 0;
        for (int it = 1; it <= cs.max_iter; ++it) {
            if (use_mg) {
                vcycle(0);
            } else {
                k_rbgs<<<g0, blk>>>(lv[0], cs.coef, lv[0].x, lv[0].b, 0);
                k_rbgs<<<g0, blk>>>(lv[0], cs.coef, lv[0].x, lv[0].b, 1);
            }
            k_apply<<<g0, blk>>>(lv[0], cs.coef, lv[0].x, d_ap);
            k_sub<<<nblk, TPB>>>(n, d_rhs, d_ap, d_r);
            residual = std::sqrt(dot(d_r, d_r));
            iters = it;
            if (residual <= tol) break;
            if (detect_stall) {
                if (residual < best * 0.99) { best = residual; stall = 0; }
                else if (++stall >= 40) break;
            }
        }
        k_copy<<<nblk, TPB>>>(n, lv[0].x, d_x);
        residual /= (norm_b > 0.0 ? norm_b : 1.0);
    };

    auto run = [&]() {
        if (cs.solver == "pcg_jacobi" || cs.solver == "pcg_rbgs") run_pcg();
        else if (cs.solver == "rbgs") run_smoother(false);
        else if (cs.solver == "multigrid") run_smoother(true);
        else { std::fprintf(stderr, "FATAL: unknown solver %s\n",
                            cs.solver.c_str()); std::exit(1); }
        CUDA_OK(cudaDeviceSynchronize());
    };

    for (int w = 0; w < cs.n_warmup; ++w) run();
    /* R7-2: median + MAD over the repeats; the minimum stays a secondary column. */
    std::vector<double> samples;
    for (int rep = 0; rep < cs.n_repeat; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        run();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    auto median_of = [](std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size();
                                                  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); };
    const double wall = median_of(samples);
    const double wall_min = *std::min_element(samples.begin(), samples.end());
    std::vector<double> dev; for (double x : samples) dev.push_back(std::fabs(x - wall));
    const double wall_mad = median_of(dev);

    std::vector<real_t> hxr(n);
    CUDA_OK(cudaMemcpy(hxr.data(), d_x, n * sizeof(real_t), cudaMemcpyDeviceToHost));
    std::vector<double> hx(hxr.begin(), hxr.end());
    double err = -1.0;
    if (have_ref) {
        double num = 0.0, den = 0.0;
        for (int t = 0; t < n; ++t) { double d = hx[t] - href[t];
                                      num += d * d; den += href[t] * href[t]; }
        err = std::sqrt(num) / (den > 0.0 ? std::sqrt(den) : 1.0);
    }

    std::string out = cs.outfile.empty()
        ? cs.datadir + "/metrics_cuda_" + cs.solver + ".json" : cs.outfile;
    FILE *f = std::fopen(out.c_str(), "w");
    std::fprintf(f, "{\n  \"backend\": \"cuda\",\n  \"precision\": \"%s\",\n"
                    "  \"solver\": \"%s\",\n", PRECISION_NAME, cs.solver.c_str());
    std::fprintf(f, "  \"nx\": %d,\n  \"ny\": %d,\n", nx, ny);
    std::fprintf(f, "  \"iterations\": %d,\n", iters);
    std::fprintf(f, "  \"wall_s\": %.16e,\n", wall);
    std::fprintf(f, "  \"wall_mad_s\": %.16e,\n", wall_mad);
    std::fprintf(f, "  \"wall_min_s\": %.16e,\n", wall_min);
    std::fprintf(f, "  \"residual\": %.16e,\n", residual);
    std::fprintf(f, "  \"l2_rel_vs_reference\": %.16e,\n", err);
    std::fprintf(f, "  \"n_repeat\": %d\n}\n", cs.n_repeat);
    std::fclose(f);
    std::printf("solver=%12s iters=%8d wall=%12.5e l2_vs_ref=%12.5e\n",
                cs.solver.c_str(), iters, wall, err);
    return 0;
}
