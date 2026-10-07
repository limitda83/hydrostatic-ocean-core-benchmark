/************************************************************************
 *  Module: eos_bench (CUDA)                                            *
 *  Description: Native CUDA counterpart of libs/fortran/src/           *
 *               eos_bench.f90. Same three equations of state, same     *
 *               inputs, so the checksum must match the Fortran and     *
 *               NumPy ones (R2). RQ7: the EOS is the only compute-     *
 *               bound kernel in the core, so it is where the GPU-to-   *
 *               CPU ratio should be largest.                           *
 *  Pipeline: eos_bench -> docs/25                                      *
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>

#define CUDA_OK(call)                                                        \
    do { cudaError_t _e = (call);                                            \
         if (_e != cudaSuccess) {                                            \
             std::fprintf(stderr, "FATAL: %s at %s:%d\n",                    \
                          cudaGetErrorString(_e), __FILE__, __LINE__);       \
             std::exit(1); } } while (0)

__constant__ double RDELTA_S = 32.0;
#define R1_S0 (0.875 / 35.16504)
#define R1_T0 (1.0 / 40.0)
#define R1_Z0 1.0e-4

__global__ void k_linear(int n, const double *t, const double *s,
                         const double *z, double *rho,
                         double alpha, double beta, double rho0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) rho[i] = rho0 * (-alpha * (t[i] - 10.0) + beta * (s[i] - 35.0));
}

__global__ void k_seos(int n, const double *t, const double *s,
                       const double *z, double *rho)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double a0 = 1.6550e-1, b0 = 7.6554e-1, l1 = 5.9520e-2,
                 l2 = 7.4914e-4, m1 = 1.4970e-4, m2 = 1.1090e-5,
                 nu = 2.4341e-3;
    double ta = t[i] - 10.0, sa = s[i] - 35.0, zz = z[i];
    rho[i] = -a0 * (1.0 + 0.5 * l1 * ta + m1 * zz) * ta
           +  b0 * (1.0 - 0.5 * l2 * sa - m2 * zz) * sa
           -  nu * ta * sa;
}

__global__ void k_teos10(int n, const double *t, const double *s,
                         const double *z, double *rho)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double R00= 4.6494977072e+01, R01=-5.2099962525e+00,
                 R02= 2.2601900708e-01, R03= 6.4326772569e-02,
                 R04= 1.5616995503e-02, R05=-1.7243708991e-03;
    const double E000= 8.0189615746e+02, E100= 8.6672408165e+02,
                 E200=-1.7864682637e+03, E300= 2.0375295546e+03,
                 E400=-1.2849161071e+03, E500= 4.3227585684e+02,
                 E600=-6.0579916612e+01, E010= 2.6010145068e+01,
                 E110=-6.5281885265e+01, E210= 8.1770425108e+01,
                 E310=-5.6888046321e+01, E410= 1.7681814114e+01,
                 E510=-1.9193502195e+00, E020=-3.7074170417e+01,
                 E120= 6.1548258127e+01, E220=-6.0362551501e+01,
                 E320= 2.9130021253e+01, E420=-5.4723692739e+00,
                 E030= 2.1661789529e+01, E130=-3.3449108469e+01,
                 E230= 1.9717078466e+01, E330=-3.1742946532e+00,
                 E040=-8.3627885467e+00, E140= 1.1311538584e+01,
                 E240=-5.3563304045e+00, E050= 5.4048723791e-01,
                 E150= 4.5111434961e-01, E060=-1.9098268277e-01,
                 E001= 1.9681925209e+01, E101=-4.2549998214e+01,
                 E201= 5.0774768218e+01, E301=-3.0938076334e+01,
                 E401= 6.6051753097e+00, E011=-1.3336301113e+01,
                 E111=-4.4870114575e+00, E211= 5.0042598061e+00,
                 E311=-6.5399043664e-01, E021= 6.7080479603e+00,
                 E121= 3.5063081279e+00, E221=-1.8795372996e+00,
                 E031=-2.4649669534e+00, E131=-5.5077101279e-01,
                 E041= 5.5927935970e-01, E002= 2.0660924175e+00,
                 E102=-4.9527603989e+00, E202= 2.5019633803e+00,
                 E012= 2.0564311499e+00, E112=-2.1311365518e-01,
                 E022=-1.2419983026e+00, E003=-2.3342758797e-02,
                 E103=-1.8507636718e-02, E013= 3.7969820455e-01;
    double zh = z[i] * R1_Z0;
    double ss = sqrt((s[i] + RDELTA_S) * R1_S0);
    double tt = t[i] * R1_T0;
    double r0 = (((((R05*zh + R04)*zh + R03)*zh + R02)*zh + R01)*zh + R00)*zh;
    double rz3 = E013*tt + E103*ss + E003;
    double rz2 = (E022*tt + E112*ss + E012)*tt + (E202*ss + E102)*ss + E002;
    double rz1 = (((E041*tt + E131*ss + E031)*tt
                   + (E221*ss + E121)*ss + E021)*tt
                  + ((E311*ss + E211)*ss + E111)*ss + E011)*tt
               + (((E401*ss + E301)*ss + E201)*ss + E101)*ss + E001;
    double rz0 = (((((E060*tt + E150*ss + E050)*tt
                     + (E240*ss + E140)*ss + E040)*tt
                    + ((E330*ss + E230)*ss + E130)*ss + E030)*tt
                   + (((E420*ss + E320)*ss + E220)*ss + E120)*ss + E020)*tt
                  + ((((E510*ss + E410)*ss + E310)*ss + E210)*ss + E110)*ss
                  + E010)*tt
               + (((((E600*ss + E500)*ss + E400)*ss + E300)*ss + E200)*ss
                  + E100)*ss + E000;
    rho[i] = ((rz3*zh + rz2)*zh + rz1)*zh + rz0 + r0;
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? std::atoi(argv[1]) : 3000000;
    std::string kind = (argc > 2) ? argv[2] : "teos10";
    int n_repeat = (argc > 3) ? std::atoi(argv[3]) : 20;

    std::vector<double> ht(n), hs(n), hz(n), hrho(n);
    for (int i = 1; i <= n; ++i) {
        ht[i-1] = 5.0 + 15.0 * double(i % 977) / 977.0;
        hs[i-1] = 33.0 + 3.0 * double(i % 733) / 733.0;
        hz[i-1] = 5000.0 * double(i % 521) / 521.0;
    }
    double *dt, *ds, *dz, *drho;
    size_t bytes = size_t(n) * sizeof(double);
    CUDA_OK(cudaMalloc(&dt, bytes));  CUDA_OK(cudaMalloc(&ds, bytes));
    CUDA_OK(cudaMalloc(&dz, bytes));  CUDA_OK(cudaMalloc(&drho, bytes));
    CUDA_OK(cudaMemcpy(dt, ht.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ds, hs.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dz, hz.data(), bytes, cudaMemcpyHostToDevice));

    const int TPB = 256, nb = (n + TPB - 1) / TPB;
    auto run = [&]() {
        if (kind == "linear") k_linear<<<nb, TPB>>>(n, dt, ds, dz, drho,
                                                    2.0e-4, 7.4e-4, 1025.0);
        else if (kind == "seos") k_seos<<<nb, TPB>>>(n, dt, ds, dz, drho);
        else if (kind == "teos10") k_teos10<<<nb, TPB>>>(n, dt, ds, dz, drho);
        else { std::fprintf(stderr, "FATAL: unknown eos %s\n", kind.c_str());
               std::exit(1); }
        CUDA_OK(cudaDeviceSynchronize());
    };
    for (int w = 0; w < 3; ++w) run();
    /* R7-2: median + MAD of the repeats, not the best of them. */
    std::vector<double> samples;
    for (int r = 0; r < n_repeat; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        run();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    auto median_of = [](std::vector<double> v) { std::sort(v.begin(), v.end()); size_t m = v.size();
                                                  return m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]); };
    const double wall = median_of(samples);
    std::vector<double> dev; for (double x : samples) dev.push_back(std::fabs(x - wall));
    const double wall_mad = median_of(dev);
    CUDA_OK(cudaMemcpy(hrho.data(), drho, bytes, cudaMemcpyDeviceToHost));
    double chk = 0.0;
    int step = std::max(1, n / 1000);
    for (int i = 0; i < n; i += step) chk += hrho[i];
    std::printf("eos=%10s n=%10d wall=%13.6e mad=%13.6e checksum=%13.6e\n",
                kind.c_str(), n, wall, wall_mad, chk);
    return 0;
}
