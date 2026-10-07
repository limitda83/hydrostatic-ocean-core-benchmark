/************************************************************************
 *  Module: module_boundary                                             *
 *  Description: E17 - what a per-timestep MODULE BOUNDARY costs. If an *
 *               empirical closure is replaced by a learned module that *
 *               lives on the other side of a device boundary, every    *
 *               step pays a round trip of that closure's inputs and    *
 *               outputs. This measures the round trip for the TKE      *
 *               closure's REAL footprint (the signature of             *
 *               libs/core/closure.py::tke_step), so it can be compared *
 *               with the in-place closure cost the model itself        *
 *               measures (docs/31 S3b).                                *
 *               It is a LOWER BOUND: a real module also has to run.    *
 *  Pipeline: module_boundary -> expr/E17 -> docs/37 S6                 *
 ************************************************************************/

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* A silent failure here would look like a very fast transfer, which is the
 * same class of bug as docs/90 N25 (a launch error read as a result). */
#define CK(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, \
                         #call, cudaGetErrorString(_e));                       \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

/* Values per column, counted from the code - not estimated. Cross-checked
 * against BOTH implementations, which do not read exactly the same set:
 *   libs/core/closure.py::tke_step  - u,v,b,e,K_m_old,K_h_old + dz3,mask3
 *   k_closure in cfd_exp3d5_cuda.cu - the same, plus the face thickness and
 *                                     face masks dz3u,dz3v,masku,maskv
 * The extra four are topography-derived and constant in a `zlevel` run, so
 * they go in the STATIC bucket, which a sane implementation uploads once and
 * which is excluded unless --static-every-step is given.
 *   dynamic in : u,v,b on nz centres; e,K_m_old,K_h_old on nz+1 interfaces
 *   static     : dz3,mask3,dz3u,dz3v,masku,maskv on nz centres
 *   out        : e_new,K_m,K_h on nz+1 interfaces                          */
static const int N_CELL_IN = 3, N_IFACE_IN = 3, N_CELL_STATIC = 6, N_IFACE_OUT = 3;

struct Stat { double median, mad, min; };

static Stat stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const double med = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    std::vector<double> d(n);
    for (size_t i = 0; i < n; ++i) d[i] = std::fabs(v[i] - med);
    std::sort(d.begin(), d.end());
    const double mad = (n % 2) ? d[n / 2] : 0.5 * (d[n / 2 - 1] + d[n / 2]);
    return {med, mad, v.front()};
}

/* One CUDA-event-timed sample per repeat; every sample ends synchronized. */
template <typename F>
static Stat timed(F op, int n_repeat, int n_warmup) {
    for (int i = 0; i < n_warmup; ++i) op();
    CK(cudaDeviceSynchronize());
    std::vector<double> s;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < n_repeat; ++i) {
        CK(cudaEventRecord(a));
        op();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        s.push_back(ms / 1e3);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return stats(s);
}

int main(int argc, char **argv) {
    int nx = 400, nz = 30, itemsize = 8, n_repeat = 5, n_warmup = 3, statics = 0;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() { return std::atoi(argv[++i]); };
        if (k == "--nx") nx = val();
        else if (k == "--nz") nz = val();
        else if (k == "--itemsize") itemsize = val();
        else if (k == "--n-repeat") n_repeat = val();
        else if (k == "--n-warmup") n_warmup = val();
        else if (k == "--static-every-step") statics = 1;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (n_repeat < 5) { std::fprintf(stderr, "R7-2: n_repeat floor is 5\n"); return 2; }

    const size_t cols = (size_t)nx * (size_t)nx;
    const size_t per_in = (size_t)N_CELL_IN * nz + (size_t)N_IFACE_IN * (nz + 1);
    const size_t per_st = (size_t)N_CELL_STATIC * nz;
    const size_t per_out = (size_t)N_IFACE_OUT * (nz + 1);
    const size_t n_in = cols * (per_in + (statics ? per_st : 0)) * itemsize;
    const size_t n_out = cols * per_out * itemsize;

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));

    void *d_in, *d_out, *hp_in, *hp_out;
    CK(cudaMalloc(&d_in, n_in));
    CK(cudaMalloc(&d_out, n_out));
    CK(cudaMemset(d_in, 1, n_in));
    CK(cudaMemset(d_out, 1, n_out));
    CK(cudaMallocHost(&hp_in, n_in));            /* pinned */
    CK(cudaMallocHost(&hp_out, n_out));
    void *hg_in = std::malloc(n_in);             /* pageable */
    void *hg_out = std::malloc(n_out);
    if (!hg_in || !hg_out) { std::fprintf(stderr, "host malloc failed\n"); return 2; }
    std::memset(hg_in, 1, n_in);
    std::memset(hg_out, 1, n_out);

    /* B: host model ON the device, module elsewhere - inputs go out, answer back.
     * C: host model on the CPU, module on the device - same bytes, other way.
     * Both are real per-step round trips through the host.
     *
     * DISCARDED 2026-09-15 (external review, docs/90 N44): a "device-to-device"
     * case D and a "two-stream overlap" case E used to live here. D copied the
     * footprint into second device buffers - a cost an in-place module never
     * pays, so it measured nothing a real design must incur and could not be
     * called a lower bound. E split one D2H over two streams with nothing to
     * overlap, so it could not speak to overlap at all. Neither is a
     * measurement of the thing this experiment is about. */
    auto B = [&](void *h_in, void *h_out) {
        CK(cudaMemcpy(h_in, d_in, n_in, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(d_out, h_out, n_out, cudaMemcpyHostToDevice));
    };
    auto C = [&](void *h_in, void *h_out) {
        CK(cudaMemcpy(d_in, h_in, n_in, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(h_out, d_out, n_out, cudaMemcpyDeviceToHost));
    };

    struct Row { const char *name; Stat s; };
    std::vector<Row> rows;
    rows.push_back({"B_dev_to_host_roundtrip_pinned",
                    timed([&] { B(hp_in, hp_out); }, n_repeat, n_warmup)});
    rows.push_back({"B_dev_to_host_roundtrip_pageable",
                    timed([&] { B(hg_in, hg_out); }, n_repeat, n_warmup)});
    rows.push_back({"C_host_to_dev_roundtrip_pinned",
                    timed([&] { C(hp_in, hp_out); }, n_repeat, n_warmup)});
    rows.push_back({"C_host_to_dev_roundtrip_pageable",
                    timed([&] { C(hg_in, hg_out); }, n_repeat, n_warmup)});

    std::printf("gpu,cc,nx,nz,itemsize,static_every_step,columns,bytes_in,bytes_out,"
                "case,median_s,mad_s,min_s,n_repeat,gbytes_per_s\n");
    for (const Row &r : rows) {
        const size_t bytes = n_in + n_out;
        std::printf("%s,%d.%d,%d,%d,%d,%d,%zu,%zu,%zu,%s,%.9e,%.9e,%.9e,%d,%.6f\n",
                    p.name, p.major, p.minor, nx, nz, itemsize, statics, cols,
                    n_in, n_out, r.name, r.s.median, r.s.mad, r.s.min, n_repeat,
                    bytes / r.s.median / 1e9);
    }
    CK(cudaFreeHost(hp_in));
    CK(cudaFreeHost(hp_out));
    std::free(hg_in);
    std::free(hg_out);
    CK(cudaFree(d_in));
    CK(cudaFree(d_out));
    return 0;
}
