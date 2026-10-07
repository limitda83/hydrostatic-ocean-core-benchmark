/************************************************************************
 *  Module: cfd_exp3d5_cuda_multi                                       *
 *  Description: Single-process multi-GPU driver for the spec v0.5/v0.6 *
 *               CUDA core (E19). The domain is cut into y-slabs, one   *
 *               per device, each carried with a deep halo of H rows on *
 *               both sides. Every device runs the UNMODIFIED single-    *
 *               device step on its extended slab; halos are refreshed   *
 *               once per step by peer copies (P2P over PCIe/NVLink, or  *
 *               staged through the host when P2P is unavailable). With  *
 *               H larger than the stencil-chain depth of one step, the  *
 *               interior of every slab is bit-identical to the single-  *
 *               device run (verified by the gate, docs/03 S12).         *
 *               Explicit forward-backward scheme only: the theta scheme *
 *               needs a distributed elliptic solve and the split scheme *
 *               a halo per sub-step; both are out of scope (docs/41).   *
 *  Pipeline: build with `make multi`; run                              *
 *            cfd_exp3d5_cuda_multi <namelist> --ndev N [--halo H]      *
 *                                   [--devices 0,1,...]                 *
 ************************************************************************/
#define CFD_MULTI_NO_MAIN
#include "cfd_exp3d5_cuda.cu"

namespace {

struct Slab {
    int dev = 0;              /* CUDA device id */
    int j0 = 0, ny_loc = 0;   /* first global row owned, rows owned */
    int ny_ext = 0;           /* ny_loc + 2H */
    Stepper5* st = nullptr;
    double *u = nullptr, *v = nullptr, *b = nullptr, *t = nullptr, *s = nullptr, *eta = nullptr;
    double* part = nullptr;   /* divergence-check partials */
    double *send_lo = nullptr, *send_hi = nullptr, *recv_lo = nullptr, *recv_hi = nullptr;  /* halo buffers */
    size_t buf_len = 0;       /* doubles per buffer */
};

/* rows of a 2-D (nlev = 1) or 3-D (nlev = nz or nz+1) host array, in local order */
static std::vector<double> slice_rows(const std::vector<double>& g, int nx, int ny, int nlev, const std::vector<int>& rows) {
    std::vector<double> out(static_cast<size_t>(nx) * rows.size() * nlev);
    for (int k = 0; k < nlev; ++k)
        for (size_t r = 0; r < rows.size(); ++r)
            std::memcpy(out.data() + (static_cast<size_t>(k) * rows.size() + r) * nx,
                        g.data() + (static_cast<size_t>(k) * ny + rows[r]) * nx, nx * sizeof(double));
    return out;
}

/* halo rows are packed into contiguous send buffers on the owning device, copied with one
 * cudaMemcpyPeerAsync per neighbour, and unpacked on the receiving device: the strided
 * cudaMemcpy3DPeer path is an order of magnitude slower (E19, docs/42). */
__global__ void k_pack(const double* a, double* buf, int nx, int ny_ext, int r0, int H, int nlev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, k = blockIdx.z;
    if (i < nx && r < H && k < nlev) buf[i + nx * (r + H * k)] = a[i + nx * (r0 + r + ny_ext * k)];
}
__global__ void k_unpack(double* a, const double* buf, int nx, int ny_ext, int r0, int H, int nlev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, k = blockIdx.z;
    if (i < nx && r < H && k < nlev) a[i + nx * (r0 + r + ny_ext * k)] = buf[i + nx * (r + H * k)];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: cfd_exp3d5_cuda_multi <namelist> [--ndev N] [--halo H] [--devices 0,1,..]\n"); return 1; }
    int ndev = 1, halo = 16; std::vector<int> dev_ids;
    for (int a = 2; a < argc; ++a) {
        std::string k = argv[a];
        if (k == "--ndev" && a + 1 < argc) ndev = std::atoi(argv[++a]);
        else if (k == "--halo" && a + 1 < argc) halo = std::atoi(argv[++a]);
        else if (k == "--devices" && a + 1 < argc) { std::string l = argv[++a]; size_t p = 0; while (p <= l.size()) { size_t q = l.find(',', p); if (q == std::string::npos) q = l.size(); dev_ids.push_back(std::atoi(l.substr(p, q - p).c_str())); p = q + 1; } }
    }
    if (dev_ids.empty()) for (int g = 0; g < ndev; ++g) dev_ids.push_back(g);
    ndev = static_cast<int>(dev_ids.size());
    const RunConfig c = read_config(argv[1]);
    const V05Config v = read_v05(argv[1]);
    if (c.scheme_name != "fb") { std::fprintf(stderr, "FATAL: multi-device driver supports scheme='fb' only (got %s)\n", c.scheme_name.c_str()); return 2; }
    const int nx = c.nx, ny = c.ny, nz = c.nz, n2 = nx * ny, n3 = n2 * nz;
    const bool ts = v.tracers == "TS";
    if (ny / ndev < halo) { std::fprintf(stderr, "FATAL: ny/ndev = %d < halo = %d\n", ny / ndev, halo); return 2; }

    /* ---- global domain and initial state, exactly as the single-device program ---- */
    std::vector<double> io2(n2), io3(n3);
    Domain dom; dom.h.resize(n2); dom.mask.resize(n2);
    { FILE* f = std::fopen(v.domain_file.c_str(), "rb");
      if (!f) { std::fprintf(stderr, "FATAL: cannot read %s\n", v.domain_file.c_str()); return 1; }
      std::fread(io2.data(), sizeof(double), n2, f); dom.h.assign(io2.begin(), io2.end());
      std::fread(io2.data(), sizeof(double), n2, f); dom.mask.assign(io2.begin(), io2.end());
      std::fclose(f); }
    dom.build(v, nx, ny, nz);
    std::vector<double> e0(n2), u0(n3), v0(n3), b0(n3), t0(n3, 0.0), s0(n3, 0.0);
    { FILE* f = std::fopen(v.init_file.c_str(), "rb");
      if (!f) { std::fprintf(stderr, "FATAL: cannot read %s\n", v.init_file.c_str()); return 1; }
      auto rd2 = [&](std::vector<double>& dst) { std::fread(dst.data(), sizeof(double), n2, f); };
      auto rd3 = [&](std::vector<double>& dst) { std::fread(dst.data(), sizeof(double), n3, f); };
      rd2(e0); rd3(u0); rd3(v0); rd3(b0);
      if (ts) { rd3(t0); rd3(s0); }
      std::fclose(f); }

    /* ---- slabs ---- */
    std::vector<Slab> sl(ndev);
    std::vector<std::string> dev_names;
    size_t dev_bytes_total = 0;
    for (int g = 0; g < ndev; ++g) {
        Slab& S = sl[g]; S.dev = dev_ids[g];
        S.j0 = (ny * g) / ndev; S.ny_loc = (ny * (g + 1)) / ndev - S.j0; S.ny_ext = S.ny_loc + 2 * halo;
        std::vector<int> rows(S.ny_ext);
        for (int r = 0; r < S.ny_ext; ++r) rows[r] = ((S.j0 - halo + r) % ny + ny) % ny;
        Domain dl; dl.nx = nx; dl.ny = S.ny_ext; dl.nz = nz; dl.hmax = dom.hmax;
        for (auto pr : {std::pair<std::vector<double>*, const std::vector<double>*>{&dl.h, &dom.h}, {&dl.mask, &dom.mask}, {&dl.masku, &dom.masku},
                        {&dl.maskv, &dom.maskv}, {&dl.hu, &dom.hu}, {&dl.hv, &dom.hv}, {&dl.hcu, &dom.hcu}, {&dl.hcv, &dom.hcv},
                        {&dl.inv_hcu, &dom.inv_hcu}, {&dl.inv_hcv, &dom.inv_hcv}})
            *pr.first = slice_rows(*pr.second, nx, ny, 1, rows);
        for (auto pr : {std::pair<std::vector<double>*, const std::vector<double>*>{&dl.dz3, &dom.dz3}, {&dl.dz3u, &dom.dz3u}, {&dl.dz3v, &dom.dz3v},
                        {&dl.mask3, &dom.mask3}, {&dl.mask3u, &dom.mask3u}, {&dl.mask3v, &dom.mask3v}, {&dl.inv_dz3, &dom.inv_dz3}, {&dl.zc, &dom.zc}})
            *pr.first = slice_rows(*pr.second, nx, ny, nz, rows);
        RunConfig cl = c; cl.ny = S.ny_ext;
        CUDA_OK(cudaSetDevice(S.dev));
        cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, S.dev)); dev_names.push_back(prop.name);
        for (int h = 0; h < ndev; ++h) if (h != g && dev_ids[h] != S.dev) {
            int can = 0; CUDA_OK(cudaDeviceCanAccessPeer(&can, S.dev, dev_ids[h]));
            if (can) { cudaError_t e = cudaDeviceEnablePeerAccess(dev_ids[h], 0); if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_OK(e); }
        }
        const int n2l = nx * S.ny_ext, n3l = n2l * nz;
        S.u = dev_zero(n3l); S.v = dev_zero(n3l); S.b = dev_zero(n3l); S.t = dev_zero(n3l); S.s = dev_zero(n3l); S.eta = dev_zero(n2l);
        S.part = dev_zero(1024);
        size_t fb = 0, fa = 0, tot = 0; CUDA_OK(cudaMemGetInfo(&fb, &tot));
        S.st = new Stepper5(cl, v, std::move(dl));
        S.st->d.dx = c.lx / c.nx; S.st->d.dy = c.ly / c.ny;   /* the global spacing, bit-identical to the single-device run */
        S.st->build_coefficients(false);
        CUDA_OK(cudaMemGetInfo(&fa, &tot)); dev_bytes_total += (fb - fa);
        if (S.st->is_split || S.st->is_theta) { std::fprintf(stderr, "FATAL: fb only\n"); return 2; }
        { size_t per = static_cast<size_t>(nx) * halo; size_t len = per * (nz * (ts ? 5 : 3) + 1 + (S.st->is_tke ? 3 * (nz + 1) : 0));
          S.buf_len = len; S.send_lo = dev_zero(len); S.send_hi = dev_zero(len); S.recv_lo = dev_zero(len); S.recv_hi = dev_zero(len); }
    }
    auto for_each_dev = [&](const std::function<void(Slab&)>& fn) { for (int g = 0; g < ndev; ++g) { CUDA_OK(cudaSetDevice(sl[g].dev)); fn(sl[g]); } };
    auto sync_all = [&]() { for_each_dev([](Slab&) { CUDA_OK(cudaDeviceSynchronize()); }); };

    /* halo refresh: bottom halo <- left neighbour's last H owned rows, top halo <- right neighbour's first H owned rows */
    auto halo_arrays = [&](Slab& S, std::vector<std::pair<double*, int>>& out) {
        out.clear();
        out.push_back({S.u, nz}); out.push_back({S.v, nz}); out.push_back({S.b, nz});
        if (ts) { out.push_back({S.t, nz}); out.push_back({S.s, nz}); }
        out.push_back({S.eta, 1});
        if (S.st->is_tke) { out.push_back({S.st->d.tke, nz + 1}); out.push_back({S.st->d.nu3, nz + 1}); out.push_back({S.st->d.kap3, nz + 1}); }
    };
    std::vector<std::pair<double*, int>> arrs;
    auto exchange = [&]() {
        const dim3 pb(256, 1, 1);
        for_each_dev([&](Slab& S) {                       /* pack first H owned rows -> send_lo, last H owned rows -> send_hi */
            halo_arrays(S, arrs); size_t off = 0;
            for (auto& a : arrs) {
                dim3 pg((nx + 255) / 256, halo, a.second);
                k_pack<<<pg, pb>>>(a.first, S.send_lo + off, nx, S.ny_ext, halo, halo, a.second);
                k_pack<<<pg, pb>>>(a.first, S.send_hi + off, nx, S.ny_ext, S.ny_loc, halo, a.second);
                off += static_cast<size_t>(nx) * halo * a.second;
            }
        });
        sync_all();
        for (int g = 0; g < ndev; ++g) {                  /* contiguous peer copies */
            Slab& S = sl[g]; const Slab& L = sl[(g - 1 + ndev) % ndev]; const Slab& R = sl[(g + 1) % ndev];
            CUDA_OK(cudaSetDevice(S.dev));
            CUDA_OK(cudaMemcpyPeerAsync(S.recv_lo, S.dev, L.send_hi, L.dev, S.buf_len * sizeof(double), 0));
            CUDA_OK(cudaMemcpyPeerAsync(S.recv_hi, S.dev, R.send_lo, R.dev, S.buf_len * sizeof(double), 0));
        }
        sync_all();
        for_each_dev([&](Slab& S) {                       /* unpack into the halo rows */
            halo_arrays(S, arrs); size_t off = 0;
            for (auto& a : arrs) {
                dim3 pg((nx + 255) / 256, halo, a.second);
                k_unpack<<<pg, pb>>>(a.first, S.recv_lo + off, nx, S.ny_ext, 0, halo, a.second);
                k_unpack<<<pg, pb>>>(a.first, S.recv_hi + off, nx, S.ny_ext, halo + S.ny_loc, halo, a.second);
                off += static_cast<size_t>(nx) * halo * a.second;
            }
        });
        sync_all();
    };
    auto reset = [&]() {
        for_each_dev([&](Slab& S) {
            std::vector<int> rows(S.ny_ext); for (int r = 0; r < S.ny_ext; ++r) rows[r] = ((S.j0 - halo + r) % ny + ny) % ny;
            const int n2l = nx * S.ny_ext, n3l = n2l * nz, n3il = n2l * (nz + 1);
            auto up = [&](double* d, const std::vector<double>& h) { CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice)); };
            up(S.u, slice_rows(u0, nx, ny, nz, rows)); up(S.v, slice_rows(v0, nx, ny, nz, rows)); up(S.b, slice_rows(b0, nx, ny, nz, rows));
            up(S.t, slice_rows(t0, nx, ny, nz, rows)); up(S.s, slice_rows(s0, nx, ny, nz, rows)); up(S.eta, slice_rows(e0, nx, ny, 1, rows));
            (void)n3l;
            if (S.st->is_tke) {
                std::vector<double> nu_i(n3il, c.nu), kap_i(n3il, c.kappa), tke0(n3il, v.e_min);
                up(S.st->d.nu3, nu_i); up(S.st->d.kap3, kap_i); up(S.st->d.tke, tke0);
                S.st->build_coefficients(false);
            }
        });
        sync_all();
    };
    std::vector<double> h_part(1024);
    auto diverged = [&]() -> bool {
        double m = 0.0;
        for_each_dev([&](Slab& S) {
            const int n2l = nx * S.ny_ext; int blocks = std::min(1024, (n2l + 255) / 256);
            k1_absmax<<<blocks, 256, 256 * sizeof(double)>>>(n2l, S.eta, S.part);
            CUDA_OK(cudaMemcpy(h_part.data(), S.part, blocks * sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < blocks; ++i) if (!(h_part[i] <= m)) m = h_part[i];
        });
        return !(m < 1.0e6);
    };
    auto one_step = [&]() {
        for_each_dev([&](Slab& S) { S.st->step(S.u, S.v, S.b, S.eta, S.t, S.s); });
        sync_all();
        exchange();
    };

    std::printf("cuda3d5-multi: case=%s nx=%d ny=%d nz=%d scheme=%s ndev=%d halo=%d devices=", c.case_name.c_str(), nx, ny, nz, c.scheme_name.c_str(), ndev, halo);
    for (int g = 0; g < ndev; ++g) std::printf("%d%s", sl[g].dev, g + 1 < ndev ? "," : ""); std::printf(" gpu=%s steps=%d\n", dev_names[0].c_str(), c.n_steps);
    bool run_diverged = false;
    for (int rep = 0; rep < c.n_warmup && !run_diverged; ++rep) {
        reset();
        for (int s = 0; s < c.n_steps; ++s) { one_step(); if (s == 0) CUDA_LAUNCH_OK("first step"); if (s % 10 == 9 && diverged()) { run_diverged = true; break; } }
        sync_all(); CUDA_LAUNCH_OK("warm-up repeat");
    }
    for (int g = 0; g < ndev; ++g) { sl[g].st->substeps = 0; sl[g].st->tridiag = 0; }
    std::vector<double> samples;
    for (int rep = 0; rep < c.n_repeat && !run_diverged; ++rep) {
        reset();
        auto t0c = std::chrono::steady_clock::now();
        for (int s = 0; s < c.n_steps; ++s) { one_step(); if (s % 10 == 9 && diverged()) { run_diverged = true; break; } }
        sync_all();
        auto t1c = std::chrono::steady_clock::now();
        CUDA_LAUNCH_OK("timed repeat");
        samples.push_back(std::chrono::duration<double>(t1c - t0c).count());
        if (run_diverged) { std::printf("  DIVERGED (|eta| > 1e6 or NaN) - run stopped early\n"); break; }
    }
    const int n_done = static_cast<int>(samples.size());
    const double t_med = n_done ? median_of(samples) : std::nan("");
    std::vector<double> dev; for (double x : samples) dev.push_back(std::fabs(x - t_med));
    const double t_mad = n_done ? median_of(dev) : std::nan("");

    /* gather owned rows into the global state */
    std::vector<double> hu(n3), hv(n3), hb(n3), ht(n3), hs(n3), he(n2);
    for_each_dev([&](Slab& S) {
        const int n2l = nx * S.ny_ext, n3l = n2l * nz;
        std::vector<double> lu(n3l), lv(n3l), lb(n3l), lt(n3l), ls(n3l), le(n2l);
        auto dn = [&](std::vector<double>& h, const double* d) { CUDA_OK(cudaMemcpy(h.data(), d, h.size() * sizeof(double), cudaMemcpyDeviceToHost)); };
        dn(lu, S.u); dn(lv, S.v); dn(lb, S.b); dn(lt, S.t); dn(ls, S.s); dn(le, S.eta);
        for (int k = 0; k < nz; ++k) for (int r = 0; r < S.ny_loc; ++r) {
            size_t src = (static_cast<size_t>(k) * S.ny_ext + halo + r) * nx, dst = (static_cast<size_t>(k) * ny + S.j0 + r) * nx;
            std::memcpy(hu.data() + dst, lu.data() + src, nx * sizeof(double)); std::memcpy(hv.data() + dst, lv.data() + src, nx * sizeof(double));
            std::memcpy(hb.data() + dst, lb.data() + src, nx * sizeof(double)); std::memcpy(ht.data() + dst, lt.data() + src, nx * sizeof(double));
            std::memcpy(hs.data() + dst, ls.data() + src, nx * sizeof(double));
        }
        for (int r = 0; r < S.ny_loc; ++r) std::memcpy(he.data() + static_cast<size_t>(S.j0 + r) * nx, le.data() + static_cast<size_t>(halo + r) * nx, nx * sizeof(double));
    });
    std::printf("  wall median=%12.5E s  mad=%12.5E s  solver_iters/run=0  substeps/run=0\n", t_med, t_mad);

    const std::string bin = c.out_prefix + "_state3d5.bin";
    if (FILE* f = std::fopen(bin.c_str(), "wb")) {
        auto wr = [&](const std::vector<double>& a, int n) { std::fwrite(a.data(), sizeof(double), n, f); };
        wr(he, n2); wr(hu, n3); wr(hv, n3); wr(hb, n3);
        if (ts) { wr(ht, n3); wr(hs, n3); }
        std::fclose(f);
    }
    const std::string mj = c.out_prefix + "_metrics.json";
    if (FILE* f = std::fopen(mj.c_str(), "w")) {
        std::fprintf(f, "{\n  \"backend\": \"cuda3d5_multi\",\n  \"gpu\": \"%s\",\n  \"n_devices\": %d,\n  \"halo\": %d,\n  \"nx\": %d,\n  \"ny\": %d,\n  \"nz\": %d,\n  \"cells\": %d,\n"
                        "  \"n_steps\": %d,\n  \"scheme\": \"%s\",\n  \"solver\": \"%s\",\n  \"eos\": \"%s\",\n  \"advection\": \"%s\",\n  \"tracers\": \"%s\",\n  \"closure\": \"%s\",\n"
                        "  \"dt\": %s,\n  \"device_bytes\": %.0f,\n  \"diverged\": %s,\n  \"wall_s\": %s,\n  \"wall_mad_s\": %s,\n  \"n_repeat\": %d,\n  \"omp_num_threads\": 0\n}\n",
                     dev_names[0].c_str(), ndev, halo, nx, ny, nz, n3, c.n_steps, c.scheme_name.c_str(), c.solver_kind.c_str(), v.eos.c_str(), v.advection.c_str(),
                     v.tracers.c_str(), v.closure.c_str(), jnum(c.dt).c_str(), double(dev_bytes_total), run_diverged ? "true" : "false",
                     jnum(t_med).c_str(), jnum(t_mad).c_str(), n_done);
        std::fclose(f);
    }
    return run_diverged ? 3 : 0;
}
