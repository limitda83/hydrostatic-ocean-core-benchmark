/************************************************************************
 *  Module: cfd_exp3d5_cuda_mpi                                         *
 *  Description: MPI driver for the spec v0.5/v0.6 CUDA core (E19,      *
 *               multi-node). Rank r owns y-slab r of the domain with a  *
 *               deep halo of H rows and one GPU; each rank runs the     *
 *               UNMODIFIED single-device step on its extended slab, and *
 *               the halos are exchanged once per step with              *
 *               MPI_Sendrecv, either staged through pinned host memory  *
 *               (default) or passing device pointers to a CUDA-aware    *
 *               MPI (--device-mpi). Slab interiors are bit-identical to *
 *               the single-device run (docs/03 S12). fb scheme only.    *
 *  Pipeline: make mpi ; mpirun -np N -H host1,host2 ...                *
 *            cfd_exp3d5_cuda_mpi <namelist> [--halo H] [--device-mpi]  *
 ************************************************************************/
#include <mpi.h>
#include <unistd.h>
#define CFD_MULTI_NO_MAIN
#include "cfd_exp3d5_cuda.cu"

namespace {

__global__ void k_pack_m(const double* a, double* buf, int nx, int ny_ext, int r0, int H, int nlev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, k = blockIdx.z;
    if (i < nx && r < H && k < nlev) buf[i + nx * (r + H * k)] = a[i + nx * (r0 + r + ny_ext * k)];
}
__global__ void k_unpack_m(double* a, const double* buf, int nx, int ny_ext, int r0, int H, int nlev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, k = blockIdx.z;
    if (i < nx && r < H && k < nlev) a[i + nx * (r0 + r + ny_ext * k)] = buf[i + nx * (r + H * k)];
}

static std::vector<double> slice_rows(const std::vector<double>& g, int nx, int ny, int nlev, const std::vector<int>& rows) {
    std::vector<double> out(static_cast<size_t>(nx) * rows.size() * nlev);
    for (int k = 0; k < nlev; ++k)
        for (size_t r = 0; r < rows.size(); ++r)
            std::memcpy(out.data() + (static_cast<size_t>(k) * rows.size() + r) * nx,
                        g.data() + (static_cast<size_t>(k) * ny + rows[r]) * nx, nx * sizeof(double));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2) { if (rank == 0) std::fprintf(stderr, "usage: cfd_exp3d5_cuda_mpi <namelist> [--halo H] [--device-mpi] [--device D]\n"); MPI_Finalize(); return 1; }
    int halo = 16, dev = 0; bool device_mpi = false;
    for (int a = 2; a < argc; ++a) {
        std::string k = argv[a];
        if (k == "--halo" && a + 1 < argc) halo = std::atoi(argv[++a]);
        else if (k == "--device" && a + 1 < argc) dev = std::atoi(argv[++a]);
        else if (k == "--device-mpi") device_mpi = true;
    }
    const RunConfig c = read_config(argv[1]);
    const V05Config v = read_v05(argv[1]);
    if (c.scheme_name != "fb") { if (rank == 0) std::fprintf(stderr, "FATAL: MPI driver supports scheme='fb' only\n"); MPI_Finalize(); return 2; }
    const int nx = c.nx, ny = c.ny, nz = c.nz, n2 = nx * ny, n3 = n2 * nz;
    const bool ts = v.tracers == "TS";
    if (ny / size < halo) { if (rank == 0) std::fprintf(stderr, "FATAL: ny/size < halo\n"); MPI_Finalize(); return 2; }
    char hostname[256] = {0}; gethostname(hostname, 255);

    /* ---- global domain and initial state (every rank reads the shared files) ---- */
    std::vector<double> io2(n2);
    Domain dom; dom.h.resize(n2); dom.mask.resize(n2);
    { FILE* f = std::fopen(v.domain_file.c_str(), "rb");
      if (!f) { std::fprintf(stderr, "FATAL: cannot read %s\n", v.domain_file.c_str()); MPI_Abort(MPI_COMM_WORLD, 1); }
      std::fread(io2.data(), sizeof(double), n2, f); dom.h.assign(io2.begin(), io2.end());
      std::fread(io2.data(), sizeof(double), n2, f); dom.mask.assign(io2.begin(), io2.end());
      std::fclose(f); }
    dom.build(v, nx, ny, nz);
    std::vector<double> e0(n2), u0(n3), v0(n3), b0(n3), t0(n3, 0.0), s0(n3, 0.0);
    { FILE* f = std::fopen(v.init_file.c_str(), "rb");
      if (!f) { std::fprintf(stderr, "FATAL: cannot read %s\n", v.init_file.c_str()); MPI_Abort(MPI_COMM_WORLD, 1); }
      auto rd2 = [&](std::vector<double>& dst) { std::fread(dst.data(), sizeof(double), n2, f); };
      auto rd3 = [&](std::vector<double>& dst) { std::fread(dst.data(), sizeof(double), n3, f); };
      rd2(e0); rd3(u0); rd3(v0); rd3(b0);
      if (ts) { rd3(t0); rd3(s0); }
      std::fclose(f); }

    /* ---- this rank's slab ---- */
    const int j0 = (ny * rank) / size, ny_loc = (ny * (rank + 1)) / size - j0, ny_ext = ny_loc + 2 * halo;
    std::vector<int> rows(ny_ext);
    for (int r = 0; r < ny_ext; ++r) rows[r] = ((j0 - halo + r) % ny + ny) % ny;
    Domain dl; dl.nx = nx; dl.ny = ny_ext; dl.nz = nz; dl.hmax = dom.hmax;
    for (auto pr : {std::pair<std::vector<double>*, const std::vector<double>*>{&dl.h, &dom.h}, {&dl.mask, &dom.mask}, {&dl.masku, &dom.masku},
                    {&dl.maskv, &dom.maskv}, {&dl.hu, &dom.hu}, {&dl.hv, &dom.hv}, {&dl.hcu, &dom.hcu}, {&dl.hcv, &dom.hcv},
                    {&dl.inv_hcu, &dom.inv_hcu}, {&dl.inv_hcv, &dom.inv_hcv}})
        *pr.first = slice_rows(*pr.second, nx, ny, 1, rows);
    for (auto pr : {std::pair<std::vector<double>*, const std::vector<double>*>{&dl.dz3, &dom.dz3}, {&dl.dz3u, &dom.dz3u}, {&dl.dz3v, &dom.dz3v},
                    {&dl.mask3, &dom.mask3}, {&dl.mask3u, &dom.mask3u}, {&dl.mask3v, &dom.mask3v}, {&dl.inv_dz3, &dom.inv_dz3}, {&dl.zc, &dom.zc}})
        *pr.first = slice_rows(*pr.second, nx, ny, nz, rows);
    RunConfig cl = c; cl.ny = ny_ext;
    CUDA_OK(cudaSetDevice(dev));
    cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    const int n2l = nx * ny_ext, n3l = n2l * nz, n3il = n2l * (nz + 1);
    double *du = dev_zero(n3l), *dv = dev_zero(n3l), *db = dev_zero(n3l), *dt_ = dev_zero(n3l), *ds = dev_zero(n3l), *deta = dev_zero(n2l);
    double* part = dev_zero(1024); std::vector<double> h_part(1024);
    size_t fb = 0, fa = 0, tot = 0; CUDA_OK(cudaMemGetInfo(&fb, &tot));
    Stepper5* st = new Stepper5(cl, v, std::move(dl));
    st->d.dx = c.lx / c.nx; st->d.dy = c.ly / c.ny; st->build_coefficients(false);
    CUDA_OK(cudaMemGetInfo(&fa, &tot)); const double dev_bytes = double(fb - fa);
    if (st->is_split || st->is_theta) { MPI_Abort(MPI_COMM_WORLD, 2); }

    /* halo buffers: device send/recv, and pinned host staging unless the MPI is used with device pointers */
    std::vector<std::pair<double*, int>> arrs = {{du, nz}, {dv, nz}, {db, nz}};
    if (ts) { arrs.push_back({dt_, nz}); arrs.push_back({ds, nz}); }
    arrs.push_back({deta, 1});
    if (st->is_tke) { arrs.push_back({st->d.tke, nz + 1}); arrs.push_back({st->d.nu3, nz + 1}); arrs.push_back({st->d.kap3, nz + 1}); }
    size_t buf_len = 0; for (auto& a : arrs) buf_len += static_cast<size_t>(nx) * halo * a.second;
    double *send_lo = dev_zero(buf_len), *send_hi = dev_zero(buf_len), *recv_lo = dev_zero(buf_len), *recv_hi = dev_zero(buf_len);
    double *h_send_lo = nullptr, *h_send_hi = nullptr, *h_recv_lo = nullptr, *h_recv_hi = nullptr;
    if (!device_mpi) for (double** p : {&h_send_lo, &h_send_hi, &h_recv_lo, &h_recv_hi}) CUDA_OK(cudaMallocHost(p, buf_len * sizeof(double)));
    const int left = (rank - 1 + size) % size, right = (rank + 1) % size;
    const dim3 pb(256, 1, 1);
    auto exchange = [&]() {
        size_t off = 0;
        for (auto& a : arrs) {
            dim3 pg((nx + 255) / 256, halo, a.second);
            k_pack_m<<<pg, pb>>>(a.first, send_lo + off, nx, ny_ext, halo, halo, a.second);
            k_pack_m<<<pg, pb>>>(a.first, send_hi + off, nx, ny_ext, ny_loc, halo, a.second);
            off += static_cast<size_t>(nx) * halo * a.second;
        }
        CUDA_OK(cudaDeviceSynchronize());
        const int n = static_cast<int>(buf_len);
        if (device_mpi) {
            MPI_Sendrecv(send_hi, n, MPI_DOUBLE, right, 1, recv_lo, n, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_lo, n, MPI_DOUBLE, left, 2, recv_hi, n, MPI_DOUBLE, right, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            CUDA_OK(cudaMemcpy(h_send_lo, send_lo, buf_len * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(h_send_hi, send_hi, buf_len * sizeof(double), cudaMemcpyDeviceToHost));
            MPI_Sendrecv(h_send_hi, n, MPI_DOUBLE, right, 1, h_recv_lo, n, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(h_send_lo, n, MPI_DOUBLE, left, 2, h_recv_hi, n, MPI_DOUBLE, right, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_OK(cudaMemcpy(recv_lo, h_recv_lo, buf_len * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(recv_hi, h_recv_hi, buf_len * sizeof(double), cudaMemcpyHostToDevice));
        }
        off = 0;
        for (auto& a : arrs) {
            dim3 pg((nx + 255) / 256, halo, a.second);
            k_unpack_m<<<pg, pb>>>(a.first, recv_lo + off, nx, ny_ext, 0, halo, a.second);
            k_unpack_m<<<pg, pb>>>(a.first, recv_hi + off, nx, ny_ext, halo + ny_loc, halo, a.second);
            off += static_cast<size_t>(nx) * halo * a.second;
        }
        CUDA_OK(cudaDeviceSynchronize());
    };
    auto reset = [&]() {
        auto up = [&](double* d, const std::vector<double>& h) { CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice)); };
        up(du, slice_rows(u0, nx, ny, nz, rows)); up(dv, slice_rows(v0, nx, ny, nz, rows)); up(db, slice_rows(b0, nx, ny, nz, rows));
        up(dt_, slice_rows(t0, nx, ny, nz, rows)); up(ds, slice_rows(s0, nx, ny, nz, rows)); up(deta, slice_rows(e0, nx, ny, 1, rows));
        if (st->is_tke) {
            std::vector<double> nu_i(n3il, c.nu), kap_i(n3il, c.kappa), tke0(n3il, v.e_min);
            up(st->d.nu3, nu_i); up(st->d.kap3, kap_i); up(st->d.tke, tke0);
            st->build_coefficients(false);
        }
        CUDA_OK(cudaDeviceSynchronize());
    };
    auto diverged = [&]() -> bool {
        int blocks = std::min(1024, (n2l + 255) / 256);
        k1_absmax<<<blocks, 256, 256 * sizeof(double)>>>(n2l, deta, part);
        CUDA_OK(cudaMemcpy(h_part.data(), part, blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double m = 0.0; for (int i = 0; i < blocks; ++i) if (!(h_part[i] <= m)) m = h_part[i];
        double g = 0.0; MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return !(g < 1.0e6);
    };
    auto one_step = [&]() { st->step(du, dv, db, deta, dt_, ds); CUDA_OK(cudaDeviceSynchronize()); exchange(); };

    if (rank == 0) std::printf("cuda3d5-mpi: case=%s nx=%d ny=%d nz=%d scheme=%s ranks=%d halo=%d device_mpi=%d gpu=%s steps=%d\n",
                               c.case_name.c_str(), nx, ny, nz, c.scheme_name.c_str(), size, halo, (int)device_mpi, prop.name, c.n_steps);
    bool run_diverged = false;
    for (int rep = 0; rep < c.n_warmup && !run_diverged; ++rep) {
        reset();
        for (int s = 0; s < c.n_steps; ++s) { one_step(); if (s == 0) CUDA_LAUNCH_OK("first step"); if (s % 10 == 9 && diverged()) { run_diverged = true; break; } }
    }
    std::vector<double> samples;
    for (int rep = 0; rep < c.n_repeat && !run_diverged; ++rep) {
        reset(); MPI_Barrier(MPI_COMM_WORLD);
        auto t0c = std::chrono::steady_clock::now();
        for (int s = 0; s < c.n_steps; ++s) { one_step(); if (s % 10 == 9 && diverged()) { run_diverged = true; break; } }
        CUDA_OK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
        auto t1c = std::chrono::steady_clock::now();
        double w = std::chrono::duration<double>(t1c - t0c).count(), wmax = 0.0;
        MPI_Allreduce(&w, &wmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        samples.push_back(wmax);
        if (run_diverged) { if (rank == 0) std::printf("  DIVERGED - run stopped early\n"); break; }
    }
    const int n_done = static_cast<int>(samples.size());
    const double t_med = n_done ? median_of(samples) : std::nan("");
    std::vector<double> devv; for (double x : samples) devv.push_back(std::fabs(x - t_med));
    const double t_mad = n_done ? median_of(devv) : std::nan("");

    /* gather owned rows to rank 0: each rank packs [k][r][i] of its interior */
    auto gather = [&](const double* d, int nlev, std::vector<double>& global) {
        std::vector<double> loc(static_cast<size_t>(n2l) * nlev), pk(static_cast<size_t>(nx) * ny_loc * nlev);
        CUDA_OK(cudaMemcpy(loc.data(), d, loc.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int k = 0; k < nlev; ++k) for (int r = 0; r < ny_loc; ++r)
            std::memcpy(pk.data() + (static_cast<size_t>(k) * ny_loc + r) * nx, loc.data() + (static_cast<size_t>(k) * ny_ext + halo + r) * nx, nx * sizeof(double));
        std::vector<int> counts(size), displs(size);
        for (int q = 0; q < size; ++q) { int jq0 = (ny * q) / size, nyq = (ny * (q + 1)) / size - jq0; counts[q] = nx * nyq * nlev; displs[q] = q ? displs[q - 1] + counts[q - 1] : 0; }
        std::vector<double> all(rank == 0 ? static_cast<size_t>(nx) * ny * nlev : 0);
        MPI_Gatherv(pk.data(), static_cast<int>(pk.size()), MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            global.assign(static_cast<size_t>(nx) * ny * nlev, 0.0);
            for (int q = 0; q < size; ++q) {
                int jq0 = (ny * q) / size, nyq = (ny * (q + 1)) / size - jq0;
                for (int k = 0; k < nlev; ++k) for (int r = 0; r < nyq; ++r)
                    std::memcpy(global.data() + (static_cast<size_t>(k) * ny + jq0 + r) * nx, all.data() + displs[q] + (static_cast<size_t>(k) * nyq + r) * nx, nx * sizeof(double));
            }
        }
    };
    std::vector<double> hu, hv, hb, ht, hs, he;
    gather(du, nz, hu); gather(dv, nz, hv); gather(db, nz, hb); gather(dt_, nz, ht); gather(ds, nz, hs); gather(deta, 1, he);
    if (rank == 0) {
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
            std::fprintf(f, "{\n  \"backend\": \"cuda3d5_mpi\",\n  \"gpu\": \"%s\",\n  \"n_devices\": %d,\n  \"halo\": %d,\n  \"device_mpi\": %s,\n  \"host\": \"%s\",\n  \"nx\": %d,\n  \"ny\": %d,\n  \"nz\": %d,\n  \"cells\": %d,\n"
                            "  \"n_steps\": %d,\n  \"scheme\": \"%s\",\n  \"solver\": \"%s\",\n  \"eos\": \"%s\",\n  \"advection\": \"%s\",\n  \"tracers\": \"%s\",\n  \"closure\": \"%s\",\n"
                            "  \"dt\": %s,\n  \"device_bytes\": %.0f,\n  \"diverged\": %s,\n  \"wall_s\": %s,\n  \"wall_mad_s\": %s,\n  \"n_repeat\": %d,\n  \"omp_num_threads\": 0\n}\n",
                         prop.name, size, halo, device_mpi ? "true" : "false", hostname, nx, ny, nz, n3, c.n_steps, c.scheme_name.c_str(), c.solver_kind.c_str(), v.eos.c_str(),
                         v.advection.c_str(), v.tracers.c_str(), v.closure.c_str(), jnum(c.dt).c_str(), dev_bytes, run_diverged ? "true" : "false",
                         jnum(t_med).c_str(), jnum(t_mad).c_str(), n_done);
            std::fclose(f);
        }
    }
    MPI_Finalize();
    return run_diverged ? 3 : 0;
}
