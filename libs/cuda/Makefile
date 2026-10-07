# Build the native CUDA backend.
# Flags pinned in config/backends.toml (threat T3). cc120 = RTX 5090 (GB202).
NVCC     ?= nvcc
ARCH     ?= sm_120
# No --use_fast_math: fp64 verification requires IEEE semantics (R9).
NVFLAGS  ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
SRCDIR   := src
BUILD    := build
BIN      := $(BUILD)/cfd_exp_cuda
BIN3D    := $(BUILD)/cfd_exp3d_cuda

.PHONY: all all3d clean
all: $(BIN)
all3d: $(BIN3D)

$(BIN): $(SRCDIR)/cfd_exp_cuda.cu $(SRCDIR)/kernels.cuh $(SRCDIR)/config.hpp
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(SRCDIR)/cfd_exp_cuda.cu
	@echo "built $@ with: $(NVCC) $(NVFLAGS)"

$(BIN3D): $(SRCDIR)/cfd_exp3d_cuda.cu $(SRCDIR)/kernels3d.cuh $(SRCDIR)/pcg_device.cuh $(SRCDIR)/config.hpp
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(SRCDIR)/cfd_exp3d_cuda.cu
	@echo "built $@ with: $(NVCC) $(NVFLAGS)"

clean:
	rm -rf $(BUILD)

BINH := $(BUILD)/helmholtz_bench_cuda
.PHONY: helm
helm: $(BINH)
$(BINH): $(SRCDIR)/helmholtz_bench.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(SRCDIR)/helmholtz_bench.cu
	@echo "built $@ with: $(NVCC) $(NVFLAGS)"

BINE := $(BUILD)/eos_bench_cuda
.PHONY: eos
eos: $(BINE)
$(BINE): $(SRCDIR)/eos_bench.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(SRCDIR)/eos_bench.cu
	@echo "built $@ with: $(NVCC) $(NVFLAGS)"

# fp32 solver build (axis G); the problem on disk stays fp64.
BINHSP := $(BUILD)/helmholtz_bench_cuda_sp
.PHONY: helm-sp
helm-sp: $(BINHSP)
$(BINHSP): $(SRCDIR)/helmholtz_bench.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -DHELM_REAL=float -o $@ $(SRCDIR)/helmholtz_bench.cu
	@echo "built $@ (fp32 solver)"

# Multi-GPU communication microbenchmark (needs MPI). MPICXX must be a
# CUDA-capable wrapper; nvcc is invoked through it as the host compiler.
MPICXX ?= mpicxx
BINC := $(BUILD)/comm_bench
.PHONY: comm
comm: $(BINC)
$(BINC): $(SRCDIR)/comm_bench.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -ccbin $(MPICXX) -o $@ $(SRCDIR)/comm_bench.cu
	@echo "built $@"

# spec v0.5 full core (docs/03 S10): same namelist + binaries as cfd_exp3d5.
BIN3D5 := $(BUILD)/cfd_exp3d5_cuda
.PHONY: all3d5
all3d5: $(BIN3D5)
$(BIN3D5): $(SRCDIR)/cfd_exp3d5_cuda.cu $(SRCDIR)/config.hpp
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(SRCDIR)/cfd_exp3d5_cuda.cu
	@echo "built $@ with: $(NVCC) $(NVFLAGS)"

# fp32 working precision (axis G). The variant is GENERATED from the fp64
# source by tools/make_sp_cuda.py; `sp-check` builds it in fp64 mode, which must
# reproduce the original binary's output - that is the transformation's proof.
BIN3D5SP := $(BUILD)/cfd_exp3d5_cuda_sp
BIN3D5DPCHK := $(BUILD)/cfd_exp3d5_cuda_spcheck
.PHONY: all3d5-sp all3d5-spcheck
all3d5-sp: $(SRCDIR)/cfd_exp3d5_cuda_sp.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -DCFD_REAL=float -o $(BIN3D5SP) $(SRCDIR)/cfd_exp3d5_cuda_sp.cu
	@echo "built $(BIN3D5SP) (fp32) with: $(NVCC) $(NVFLAGS) -DCFD_REAL=float"
all3d5-spcheck: $(SRCDIR)/cfd_exp3d5_cuda_sp.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -DCFD_REAL=double -o $(BIN3D5DPCHK) $(SRCDIR)/cfd_exp3d5_cuda_sp.cu
	@echo "built $(BIN3D5DPCHK) (generated file in fp64 mode - must match the original)"

# E17 (docs/37 S6): the cost of a per-step module boundary for the closure's
# real data footprint. No model code - transfers only.
BINMB := $(BUILD)/module_boundary
.PHONY: boundary
boundary: $(BINMB)
$(BINMB): $(SRCDIR)/module_boundary.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $(BINMB) $(SRCDIR)/module_boundary.cu
	@echo "built $(BINMB) with: $(NVCC) $(NVFLAGS)"

# Mixed precision (docs/37 S2-3, expr/E18): fp64 core, fp32 TKE closure. The
# source is generated from the fp64 file by tools/make_mixed_cuda.py, which
# retypes ONLY the body of k_closure; `mixedcheck` builds that generated file
# with the closure back in double, which must reproduce the original bit for bit.
BIN3D5MX := $(BUILD)/cfd_exp3d5_cuda_mixed
BIN3D5MXCHK := $(BUILD)/cfd_exp3d5_cuda_mixedcheck
.PHONY: all3d5-mixed all3d5-mixedcheck
all3d5-mixed: $(SRCDIR)/cfd_exp3d5_cuda_mixed.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -DCLOSURE_T=float -o $(BIN3D5MX) $(SRCDIR)/cfd_exp3d5_cuda_mixed.cu
	@echo "built $(BIN3D5MX) (fp64 core, fp32 closure)"
all3d5-mixedcheck: $(SRCDIR)/cfd_exp3d5_cuda_mixed.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -DCLOSURE_T=double -o $(BIN3D5MXCHK) $(SRCDIR)/cfd_exp3d5_cuda_mixed.cu
	@echo "built $(BIN3D5MXCHK) (generated file, closure in fp64 - must match the original)"

# E19: single-process multi-GPU driver (y-slabs with deep halos, fb scheme only; docs/03 S12, docs/41).
BIN3D5MULTI := $(BUILD)/cfd_exp3d5_cuda_multi
.PHONY: multi
multi: $(BIN3D5MULTI)
$(BIN3D5MULTI): $(SRCDIR)/cfd_exp3d5_cuda_multi.cu $(SRCDIR)/cfd_exp3d5_cuda.cu $(SRCDIR)/config.hpp
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $(SRCDIR)/cfd_exp3d5_cuda_multi.cu
	@echo "built $@ with: $(NVCC) $(NVFLAGS)"

comma := ,
# E19 multi-node: MPI driver (one GPU per rank, halo by MPI_Sendrecv; host-staged or CUDA-aware with --device-mpi).
MPI_CFLAGS ?= $(shell mpicxx --showme:compile 2>/dev/null)
MPI_LDFLAGS ?= $(shell mpicxx --showme:link 2>/dev/null)
BIN3D5MPI := $(BUILD)/cfd_exp3d5_cuda_mpi
.PHONY: mpi
mpi: $(BIN3D5MPI)
$(BIN3D5MPI): $(SRCDIR)/cfd_exp3d5_cuda_mpi.cu $(SRCDIR)/cfd_exp3d5_cuda.cu $(SRCDIR)/config.hpp
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $(MPI_CFLAGS) -o $@ $(SRCDIR)/cfd_exp3d5_cuda_mpi.cu $(subst -Wl$(comma)--enable-new-dtags,,$(subst -Wl$(comma)-rpath -Wl$(comma),-Xlinker -rpath -Xlinker ,$(MPI_LDFLAGS)))
	@echo "built $@ (MPI: $(MPI_CFLAGS))"
