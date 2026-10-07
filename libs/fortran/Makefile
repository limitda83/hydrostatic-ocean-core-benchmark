# Build the Fortran backends.
#
# Flags are pinned in config/backends.toml (threat T3: asymmetric optimisation
# levels invalidate cross-backend timing). Keep the two in sync.
#
#   make                      gfortran, OpenMP CPU          -> build/cfd_exp
#   make FC=nvfortran         nvfortran, OpenMP CPU         -> build/cfd_exp
#   make acc                  nvfortran, OpenACC GPU        -> build/cfd_exp_acc
#   make acc MEM=separate     OpenACC with explicit data movement
#   make serial               no OpenMP (best for small grids, docs/11 S2)
#   make debug                bounds checking
#
# make predefines FC=f77, so ?= would not take effect; only override the
# built-in default, while still honouring an FC given on the command line.
ifeq ($(origin FC),default)
FC      := gfortran
endif

SRCDIR  := src
BUILD   := build
# Shared modules, then the 2D or 3D program on top.
COMMON  := $(SRCDIR)/mod_kinds.f90 \
           $(SRCDIR)/mod_config.f90 \
           $(SRCDIR)/mod_grid.f90 \
           $(SRCDIR)/mod_operators.f90 \
           $(SRCDIR)/mod_solvers.f90 \
           $(SRCDIR)/mod_cases.f90 \
           $(SRCDIR)/mod_schemes.f90 \
           $(SRCDIR)/mod_diag.f90
SOURCES := $(COMMON) $(SRCDIR)/cfd_exp.f90
SOURCES3D := $(COMMON) \
           $(SRCDIR)/mod_vertical.f90 \
           $(SRCDIR)/mod_cases3d.f90 \
           $(SRCDIR)/mod_model3d.f90 \
           $(SRCDIR)/cfd_exp3d.f90

# Compiler-specific flag sets. nvfortran spells OpenMP -mp and has no
# -std=f2008 / -ffree-line-length-none.
FCNAME  := $(notdir $(FC))
ifeq ($(findstring nvfortran,$(FCNAME)),nvfortran)
  OPT   ?= -O3 -tp=native
  OMP   ?= -mp
  STD   ?=
  MODOUT ?= -module          # nvfortran spells the module output dir -module
  DBG   ?= -O0 -g -Mbounds -Mchkptr -traceback
else
  OPT   ?= -O3 -march=native
  OMP   ?= -fopenmp
  STD   ?= -std=f2008 -ffree-line-length-none
  MODOUT ?= -J
  DBG   ?= -O0 -g -fcheck=all -fbacktrace -Wall -Wextra
endif
FFLAGS  ?= $(OPT) $(OMP) $(STD)

# OpenACC (axis D level 'openacc'). cc120 = RTX 5090 / Blackwell GB202.
# MEM=managed uses unified memory so no data clauses are needed - the minimal
# porting-effort variant reported in RQ3. MEM=separate forces explicit data
# management through the !$acc data regions in the source.
ACCFC   ?= nvfortran
CC_ARCH ?= cc120
MEM     ?= managed
ACCFLAGS ?= -O3 -tp=native -acc=gpu -gpu=$(CC_ARCH),mem:$(MEM) -Minfo=accel
ACCMODOUT ?= -module

.PHONY: all acc serial debug clean all3d acc3d serial3d
all: $(BUILD)/cfd_exp
all3d: $(BUILD)/cfd_exp3d

$(BUILD)/cfd_exp: $(SOURCES)
	@mkdir -p $(BUILD)
	$(FC) $(FFLAGS) $(MODOUT) $(BUILD) -o $@ $(SOURCES)
	@echo "built $@ with: $(FC) $(FFLAGS)"

$(BUILD)/cfd_exp3d: $(SOURCES3D)
	@mkdir -p $(BUILD)/d3
	$(FC) $(FFLAGS) $(MODOUT) $(BUILD)/d3 -o $@ $(SOURCES3D)
	@echo "built $@ with: $(FC) $(FFLAGS)"

acc3d: $(SOURCES3D)
	@mkdir -p $(BUILD)/acc3d
	$(ACCFC) $(ACCFLAGS) $(ACCMODOUT) $(BUILD)/acc3d -o $(BUILD)/cfd_exp3d_acc $(SOURCES3D)
	@echo "built $(BUILD)/cfd_exp3d_acc with: $(ACCFC) $(ACCFLAGS)"

acc: $(SOURCES)
	@mkdir -p $(BUILD)/acc
	$(ACCFC) $(ACCFLAGS) $(ACCMODOUT) $(BUILD)/acc -o $(BUILD)/cfd_exp_acc $(SOURCES)
	@echo "built $(BUILD)/cfd_exp_acc with: $(ACCFC) $(ACCFLAGS)"

serial: $(SOURCES)
	@mkdir -p $(BUILD)
	$(FC) $(OPT) $(STD) $(MODOUT) $(BUILD) -o $(BUILD)/cfd_exp_serial $(SOURCES)
	@echo "built $(BUILD)/cfd_exp_serial (no OpenMP)"

serial3d: $(SOURCES3D)
	@mkdir -p $(BUILD)/s3d
	$(FC) $(OPT) $(STD) $(MODOUT) $(BUILD)/s3d -o $(BUILD)/cfd_exp3d_serial $(SOURCES3D)
	@echo "built $(BUILD)/cfd_exp3d_serial (no OpenMP)"

debug: $(SOURCES)
	@mkdir -p $(BUILD)
	$(FC) $(DBG) $(OMP) $(STD) $(MODOUT) $(BUILD) -o $(BUILD)/cfd_exp_debug $(SOURCES)
	@echo "built $(BUILD)/cfd_exp_debug (bounds checking on)"

clean:
	rm -rf $(BUILD)

# ------------------------------------------------ variable-coefficient
# Helmholtz benchmark (spec S10.5). Same source for CPU and OpenACC.
HSRC := $(SRCDIR)/helmholtz_bench.f90
# The benchmark carries its own working-precision switch, so it must be
# preprocessed in every build, not only the fp32 one.
PPFLAG := $(if $(findstring nvfortran,$(FCNAME)),-Mpreprocess,-cpp)

.PHONY: helm helm-serial helm-acc
helm: $(HSRC)
	@mkdir -p $(BUILD)/helm
	$(FC) $(FFLAGS) $(PPFLAG) $(MODOUT) $(BUILD)/helm -o $(BUILD)/helmholtz_bench $(HSRC)
	@echo "built $(BUILD)/helmholtz_bench with: $(FC) $(FFLAGS)"

helm-serial: $(HSRC)
	@mkdir -p $(BUILD)/helms
	$(FC) $(OPT) $(STD) $(PPFLAG) $(MODOUT) $(BUILD)/helms -o $(BUILD)/helmholtz_bench_serial $(HSRC)
	@echo "built $(BUILD)/helmholtz_bench_serial (no OpenMP)"

helm-acc: $(HSRC)
	@mkdir -p $(BUILD)/helma
	$(ACCFC) $(ACCFLAGS) -Mpreprocess $(ACCMODOUT) $(BUILD)/helma -o $(BUILD)/helmholtz_bench_acc $(HSRC)
	@echo "built $(BUILD)/helmholtz_bench_acc with: $(ACCFC) $(ACCFLAGS)"

ESRC := $(SRCDIR)/mod_kinds.f90 $(SRCDIR)/eos_bench.f90
.PHONY: eos eos-acc
eos: $(ESRC)
	@mkdir -p $(BUILD)/eos
	$(FC) $(FFLAGS) $(PPFLAG) $(MODOUT) $(BUILD)/eos -o $(BUILD)/eos_bench $(ESRC)
	@echo "built $(BUILD)/eos_bench with: $(FC) $(FFLAGS)"

eos-acc: $(ESRC)
	@mkdir -p $(BUILD)/eosa
	$(ACCFC) $(ACCFLAGS) -Mpreprocess $(ACCMODOUT) $(BUILD)/eosa -o $(BUILD)/eos_bench_acc $(ESRC)
	@echo "built $(BUILD)/eos_bench_acc with: $(ACCFC) $(ACCFLAGS)"

# fp32 build of the Helmholtz benchmark (axis G). The problem on disk stays
# fp64 and only the solver arithmetic changes, so the error against the fp64
# reference is attributable to the precision alone.
.PHONY: helm-sp helm-acc-sp
helm-sp: $(HSRC)
	@mkdir -p $(BUILD)/helmsp
	$(FC) $(FFLAGS) $(PPFLAG) -DSINGLE_PRECISION $(MODOUT) $(BUILD)/helmsp \
	      -o $(BUILD)/helmholtz_bench_sp $(HSRC)
	@echo "built $(BUILD)/helmholtz_bench_sp (fp32 solver)"

helm-acc-sp: $(HSRC)
	@mkdir -p $(BUILD)/helmasp
	$(ACCFC) $(ACCFLAGS) -Mpreprocess -DSINGLE_PRECISION $(ACCMODOUT) $(BUILD)/helmasp \
	      -o $(BUILD)/helmholtz_bench_acc_sp $(HSRC)
	@echo "built $(BUILD)/helmholtz_bench_acc_sp (fp32 OpenACC)"

# ------------------------------------------------ spec v0.5 full core
# Same source, three builds: gfortran/nvfortran OpenMP, serial, OpenACC.
SOURCES3D5 := $(COMMON) \
           $(SRCDIR)/mod_vertical.f90 \
           $(SRCDIR)/mod_helm_var.f90 \
           $(SRCDIR)/mod_domain5.f90 \
           $(SRCDIR)/mod_model3d_v05.f90 \
           $(SRCDIR)/cfd_exp3d5.f90

.PHONY: all3d5 serial3d5 acc3d5 all3d5-sp serial3d5-sp acc3d5-sp
all3d5: $(SOURCES3D5)
	@mkdir -p $(BUILD)/d35
	$(FC) $(FFLAGS) $(PPFLAG) $(MODOUT) $(BUILD)/d35 -o $(BUILD)/cfd_exp3d5 $(SOURCES3D5)
	@echo "built $(BUILD)/cfd_exp3d5 with: $(FC) $(FFLAGS)"

# fp32 working precision (axis G). The on-disk bundle stays fp64 in both
# builds, so an fp32 run is comparable with the fp64 reference (R9).
all3d5-sp: $(SOURCES3D5)
	@mkdir -p $(BUILD)/d35sp
	$(FC) $(FFLAGS) $(PPFLAG) -DWP_SP $(MODOUT) $(BUILD)/d35sp -o $(BUILD)/cfd_exp3d5_sp $(SOURCES3D5)
	@echo "built $(BUILD)/cfd_exp3d5_sp (fp32 working precision)"

serial3d5-sp: $(SOURCES3D5)
	@mkdir -p $(BUILD)/s35sp
	$(FC) $(OPT) $(STD) $(PPFLAG) -DWP_SP $(MODOUT) $(BUILD)/s35sp -o $(BUILD)/cfd_exp3d5_serial_sp $(SOURCES3D5)
	@echo "built $(BUILD)/cfd_exp3d5_serial_sp (fp32, no OpenMP)"

acc3d5-sp: $(SOURCES3D5)
	@mkdir -p $(BUILD)/acc35sp
	$(ACCFC) $(ACCFLAGS) -Mpreprocess -DWP_SP $(ACCMODOUT) $(BUILD)/acc35sp -o $(BUILD)/cfd_exp3d5_acc_sp $(SOURCES3D5)
	@echo "built $(BUILD)/cfd_exp3d5_acc_sp (fp32) with: $(ACCFC) $(ACCFLAGS)"

serial3d5: $(SOURCES3D5)
	@mkdir -p $(BUILD)/s35
	$(FC) $(OPT) $(STD) $(PPFLAG) $(MODOUT) $(BUILD)/s35 -o $(BUILD)/cfd_exp3d5_serial $(SOURCES3D5)
	@echo "built $(BUILD)/cfd_exp3d5_serial (no OpenMP)"

acc3d5: $(SOURCES3D5)
	@mkdir -p $(BUILD)/acc35
	$(ACCFC) $(ACCFLAGS) -Mpreprocess $(ACCMODOUT) $(BUILD)/acc35 -o $(BUILD)/cfd_exp3d5_acc $(SOURCES3D5)
	@echo "built $(BUILD)/cfd_exp3d5_acc with: $(ACCFC) $(ACCFLAGS)"
