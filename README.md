# hydrostatic-ocean-core-benchmark

Code and measurement data for

> Park, Y. and Kim, C. (2026). *Performance of a hydrostatic ocean core across computing devices and programming frameworks under verified double-precision agreement.* Submitted to **Computers & Fluids**.

One hydrostatic Boussinesq primitive-equation ocean core (partial-cell bathymetry, implicit vertical mixing, a one-equation TKE closure, third-order TVD advection) is implemented **five times from one discretization document** and timed on three kinds of hardware. Every timed binary must first reproduce the NumPy reference to a relative L2 error below 1e-12 per step, with elliptic-solver iteration counts agreeing to the digit, on the node where it is timed. Only then is its wall time reported.

| Implementation | Where | Build |
|---|---|---|
| NumPy fp64 reference (the oracle) | `libs/core/`, `main.py` | `pip install -r requirements.txt` |
| Fortran serial / OpenMP | `libs/fortran/src/` | `make -C libs/fortran all3d5 serial3d5` |
| OpenACC (nvfortran) | `libs/fortran/src/` | `make -C libs/fortran acc3d5` |
| CUDA (single card, 2–4 cards in one node, MPI across nodes) | `libs/cuda/src/` | `make -C libs/cuda all3d5 multi mpi ARCH=sm_90` |
| JAX | `libs/jax/` | `python3 libs/jax/cfd_exp3d5_jax.py <case>.nml` |

All backends read the **same** TOML configuration (`config/*.toml`); the Fortran/OpenACC/CUDA/JAX drivers read a namelist and binary bundle generated from it by `tools/toml2nml.py`, so no two backends can solve different problems by accident. The discretization they implement is `docs/03_discretization_spec.md` (the single source of truth; multi-GPU slab decomposition in its §12).

Archived release: **Zenodo DOI: (assigned at the first GitHub release; see the paper's Code/Data availability statements)**.

## Layout

```
main.py                 reference driver: run | verify | bench
config/                 all parameters (grid, cases, schemes, backends, settings) — nothing is hard-coded
libs/core,io,utils      NumPy reference implementation (fp64 oracle)
libs/fortran, cuda, jax the four ported backends
tools/                  verification gates, benchmark sweeps, collectors, audits, promotion
docs/                   design, discretization specification (03), result documents (11–42), negative results (90)
expr/                   experiment ledger: one directory per experiment with README (question, design,
                        commands, node, date, status) and data/ (the preserved raw CSV/JSON/logs)
paper/cases/            the promoted data behind every figure and table of the paper (see below) — CC BY 4.0
paper/figures/          make_figures.py regenerates the six figures from paper/cases/
paper/TARGET.md         the claim map: which case supports which claim
usage.md                full usage notes (Korean), including the three bench nodes and their roles
```

Most internal documents (`docs/`, `expr/*/README.md`, `usage.md`) are written in Korean; the code, comments, configuration keys and data files are in English.

## Reproducing the paper's figures and tables

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python >= 3.11
pip install -r requirements.txt
python3 tools/promote_cases.py --check               # paper/cases/ is self-consistent (sha256 of every file)
python3 paper/figures/make_figures.py                # Fig. 1–6 -> paper/figures/*.pdf|png
python3 tools/expr_audit.py                          # every ledger claim is checked against data/
python3 tools/audit_data.py                          # every preserved CSV passes format/MAD checks
```

The mapping from the paper to the promoted case, the experiment and the result document:

| Paper element | Case (`paper/cases/`) | Experiment (`expr/`) | Result document |
|---|---|---|---|
| Table 1 verification gates; Appendix A | `verification` | E05, E07, E14, E16 | `docs/40_verification_summary.md` |
| Table 2 nodes and toolchains | — | — | `config/backends.toml`, `usage.md` §12 and `RULES.md` |
| Table 3 speed-up map; Fig. 1 hardware ladder; Table 8; Tables B.1–B.3 | `hardware_ladder` | E12, E13, E16 | `docs/32_hardware_ladder.md`, `docs/39_speedup_generation_framework.md` |
| Fig. 2 accuracy–wall-time plane; Table 4 | `pareto_plane` | E12, E13 | `docs/30_tier2_pareto.md` |
| Table 5 Helmholtz solver kernels; Table 6 equation of state; Table B.4 | `kernel_matrix` | E03, E08 | `docs/25_kernel_matrix.md`, `docs/12_gpu_microbench.md` |
| Table 7 two and four GPUs (one node and two nodes) | `multi_gpu` | E19 | `docs/42_multigpu_scaling.md`, `docs/41_multinode_design.md` |
| Fig. 3 precision (fp32 and mixed) | `precision_axis` | E15, E18 | `docs/34_precision_axis.md` |
| Fig. 4 cost of the physics | `physics_cost` (+ `hardware_ladder`) | E14 | `docs/31_physics_cost.md` |
| Fig. 5 mixing-length formulation | `mixing_length_axis` | E16 | `docs/36_mixing_length_axis.md` |
| Fig. 6 module boundary (host round trip) | `module_boundary` | E17 | `docs/38_module_boundary.md` |
| Table C.1 communication constants | `comm_bound` | E11 | `docs/28_multigpu_communication.md` |

Each `paper/cases/<case>/README.md` names the claim it supports, the source experiment, the reproduction commands and the sha256 of every file. Files carrying a quarantine tag in the ledger (`SUPERSEDED`, `_CONTAMINATED`, `_DISCARDED`, `_INVALID`, `_RESULT`, `_UNGATED`, `_IRREPRODUCIBLE`, `_MISMATCH`) are kept in `expr/` for the record but never enter `paper/cases/` or the paper.

## Running the core yourself

```bash
python3 main.py verify --study space --case igw                 # convergence order of the reference
python3 tools/verify_v05.py ; python3 tools/verify_v06.py       # V5-1…V5-6, V6-1…V6-4 physics tests
python3 tools/mutation_check.py                                 # every test must be able to fail (13/13)
bash tools/gate3d5.sh                                           # R2 gate: every built backend vs the reference
python3 tools/toml2nml.py --v05 --case lock_exchange --nx 400 --nz 30 --cfl 0.5 --steps 50 \
        --set scheme.name=fb --prefix output/le400              # a timed bundle
libs/cuda/build/cfd_exp3d5_cuda output/le400.nml                # one backend on it
CASES=lock_exchange_v06 bash tools/tier2_sweep.sh               # the ladder sweep used for Table 3
```

Timing protocol (enforced by the scripts and the audits): warm-up discarded, at least five repeats, median and MAD reported, device synchronised before the clock stops, a verification-gate record for the binary on the node it is timed on, one problem per table, and both baselines named whenever a speed-up is quoted.

## Bench nodes

Three nodes with separate roles: a workstation with 4× RTX 5090 (all CPU-vs-GPU comparisons on one node), a PBS cluster of AMD EPYC 9655 nodes (CPU scalability, exclusive nodes), and a KT Cloud container with H100 80 GB (GPU only; its shared host CPU is never timed). Their addresses have been redacted from the published copies of `config/backends.toml`, `usage.md`, `docs/01` and the run manifests (`<gpgpu-node>`, `<geo85-node>`); the manifests' embedded configuration hashes were computed before redaction.

## Licence and citation

Code: BSD 3-Clause (`LICENSE`). Data in `paper/cases/`: CC BY 4.0 (`paper/cases/LICENSE`). Please cite the paper and the Zenodo release (`CITATION.cff`).
