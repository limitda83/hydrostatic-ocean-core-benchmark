#!/usr/bin/env bash
#########################################################################
#  Module: run_tier2_gpgpu.sh                                           #
#  Description: The gpgpu (RTX 5090) share of docs/04 S6: piece S       #
#               (scheme x solver x CFL x case at 400^2 x 30, GPU         #
#               backends, error against the converged reference) and    #
#               then piece H (fixed 50 steps, CPU and GPU backends over  #
#               the grid ladder). Serial and 1-thread runs stop at 400^2 #
#               where they already take minutes per configuration.      #
#  Pipeline: nohup tools/run_tier2_gpgpu.sh -> output/tier2_*/results.csv#
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" PY=./.venv/bin/python BENCH_HOST=gpgpu
export NVHPC_BIN="${NVHPC_BIN:-$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin}"
# BACKENDS_S / BACKENDS_H select the backends of the two pieces (default: the
# piece S runs CUDA only - the error is backend-independent (R2) and the other
# backends' time to solution follows from their cost per step in piece H).
BACKENDS_S="${BACKENDS_S:-cuda}"; BACKENDS_H="${BACKENDS_H:-cpu gpu}"

# R4/R4-2: no timing before the R2 gate ON THIS NODE for the binaries about to be
# timed. This runner had no gate at all; its sibling on ktcloud had the same hole
# and that is how an ungated OpenACC binary produced a whole column of docs/32
# (docs/90 N42). Cheap to run, and it fails loudly instead of silently.
gate_binaries () {
  local out=""
  for b in $1; do
    case "$b" in
      cpu) out="$out libs/fortran/build/cfd_exp3d5_serial libs/fortran/build/cfd_exp3d5" ;;
      gpu) out="$out libs/cuda/build/cfd_exp3d5_cuda libs/fortran/build/cfd_exp3d5_acc" ;;
      cuda) out="$out libs/cuda/build/cfd_exp3d5_cuda" ;;
      openacc) out="$out libs/fortran/build/cfd_exp3d5_acc" ;;
      jax) out="$out libs/jax/build/cfd_exp3d5_jax" ;;
      fortran_omp) out="$out libs/fortran/build/cfd_exp3d5" ;;
      fortran_serial) out="$out libs/fortran/build/cfd_exp3d5_serial" ;;
    esac
  done
  echo "$out"
}
BINS=$(gate_binaries "$BACKENDS_S $BACKENDS_H" | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo "=== R2 gate on $(hostname) for [$BINS]  $(date -u +%FT%TZ)"
gate=$(BINARIES="$BINS" bash tools/gate3d5.sh 2>&1); echo "$gate" | tail -4
echo "$gate" | grep -q " 0 fail" || {
  echo "FATAL: R2 gate failed on this node - no timings are taken (R4)." >&2
  exit 1; }

# SKIP_S=1 runs piece H only: the error of a configuration is backend-independent
# (R2), so the Python-GPU pass needs cost per step, not a second Pareto plane.
if [ "${SKIP_S:-0}" != "1" ]; then
  echo "=== piece S: 400^2 x 30, backends [$BACKENDS_S], full horizon  $(date -u +%FT%TZ)"
  NX=400 NZ=30 BACKENDS="$BACKENDS_S" REPEAT=5 CASES="${CASES_S:-baroclinic_igw basin_seiche lock_exchange}" bash tools/tier2_sweep.sh
fi
echo "=== piece H: 50 steps, backends [$BACKENDS_H], grid ladder  $(date -u +%FT%TZ)"
for nx in 100 200 400 1000 2000; do
  if [ "$nx" -le 400 ]; then t="1 16"; ser=1; else t="16"; ser=0; fi
  NX=$nx NZ=30 STEPS=50 BACKENDS="$BACKENDS_H" THREADS="$t" CPU_SERIAL=$ser REPEAT=5 \
    CASES="${CASES_H:-baroclinic_igw}" SCHEMES="fb theta0.5 split" SOLVERS="multigrid pcg_jacobi" \
    CFLS=4 CFLS_FB=0.5 bash tools/tier2_sweep.sh
done
echo "=== done $(date -u +%FT%TZ)"
