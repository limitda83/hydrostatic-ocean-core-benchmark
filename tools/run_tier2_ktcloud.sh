#!/usr/bin/env bash
#########################################################################
#  Module: run_tier2_ktcloud.sh                                         #
#  Description: The ktcloud (H100) share of docs/04 S6: piece H only,   #
#               GPU backends (R7-1: this node's CPU is shared and never  #
#               timed), fixed 50 steps over the grid ladder up to        #
#               2000^2 x 30 - the 80 GB card is the only one that holds  #
#               it. Piece S is not repeated here: accuracy is backend-   #
#               independent (R2), so only cost per step is compared.     #
#  Pipeline: nohup tools/run_tier2_ktcloud.sh -> output/tier2_*/results.csv#
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
export PY=python3 BENCH_HOST=ktcloud XLA_PYTHON_CLIENT_PREALLOCATE=false
export NVHPC_BIN="${NVHPC_BIN:-/home/work/cello/opt/nvhpc/Linux_x86_64/25.9/compilers/bin}"
BACKENDS_H="${BACKENDS_H:-gpu}"      # BACKENDS_H=jax for the Python-GPU pass

# R4/R4-2: no timing before the R2 gate ON THIS NODE, for the binaries that are
# about to be timed. This script used to go straight to the sweep, which is how
# the H100 OpenACC column of docs/32 came to rest on a binary that had never
# passed a gate here - and could not even read the current namelist by the time
# anyone checked (docs/90 N42). The JAX pass always gated; this path did not.
gate_binaries () {
  case "$1" in
    jax) echo "libs/jax/build/cfd_exp3d5_jax" ;;
    gpu) echo "libs/cuda/build/cfd_exp3d5_cuda libs/fortran/build/cfd_exp3d5_acc" ;;
    *)   for b in $1; do
           case "$b" in
             cuda) echo -n "libs/cuda/build/cfd_exp3d5_cuda " ;;
             openacc) echo -n "libs/fortran/build/cfd_exp3d5_acc " ;;
             jax) echo -n "libs/jax/build/cfd_exp3d5_jax " ;;
           esac
         done; echo ;;
  esac
}
BINS=$(gate_binaries "$BACKENDS_H")
echo "=== R2 gate on $(hostname) for [$BINS]  $(date -u +%FT%TZ)"
gate=$(BINARIES="$BINS" bash tools/gate3d5.sh 2>&1); echo "$gate" | tail -4
echo "$gate" | grep -q " 0 fail" || {
  echo "FATAL: R2 gate failed on this node - no timings are taken (R4)." >&2
  echo "       A fast wrong answer is not a result." >&2
  exit 1; }

echo "=== piece H: 50 steps, backends [$BACKENDS_H], grid ladder  $(date -u +%FT%TZ)"
for nx in 100 200 400 1000 2000; do
  NX=$nx NZ=30 STEPS=50 BACKENDS="$BACKENDS_H" REPEAT=5 \
    CASES="${CASES_H:-baroclinic_igw}" SCHEMES="fb theta0.5 split" SOLVERS="multigrid pcg_jacobi" \
    CFLS=4 CFLS_FB=0.5 bash tools/tier2_sweep.sh
done
echo "=== done $(date -u +%FT%TZ)"
