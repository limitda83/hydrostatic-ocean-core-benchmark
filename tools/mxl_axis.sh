#!/usr/bin/env bash
#########################################################################
#  Module: mxl_axis.sh                                                  #
#  Description: The S11.2 mixing-length axis: the same TKE closure with  #
#               the O(nz^2) potential-energy budget of Gaspar et al.     #
#               (1990) and with the two O(nz) sweeps of NEMO nn_mxl=2,   #
#               on a byte-identical fixed-step problem. Crossed with     #
#               precision, because the closure is the arithmetic-bound   #
#               kernel that a card with 1/48 fp64 throughput punishes    #
#               (docs/34, docs/35). rtol is pinned for both forms so the #
#               iteration counts match and the difference is cost.       #
#  Pipeline: nsys profile (docs/35) -> mxl_axis.sh -> tier2_collect      #
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
NZ="${NZ:-30}"; STEPS="${STEPS:-50}"; REPEAT="${REPEAT:-5}"
GRIDS="${GRIDS:-400 1000}"
BACKENDS="${BACKENDS:-cuda cuda_sp}"
export EXTRA_SET="${EXTRA_SET:-} --set solver.rtol=1e-6"
for nx in $GRIDS; do
  cfl=4; [ "$nx" = 2000 ] && cfl=2
  NX=$nx NZ=$NZ STEPS=$STEPS REPEAT=$REPEAT BACKENDS="$BACKENDS" \
    CASES="lock_exchange_v06 lock_exchange_v06r" SCHEMES="${SCHEMES:-fb theta0.5}" \
    SOLVERS="${SOLVERS:-multigrid}" CFLS=$cfl CFLS_FB=0.5 TAG=".mxl" \
    bash tools/tier2_sweep.sh
done
echo "=== mxl axis done $(date -u +%FT%TZ)"
