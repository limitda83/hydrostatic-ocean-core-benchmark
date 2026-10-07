#!/usr/bin/env bash
#########################################################################
#  Module: precision_axis.sh                                            #
#  Description: Axis G of docs/04 for the full v0.6 core: the same       #
#               fixed-step problem in fp64 and fp32 on one device, with  #
#               rtol = 1e-6 for BOTH precisions so the iteration counts  #
#               match and the difference is cost, not algorithm. Runs    #
#               the dynamical core (v0.5) and the representative physics #
#               (v0.6) so the interaction of precision with physics is   #
#               visible - the question a card with 1/48 fp64 throughput  #
#               raises (docs/12, docs/33 finding 2).                      #
#  Pipeline: precision_axis.sh -> tier2_sweep -> CSV (backend *_sp)      #
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
NZ="${NZ:-30}"; STEPS="${STEPS:-50}"; REPEAT="${REPEAT:-5}"
GRIDS="${GRIDS:-400 1000 2000}"
BACKENDS="${BACKENDS:-cuda cuda_sp openacc openacc_sp}"
export EXTRA_SET="${EXTRA_SET:-} --set solver.rtol=1e-6"
for nx in $GRIDS; do
  cfl=4; [ "$nx" = 2000 ] && cfl=2
  NX=$nx NZ=$NZ STEPS=$STEPS REPEAT=$REPEAT BACKENDS="$BACKENDS" \
    CASES="lock_exchange lock_exchange_v06" SCHEMES="${SCHEMES:-fb theta0.5}" \
    SOLVERS="${SOLVERS:-multigrid}" CFLS=$cfl CFLS_FB=0.5 TAG=".rtol6" \
    bash tools/tier2_sweep.sh
done
echo "=== precision axis done $(date -u +%FT%TZ)"
