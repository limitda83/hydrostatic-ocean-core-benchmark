#!/bin/bash
#PBS -N cfd_omp_policy
#PBS -l select=1:ncpus=192
#PBS -l place=excl
#PBS -j oe
#PBS -V
#
# Diagnose the erratic OpenMP scaling past 16 threads (docs/11 section 4).
# Hypothesis: libgomp's default active spin-wait plus NUMA-crossing barriers,
# not raw fork/join count. Compares runtime policies at fixed thread counts
# BEFORE committing to a source restructure - fix the cause, not the symptom.
#
# submit: qsub -v NX=512 tools/pbs_omp_policy.sh
set -euo pipefail
cd "${PBS_O_WORKDIR:-$HOME/cfd_exp}"
NX="${NX:-512}"; CASE="${CASE:-igw}"; CFL="${CFL:-0.5}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/omp_policy_nx${NX}_${STAMP}"; mkdir -p "$OUT"

echo "host=$(hostname) nx=${NX}"
./.venv/bin/python -m tools.toml2nml --set solver.kind=pcg_jacobi \
    --case "${CASE}" --nx "${NX}" --cfl "${CFL}" \
    --out "${OUT}/run.nml" --prefix "${OUT}/run" --n-repeat 5 --n-warmup 1 > /dev/null

for t in 8 32 64; do
  for policy in "active:close:cores" "passive:close:cores" "passive:spread:cores" "active:false:none"; do
    IFS=: read -r wait bind places <<< "$policy"
    echo "=== threads=${t} OMP_WAIT_POLICY=${wait} PROC_BIND=${bind} PLACES=${places} ==="
    env OMP_NUM_THREADS="$t" OMP_WAIT_POLICY="$wait" OMP_PROC_BIND="$bind" \
        OMP_PLACES="$places" GOMP_SPINCOUNT=$([ "$wait" = passive ] && echo 0 || echo 300000) \
        ./libs/fortran/build/cfd_exp "${OUT}/run.nml" 2>&1 | grep "wall median"
  done
done
echo "results in ${OUT}"
