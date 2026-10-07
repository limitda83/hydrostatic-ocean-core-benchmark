#!/bin/bash
#PBS -N cfd_helm
#PBS -l select=1:ncpus=192
#PBS -l place=excl
#PBS -j oe
#PBS -V
#
# Variable-coefficient Helmholtz thread sweep on a geo85 compute node
# (192 cores, exclusive). geo85 has no GPU: these numbers belong to the
# CPU-scalability axis only and must never be paired with a GPU timing
# from another node (RULES.md R7-1).
#
# submit: qsub -v NX=512,CFL=32,TOPO=rough,RT=0.10 tools/pbs_helm_sweep.sh
set -uo pipefail
cd "${PBS_O_WORKDIR:-$HOME/cfd_exp}"
NX="${NX:-512}"; CFL="${CFL:-32}"; TOPO="${TOPO:-rough}"; RT="${RT:-0.10}"
SOLVERS="${SOLVERS:-pcg_jacobi pcg_rbgs multigrid}"
THREADS="${THREADS:-1 2 4 8 16 32 64 128 192}"
PY="${PY:-./.venv/bin/python}"
OUT="output/helm_omp_nx${NX}_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
# R7-2: the dispersion travels with the number (the pre-2026-09-13 data set
# was min-of-3 and is not usable in the manuscript).
echo "backend,threads,nx,topo,r_std,rx0,cfl,solver,iterations,wall_s,wall_mad_s,wall_min_s,n_repeat" > "$CSV"

echo "host=$(hostname) nx=${NX} cfl=${CFL} topo=${TOPO}${RT}"
lscpu | egrep "^Model name|^CPU\(s\)|^NUMA node\(s\)" || true

"$PY" tools/write_helmholtz_case.py --nx "$NX" --topo "$TOPO" --r-target "$RT" \
    --cfl "$CFL" --Lx 4.0e5 --out "$OUT/case" --reference > /dev/null

for sv in $SOLVERS; do
  echo; echo "=== solver=${sv} ==="
  printf "%8s %12s %10s %9s\n" threads "wall [s]" speedup iters
  base=""
  for t in $THREADS; do
    rm -f "$OUT/case/metrics_${sv}.json"
    OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
      libs/fortran/build/helmholtz_bench "$OUT/case/case.nml" "$sv" > /dev/null 2>&1
    [ -f "$OUT/case/metrics_${sv}.json" ] || { echo "  t=$t FAILED"; continue; }
    # `read` collapses a run of IFS-whitespace, and tab IS IFS-whitespace, so
    # setting IFS=$'\t' alone does NOT stop an empty middle field from shifting
    # every later value left (verified). What prevents it is the 'NA'
    # placeholder the emitter substitutes for an empty value; the field-count
    # check below then catches a short list. Both are needed.
    IFS=$'\t' read -r w it mad wmin nrep < <("$PY" -c "
import json,sys
m=json.load(open('$OUT/case/metrics_${sv}.json'))
f=[m['wall_s'], m['iterations'], m.get('wall_mad_s','nan'),
   m.get('wall_min_s','nan'), m.get('n_repeat','NA')]
print('\t'.join('NA' if x is None or x=='' else str(x) for x in f))")
    if [ -z "$w" ] || [ -z "$it" ] || [ -z "$nrep" ]; then
      echo "  t=$t MALFORMED METRICS (w='$w' it='$it' mad='$mad' wmin='$wmin' nrep='$nrep')"
      continue
    fi
    [ -z "$base" ] && base="$w"
    # Nested same-type quotes inside an f-string are a syntax error before
    # Python 3.12, and geo85 runs 3.9: keep the values as argv, not literals.
    sp=$("$PY" -c 'import sys; print("%.2f" % (float(sys.argv[1]) / float(sys.argv[2])))' "$base" "$w")
    printf "%8s %12.5f %9sx %9s\n" "$t" "$w" "$sp" "$it"
    "$PY" -c "
c=__import__('json').load(open('$OUT/case/case.json'))
print(','.join(['fortran_omp','$t','$NX','$TOPO$RT',f\"{c['r_std']:.6f}\",f\"{c['rx0']:.6f}\",'$CFL','$sv','$it','$w','$mad','$wmin','$nrep']))" >> "$CSV"
  done
done
echo; echo "csv: $CSV"
