#!/usr/bin/env bash
#########################################################################
#  Module: scheme_hw3d.sh                                               #
#  Description: The scheme x hardware plate of docs/22, run on whatever #
#               GPU this node has. Semi-implicit theta against          #
#               split-explicit at the target grid, swept over CFL, with #
#               the accuracy reported alongside so the comparison stays #
#               a cost comparison (R8).                                 #
#  Pipeline: toml2nml -> cfd_exp3d{,_acc,_cuda} -> CSV -> docs/27       #
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
NVHPC_BIN="${NVHPC_BIN:-}"
[ -n "$NVHPC_BIN" ] && export PATH="$NVHPC_BIN:$PATH" && \
    export LD_LIBRARY_PATH="${NVHPC_BIN%/bin}/lib:${LD_LIBRARY_PATH:-}"
BACKENDS="${BACKENDS:-cpu gpu}"
want() { case " $BACKENDS " in *" $1 "*) return 0;; esac; return 1; }
source "$(dirname "$0")/require_idle_node.sh"
want cpu || MAX_CPU_LOAD=1e9
require_idle_node || exit 1

NX="${NX:-100}"; NZ="${NZ:-30}"; STEPS="${STEPS:-20}"; REPEAT="${REPEAT:-5}"
CFLS="${CFLS:-0.5 2 8 32}"
CASE="${CASE:-baroclinic_igw}"
THREADS="${THREADS:-16}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/scheme3d_${STAMP}"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "backend,threads,scheme,cfl,nx,nz,n_steps,wall_s,wall_mad_s,l2_rel_b,work_per_step" > "$CSV"
COMMON="--set solver.kind=pcg_jacobi --set grid.Lx=4.0e5 --set grid.Ly=4.0e5"

record() {  # record <backend> <threads> <json> <scheme> <cfl>
  "$PY" - "$3" "$1" "$2" "$4" "$5" "$NX" "$NZ" >> "$CSV" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
backend, threads, scheme, cfl, nx, nz = sys.argv[2:8]
def num(k):
    v = d.get(k)
    return "nan" if v is None else f"{float(v):.6e}"
work = d.get("solver_iterations", 0) or d.get("barotropic_substeps", 0) or 0
print(",".join([backend, threads, scheme, cfl, nx, nz, str(d.get("n_steps", 0)),
                num("wall_s"), num("wall_mad_s"), num("l2_rel_b"), str(work)]))
PYEOF
}

echo "host=$(hostname) nx=$NX nz=$NZ steps=$STEPS case=$CASE"
printf "\n%-16s %6s %12s %12s %12s %12s\n" scheme cfl "cpu[s]" "openacc[s]" "cuda[s]" "L2(b)"
for scheme in theta split_explicit; do
  for cfl in $CFLS; do
    tag="${scheme}_c${cfl}"
    "$PY" -m tools.toml2nml $COMMON --case "$CASE" --nx "$NX" --nz "$NZ" \
        --cfl "$cfl" --steps "$STEPS" --out "$OUT/$tag.nml" --prefix "$OUT/$tag" \
        --n-repeat "$REPEAT" --n-warmup 1 \
        --set "scheme.name=$scheme" > /dev/null || { echo "nml $tag failed"; continue; }
    sed -i 's/omp_min_points = .*/omp_min_points = 0/' "$OUT/$tag.nml"
    sed -i "s/pcg_sync = .*/pcg_sync = 'device'/" "$OUT/$tag.nml"
    cpu="-"; acc="-"; cu="-"; l2="-"
    if want cpu; then
      OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=close OMP_PLACES=cores \
        libs/fortran/build/cfd_exp3d "$OUT/$tag.nml" >/dev/null 2>&1 && {
        cpu=$(grep -o '"wall_s":[^,]*' "$OUT/${tag}_metrics.json" | sed 's/.*: *//')
        record fortran_omp "$THREADS" "$OUT/${tag}_metrics.json" "$scheme" "$cfl"; }
    fi
    if want gpu && [ -x libs/fortran/build/cfd_exp3d_acc ]; then
      libs/fortran/build/cfd_exp3d_acc "$OUT/$tag.nml" >/dev/null 2>&1 && {
        acc=$(grep -o '"wall_s":[^,]*' "$OUT/${tag}_metrics.json" | sed 's/.*: *//')
        record openacc 0 "$OUT/${tag}_metrics.json" "$scheme" "$cfl"; }
    fi
    if want gpu && [ -x libs/cuda/build/cfd_exp3d_cuda ]; then
      libs/cuda/build/cfd_exp3d_cuda "$OUT/$tag.nml" >/dev/null 2>&1 && {
        cu=$(grep -o '"wall_s":[^,]*' "$OUT/${tag}_metrics.json" | sed 's/.*: *//')
        l2=$(grep -o '"l2_rel_b":[^,}]*' "$OUT/${tag}_metrics.json" | sed 's/.*: *//')
        record cuda 0 "$OUT/${tag}_metrics.json" "$scheme" "$cfl"; }
    fi
    printf "%-16s %6s %12s %12s %12s %12s\n" "$scheme" "$cfl" "$cpu" "$acc" "$cu" "$l2"
  done
done
echo; echo "csv: $CSV"
recheck_idle_node || true
