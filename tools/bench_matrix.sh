#!/bin/bash
# Full performance matrix on ONE node (RULES.md R7-1).
#
#   Part A  cost:     fixed step count, grid sweep, every backend
#   Part B  accuracy: fixed physical time, CFL x scheme sweep -> Pareto data
#
# Every backend here has passed the R2 gate; no unverified code is timed (R4).
# usage: CUDA_VISIBLE_DEVICES=<free gpu> tools/bench_matrix.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin:$PATH"

# Refuse to produce timings on a contended shared node (R7-7).
source "$(dirname "$0")/require_idle_node.sh"
require_idle_node || exit 1

CASE="${CASE:-igw_broadband}"
# Part B is worth running for BOTH regimes:
#   igw_broadband - the fast gravity wave IS the signal, so no scheme can take
#                   a long step without losing the answer
#   geo_balance   - the fast wave is noise around a slow steady state, which is
#                   the regime semi-implicit schemes were designed for
PART_B_ONLY="${PART_B_ONLY:-0}"
REPEAT="${REPEAT:-5}"
STEPS="${STEPS:-50}"
PART_B_NX="${PART_B_NX:-512}"
EXTRA_SET="${EXTRA_SET:-}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/matrix_${STAMP}"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "part,backend,threads,nx,cfl,scheme,theta,n_steps,wall_s,mad_s,l2_rel_eta,pcg_iters" > "$CSV"

nml() {  # nml <tag> <nx> <cfl> <scheme> <theta> [--steps N]
  ./.venv/bin/python -m tools.toml2nml --set solver.kind=pcg_jacobi \
      --set scheme.name="$4" --set scheme.theta="$5" ${EXTRA_SET} \
      --case "$CASE" --nx "$2" --cfl "$3" ${6:+--steps $6} \
      --out "$OUT/$1.nml" --prefix "$OUT/$1" --n-repeat "$REPEAT" --n-warmup 1 >/dev/null
  sed -i 's/omp_min_points = .*/omp_min_points = 0/' "$OUT/$1.nml"
}

record() {  # record <part> <backend> <threads> <tag> <nx> <cfl> <scheme> <theta>
  local j="$OUT/$4_metrics.json"
  [ -f "$j" ] || { echo "$1,$2,$3,$5,$6,$7,$8,NA,NA,NA,NA,NA" >> "$CSV"; return; }
  ./.venv/bin/python - "$j" "$1" "$2" "$3" "$5" "$6" "$7" "$8" >> "$CSV" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
part, backend, threads, nx, cfl, scheme, theta = sys.argv[2:9]

def num(key, default=0.0):
    """A diverged run writes JSON null; emit nan so the row is still recorded
    rather than dropped (RULES.md R12 - failures are results too)."""
    v = d.get(key, default)
    return "nan" if v is None else f"{float(v):.6e}"

print(",".join([part, backend, threads, nx, cfl, scheme, theta,
                str(d.get("n_steps", 0)), num("wall_s"), num("wall_mad_s"),
                num("l2_rel_eta"), str(d.get("solver_iterations", 0) or 0)]))
PY
}

run_cpu() {  # run_cpu <threads> <binary> <tag>
  OMP_NUM_THREADS="$1" OMP_PROC_BIND=close OMP_PLACES=cores "$2" "$OUT/$3.nml" >/dev/null 2>&1
}

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-0}  case=$CASE  repeat=$REPEAT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

# ---------------- Part A: cost at a pinned step count -------------------
if [ "$PART_B_ONLY" != "1" ]; then
echo; echo "=== Part A: cost, ${STEPS} steps, cfl=2.0, theta=0.5 ==="
       nx numpy cpu_serial cpu_1t cpu_16t cpu_32t openacc cuda
for nx in 64 128 256 512 1024; do
  tag="A_n${nx}"
  nml "$tag" "$nx" 2.0 theta 0.5 "$STEPS"

  np="skip"
  if [ "$nx" -le 512 ]; then
    np=$(CASE="$CASE" NX="$nx" STEPS="$STEPS" ./.venv/bin/python - <<'PY'
import os, sys, time; sys.path.insert(0, '.')
from libs.utils.config import load_config
import main as M
cfg = load_config().with_overrides({'solver.kind': 'pcg_jacobi'})
r = M.simulate(cfg, int(os.environ['NX']), 2.0, os.environ['CASE'],
               n_steps_override=int(os.environ['STEPS']))['metrics']
print(f"{r['wall_s']:.6e}")
PY
)
  fi

  run_cpu 1  libs/fortran/build/cfd_exp_serial "$tag"; s1=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2); record A fortran_serial 1 "$tag" "$nx" 2.0 theta 0.5
  run_cpu 1  libs/fortran/build/cfd_exp "$tag";        c1=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2); record A fortran_omp 1 "$tag" "$nx" 2.0 theta 0.5
  run_cpu 16 libs/fortran/build/cfd_exp "$tag";        c16=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2); record A fortran_omp 16 "$tag" "$nx" 2.0 theta 0.5
  run_cpu 32 libs/fortran/build/cfd_exp "$tag";        c32=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2); record A fortran_omp 32 "$tag" "$nx" 2.0 theta 0.5
  libs/fortran/build/cfd_exp_acc "$OUT/$tag.nml" >/dev/null 2>&1; ac=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2); record A openacc 0 "$tag" "$nx" 2.0 theta 0.5
  libs/cuda/build/cfd_exp_cuda   "$OUT/$tag.nml" >/dev/null 2>&1; cu=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2); record A cuda 0 "$tag" "$nx" 2.0 theta 0.5
  echo "$np" > /dev/null
  printf "%6d %12s %12s %12s %12s %12s %12s %12s\n" "$nx" "$np" "$s1" "$c1" "$c16" "$c32" "$ac" "$cu"
  echo "A,numpy_ref,1,$nx,2.0,theta,0.5,$STEPS,$np,NA,NA,NA" >> "$CSV"
done


fi

# --------- Part B: accuracy vs cost at fixed physical time --------------
echo; echo "=== Part B: error-vs-cost, nx=$PART_B_NX, fixed physical time, case=$CASE ==="
printf "%-10s %6s %8s %14s %12s %12s %10s\n" scheme cfl steps "L2(eta)" "cpu16t[s]" "cuda[s]" pcg/solve
for sch in "fb 0.5" "theta 0.5" "theta 0.6" "theta 1.0"; do
  set -- $sch; s=$1; th=$2
  for cfl in 0.5 2.0 8.0 32.0; do
    tag="B_${s}${th}_c${cfl}"
    nml "$tag" "$PART_B_NX" "$cfl" "$s" "$th"
    run_cpu 16 libs/fortran/build/cfd_exp "$tag"
    cw=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2)
    l2=$(grep -o '"l2_rel_eta": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2)
    st=$(grep -o '"n_steps": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2)
    pi=$(grep -o '"solver_iterations": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2)
    record B fortran_omp 16 "$tag" "$PART_B_NX" "$cfl" "$s" "$th"
    libs/cuda/build/cfd_exp_cuda "$OUT/$tag.nml" >/dev/null 2>&1
    gw=$(grep -o '"wall_s": [^,]*' "$OUT/${tag}_metrics.json" | cut -d' ' -f2)
    record B cuda 0 "$tag" "$PART_B_NX" "$cfl" "$s" "$th"
    ips=$(awk -v p="$pi" -v n="$st" 'BEGIN{printf "%.1f", (n>0)? p/(n*2) : 0}')
    printf "%-10s %6s %8s %14s %12s %12s %10s\n" "$s$th" "$cfl" "$st" "$l2" "$cw" "$gw" "$ips"
  done
done
echo; echo "csv: $CSV"
recheck_idle_node || true
