#!/bin/bash
# Grid-size sweep comparing the CPU and GPU backends ON THE SAME NODE.
# This is the only valid setting for RQ5 (where does GPU overtake CPU?) and
# RQ2 (does the ranking invert between CPU and GPU) - see RULES.md R7-1.
#
# usage: CUDA_VISIBLE_DEVICES=<free gpu> tools/bench_grid_sweep.sh [nx ...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin:$PATH"

# Refuse to produce timings on a contended shared node (R7-7).
source "$(dirname "$0")/require_idle_node.sh"
require_idle_node || exit 1

SIZES="${*:-64 128 256 512 1024}"
CASE="${CASE:-igw}"; CFL="${CFL:-0.5}"; REPEAT="${REPEAT:-3}"
CPU_THREADS="${CPU_THREADS:-16}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/grid_sweep_${STAMP}"; mkdir -p "$OUT"

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-0}  cpu_threads=${CPU_THREADS}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
printf "\n%6s %10s %14s %14s %14s %10s\n" nx points "cpu_1t[s]" "cpu_${CPU_THREADS}t[s]" "gpu_acc[s]" "gpu/cpu"

for nx in $SIZES; do
  ./.venv/bin/python -m tools.toml2nml --set solver.kind=pcg_jacobi \
      --case "$CASE" --nx "$nx" --cfl "$CFL" \
      --out "$OUT/n${nx}.nml" --prefix "$OUT/n${nx}" \
      --n-repeat "$REPEAT" --n-warmup 1 > /dev/null
  # omp_min_points must not serialise the CPU runs we are timing.
  sed -i 's/omp_min_points = .*/omp_min_points = 0/' "$OUT/n${nx}.nml"

  t1=$(OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
       ./libs/fortran/build/cfd_exp "$OUT/n${nx}.nml" | awk '/wall median/{print $3}')
  cp "$OUT/n${nx}_metrics.json" "$OUT/n${nx}_cpu1.json"
  tn=$(OMP_NUM_THREADS=$CPU_THREADS OMP_PROC_BIND=close OMP_PLACES=cores \
       ./libs/fortran/build/cfd_exp "$OUT/n${nx}.nml" | awk '/wall median/{print $3}')
  cp "$OUT/n${nx}_metrics.json" "$OUT/n${nx}_cpu${CPU_THREADS}.json"
  tg=$(./libs/fortran/build/cfd_exp_acc "$OUT/n${nx}.nml" | awk '/wall median/{print $3}')
  cp "$OUT/n${nx}_metrics.json" "$OUT/n${nx}_gpu.json"

  printf "%6d %10d %14s %14s %14s %10s\n" "$nx" $((nx*nx)) "$t1" "$tn" "$tg" \
     "$(awk -v a="$tn" -v b="$tg" 'BEGIN{printf "%.2fx", a/b}')"
done
echo
echo "results in $OUT"
recheck_idle_node || true
