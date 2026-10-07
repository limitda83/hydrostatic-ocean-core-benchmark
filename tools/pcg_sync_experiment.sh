#!/bin/bash
# The decisive test of the docs/21 S3.2 hypothesis: that the GPU loses on the
# semi-implicit elliptic solve because every PCG inner product is a
# device-to-host synchronisation.
#
# One binary, one GPU, one problem; only the synchronisation strategy changes.
#   host   - copy every inner product back (what OpenACC's reduction does)
#   device - alpha/beta stay in device memory; only the convergence test
#            copies, every pcg_check_every iterations
# OpenACC and the CPU are included as reference points.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin:$PATH"

# Refuse to produce timings on a contended shared node (R7-7).
source "$(dirname "$0")/require_idle_node.sh"
require_idle_node || exit 1

NX="${NX:-100}"; NZ="${NZ:-30}"; STEPS="${STEPS:-20}"; REPEAT="${REPEAT:-5}"
CHECK="${CHECK:-5}"
CFLS="${*:-0.125 0.5 2.0 8.0}"
OUT="output/pcg_sync_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
val() { grep -o "\"$2\":[^,]*" "$1" | sed 's/.*: *//' | tr -d '"'; }

# Stale metrics files silently produce wrong rows, so every backend run must
# be seen to succeed and to have rewritten its output.
run_or_die() {  # run_or_die <label> <command...>
  local label="$1"; shift
  rm -f "$OUT/c_metrics.json"
  if ! "$@" > "$OUT/last.log" 2>&1; then
    echo "FATAL: $label failed"; sed -n '1,10p' "$OUT/last.log"; exit 1
  fi
  if [ ! -f "$OUT/c_metrics.json" ]; then
    echo "FATAL: $label wrote no metrics"; sed -n '1,10p' "$OUT/last.log"; exit 1
  fi
}

echo "host=$(hostname) nx=$NX nz=$NZ steps=$STEPS check_every=$CHECK"
printf "%7s %10s %11s %13s %13s %11s %12s\n" \
       cfl "pcg/step" "omp16 [s]" "acc [s]" "cuda-host[s]" "cuda-dev[s]" "dev/host"
for cfl in $CFLS; do
  ./.venv/bin/python -m tools.toml2nml --set solver.kind=pcg_jacobi \
      --set grid.Lx=4.0e5 --set grid.Ly=4.0e5 --case baroclinic_igw \
      --nx "$NX" --nz "$NZ" --cfl "$cfl" --steps "$STEPS" \
      --out "$OUT/c.nml" --prefix "$OUT/c" --n-repeat "$REPEAT" --n-warmup 1 >/dev/null
  sed -i "s/omp_min_points = .*/omp_min_points = 0/" "$OUT/c.nml"
  sed -i "s/tridiag_kernel = .*/tridiag_kernel = 'column'/" "$OUT/c.nml"
  sed -i "s/pcg_check_every = .*/pcg_check_every = ${CHECK}/" "$OUT/c.nml"

  sed -i "s/pcg_sync = .*/pcg_sync = 'device'/" "$OUT/c.nml"
  OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    run_or_die "fortran3d omp16" libs/fortran/build/cfd_exp3d "$OUT/c.nml"
  cw=$(val "$OUT/c_metrics.json" wall_s); pc=$(val "$OUT/c_metrics.json" solver_iterations)
  run_or_die "openacc3d" libs/fortran/build/cfd_exp3d_acc "$OUT/c.nml"
  aw=$(val "$OUT/c_metrics.json" wall_s)
  run_or_die "cuda3d device" libs/cuda/build/cfd_exp3d_cuda "$OUT/c.nml"
  dw=$(val "$OUT/c_metrics.json" wall_s)
  sed -i "s/pcg_sync = .*/pcg_sync = 'host'/" "$OUT/c.nml"
  run_or_die "cuda3d host" libs/cuda/build/cfd_exp3d_cuda "$OUT/c.nml"
  hw=$(val "$OUT/c_metrics.json" wall_s)

  CW="$cw" AW="$aw" HW="$hw" DW="$dw" PC="$pc" ST="$STEPS" CFL="$cfl" \
  ./.venv/bin/python - <<'PY'
import os
cw, aw, hw, dw = (float(os.environ[k]) for k in ('CW', 'AW', 'HW', 'DW'))
pc, st = int(float(os.environ['PC'])), int(os.environ['ST'])
print(f"{float(os.environ['CFL']):7.3f} {pc/st:10.1f} {cw:11.4f} {aw:13.4f} "
      f"{hw:13.4f} {dw:11.4f} {hw/dw:12.2f}")
PY
done
echo "results in $OUT"
recheck_idle_node || true
