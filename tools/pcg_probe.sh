#!/bin/bash
# Isolate what makes the 3D GPU run launch-bound (RQ2).
#
# The semi-implicit free surface needs a PCG solve whose iteration count grows
# as CFL^2. Every PCG iteration ends in a global reduction, and on GPU each
# reduction is a device->host synchronisation. Sweeping CFL at a fixed grid
# therefore varies the number of synchronisations while holding everything
# else constant, which is exactly the measurement RQ2 asks for.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin:$PATH"

# Refuse to produce timings on a contended shared node (R7-7).
source "$(dirname "$0")/require_idle_node.sh"
require_idle_node || exit 1

NX="${NX:-100}"; NZ="${NZ:-30}"; STEPS="${STEPS:-20}"; REPEAT="${REPEAT:-5}"
KERNEL="${KERNEL:-column}"
CFLS="${*:-0.125 0.5 2.0 8.0}"
OUT="output/pcg_probe_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"

val() { grep -o "\"$2\":[^,]*" "$1" | sed 's/.*: *//' | tr -d '"'; }

echo "host=$(hostname) nx=$NX nz=$NZ steps=$STEPS kernel=$KERNEL"
printf "%7s %11s %12s %12s %12s %14s\n" cfl "pcg/step" "omp16 [s]" "acc [s]" "acc/omp16" "acc ms/step"
for cfl in $CFLS; do
  ./.venv/bin/python -m tools.toml2nml --set solver.kind=pcg_jacobi \
      --set grid.Lx=4.0e5 --set grid.Ly=4.0e5 --case baroclinic_igw \
      --nx "$NX" --nz "$NZ" --cfl "$cfl" --steps "$STEPS" \
      --out "$OUT/c.nml" --prefix "$OUT/c" --n-repeat "$REPEAT" --n-warmup 1 >/dev/null
  sed -i "s/omp_min_points = .*/omp_min_points = 0/" "$OUT/c.nml"
  sed -i "s/tridiag_kernel = .*/tridiag_kernel = '${KERNEL}'/" "$OUT/c.nml"

  OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    libs/fortran/build/cfd_exp3d "$OUT/c.nml" >/dev/null 2>&1
  cw=$(val "$OUT/c_metrics.json" wall_s)
  libs/fortran/build/cfd_exp3d_acc "$OUT/c.nml" >/dev/null 2>&1
  gw=$(val "$OUT/c_metrics.json" wall_s)
  pc=$(val "$OUT/c_metrics.json" solver_iterations)

  CW="$cw" GW="$gw" PC="$pc" STEPS="$STEPS" CFL="$cfl" ./.venv/bin/python - <<'PY'
import os
cw, gw = float(os.environ['CW']), float(os.environ['GW'])
pc, steps = int(float(os.environ['PC'])), int(os.environ['STEPS'])
print(f"{float(os.environ['CFL']):7.3f} {pc/steps:11.1f} {cw:12.4f} {gw:12.4f} "
      f"{cw/gw:12.2f} {gw/steps*1000:14.2f}")
PY
done
echo "results in $OUT"
recheck_idle_node || true
