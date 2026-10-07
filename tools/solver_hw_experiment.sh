#!/bin/bash
# The solver axis on GPU: does a multi-coloured smoother inside multigrid fix
# what the PCG's global reductions cost?
#
# docs/21 S3 measured the GPU losing on the elliptic solve because every CG
# iteration ends in three device-to-host synchronisations. A multigrid V-cycle
# needs one, and its iteration count barely grows with CFL. This sweeps CFL for
# theta+PCG, theta+multigrid and split-explicit on the same grid and case, so
# only the elliptic strategy changes.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin:$PATH"
source "$(dirname "$0")/require_idle_node.sh"
require_idle_node || exit 1

NX="${NX:-128}"; NZ="${NZ:-30}"; STEPS="${STEPS:-20}"; REPEAT="${REPEAT:-5}"
CFLS="${*:-0.5 2.0 8.0 32.0}"
OUT="output/solver_hw_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "config,cfl,n_steps,work_per_step,l2_rel_b,cuda_s,bytes_per_cell" > "$CSV"
val() { grep -o "\"$2\":[^,]*" "$1" | sed 's/.*: *//' | tr -d '"'; }
run_or_die() { local l="$1"; shift; rm -f "$OUT/c_metrics.json"
  "$@" > "$OUT/last.log" 2>&1 || { echo "FATAL: $l"; sed -n 1,8p "$OUT/last.log"; exit 1; }
  [ -f "$OUT/c_metrics.json" ] || { echo "FATAL: $l wrote nothing"; exit 1; }; }

echo "host=$(hostname) nx=$NX nz=$NZ steps=$STEPS"
printf "\n%-24s %6s %12s %13s %11s %10s\n" config cfl "work/step" "L2(b)" "cuda [s]" "B/cell"
for cfg in "theta:pcg_jacobi" "theta:multigrid" "split_explicit:pcg_jacobi"; do
  sch=${cfg%%:*}; slv=${cfg##*:}
  for cfl in $CFLS; do
    ./.venv/bin/python -m tools.toml2nml --set solver.kind="$slv" \
        --set scheme.name="$sch" --set grid.Lx=4.0e5 --set grid.Ly=4.0e5 \
        --case baroclinic_igw --nx "$NX" --nz "$NZ" --cfl "$cfl" --steps "$STEPS" \
        --out "$OUT/c.nml" --prefix "$OUT/c" --n-repeat "$REPEAT" --n-warmup 1 >/dev/null
    sed -i "s/tridiag_kernel = .*/tridiag_kernel = 'column'/" "$OUT/c.nml"
    sed -i "s/pcg_sync = .*/pcg_sync = 'device'/" "$OUT/c.nml"
    run_or_die "cuda3d $sch/$slv cfl=$cfl" libs/cuda/build/cfd_exp3d_cuda "$OUT/c.nml"
    gw=$(val "$OUT/c_metrics.json" wall_s); l2=$(val "$OUT/c_metrics.json" l2_rel_b)
    st=$(val "$OUT/c_metrics.json" n_steps); bc=$(val "$OUT/c_metrics.json" bytes_per_cell)
    if [ "$sch" = "split_explicit" ]; then
      work=$(val "$OUT/c_metrics.json" barotropic_substeps); unit="sub"
    elif [ "$slv" = "multigrid" ]; then
      work=$(val "$OUT/c_metrics.json" solver_iterations); unit="V"
    else
      work=$(val "$OUT/c_metrics.json" solver_iterations); unit="CG"
    fi
    NAME="$sch + $slv" CFL="$cfl" ST="$st" WORK="$work" UNIT="$unit" L2="$l2" \
    GW="$gw" BC="$bc" CSVF="$CSV" ./.venv/bin/python - <<'PY'
import os
w = float(os.environ['WORK']); st = int(float(os.environ['ST']))
gw, bc = float(os.environ['GW']), float(os.environ['BC'])
print(f"{os.environ['NAME']:<24} {float(os.environ['CFL']):6.1f} "
      f"{w/st:8.1f} {os.environ['UNIT']:>3} {float(os.environ['L2']):13.4e} "
      f"{gw:11.4f} {bc:10.0f}")
with open(os.environ['CSVF'], 'a') as f:
    f.write(f"{os.environ['NAME']},{os.environ['CFL']},{st},{w/st:.2f},"
            f"{float(os.environ['L2']):.6e},{gw:.6e},{bc:.1f}\n")
PY
  done
done
echo; echo "csv: $CSV"
recheck_idle_node || true
