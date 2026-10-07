#!/bin/bash
# Does split-explicit remove the semi-implicit scheme's GPU penalty?
#
# docs/21 S3 found the GPU losing on the elliptic solve: every PCG iteration
# ends in a global reduction, and the two cost curves cross at 208 iterations
# per step. split-explicit reaches the same large time step with barotropic
# substeps that are pure local stencils - no elliptic solve, no reduction. This
# sweeps CFL for both schemes on CPU and both GPU backends, holding the grid,
# the case and the answer fixed.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin:$PATH"

source "$(dirname "$0")/require_idle_node.sh"
require_idle_node || exit 1

NX="${NX:-100}"; NZ="${NZ:-30}"; STEPS="${STEPS:-20}"; REPEAT="${REPEAT:-5}"
CFLS="${*:-0.5 2.0 8.0 32.0}"
OUT="output/scheme_hw_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "scheme,cfl,n_steps,work_per_step,l2_rel_b,omp16_s,openacc_s,cuda_s" > "$CSV"
val() { grep -o "\"$2\":[^,]*" "$1" | sed 's/.*: *//' | tr -d '"'; }
run_or_die() { local l="$1"; shift; rm -f "$OUT/c_metrics.json"
  "$@" > "$OUT/last.log" 2>&1 || { echo "FATAL: $l failed"; sed -n 1,8p "$OUT/last.log"; exit 1; }
  [ -f "$OUT/c_metrics.json" ] || { echo "FATAL: $l wrote no metrics"; exit 1; }; }

echo "host=$(hostname) nx=$NX nz=$NZ steps=$STEPS"
printf "\n%-16s %6s %13s %12s %11s %11s %11s %9s\n" \
       scheme cfl "work/step" "L2(b)" "omp16 [s]" "acc [s]" "cuda [s]" "cuda/cpu"
for sch in theta split_explicit; do
  for cfl in $CFLS; do
    ./.venv/bin/python -m tools.toml2nml --set solver.kind=pcg_jacobi \
        --set scheme.name="$sch" --set grid.Lx=4.0e5 --set grid.Ly=4.0e5 \
        --case baroclinic_igw --nx "$NX" --nz "$NZ" --cfl "$cfl" --steps "$STEPS" \
        --out "$OUT/c.nml" --prefix "$OUT/c" --n-repeat "$REPEAT" --n-warmup 1 >/dev/null
    sed -i "s/omp_min_points = .*/omp_min_points = 0/" "$OUT/c.nml"
    sed -i "s/tridiag_kernel = .*/tridiag_kernel = 'column'/" "$OUT/c.nml"
    sed -i "s/pcg_sync = .*/pcg_sync = 'device'/" "$OUT/c.nml"

    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
      run_or_die "fortran3d $sch cfl=$cfl" libs/fortran/build/cfd_exp3d "$OUT/c.nml"
    cw=$(val "$OUT/c_metrics.json" wall_s); l2=$(val "$OUT/c_metrics.json" l2_rel_b)
    st=$(val "$OUT/c_metrics.json" n_steps)
    if [ "$sch" = "theta" ]; then
      work=$(val "$OUT/c_metrics.json" solver_iterations); unit="PCG"
    else
      work=$(val "$OUT/c_metrics.json" barotropic_substeps); unit="sub"
    fi
    run_or_die "openacc3d $sch cfl=$cfl" libs/fortran/build/cfd_exp3d_acc "$OUT/c.nml"
    aw=$(val "$OUT/c_metrics.json" wall_s)
    run_or_die "cuda3d $sch cfl=$cfl" libs/cuda/build/cfd_exp3d_cuda "$OUT/c.nml"
    gw=$(val "$OUT/c_metrics.json" wall_s)

    SCH="$sch" CFL="$cfl" ST="$st" WORK="$work" UNIT="$unit" L2="$l2" \
    CW="$cw" AW="$aw" GW="$gw" CSVF="$CSV" ./.venv/bin/python - <<'PY'
import os
w = float(os.environ['WORK']); st = int(float(os.environ['ST']))
cw, aw, gw = (float(os.environ[k]) for k in ('CW','AW','GW'))
per = w / st
print(f"{os.environ['SCH']:<16} {float(os.environ['CFL']):6.1f} "
      f"{per:9.1f} {os.environ['UNIT']:>3} {float(os.environ['L2']):12.4e} "
      f"{cw:11.4f} {aw:11.4f} {gw:11.4f} {cw/gw:9.2f}")
with open(os.environ['CSVF'], 'a') as f:
    f.write(f"{os.environ['SCH']},{os.environ['CFL']},{st},{per:.2f},"
            f"{float(os.environ['L2']):.6e},{cw:.6e},{aw:.6e},{gw:.6e}\n")
PY
  done
done
echo; echo "csv: $CSV"
recheck_idle_node || true
