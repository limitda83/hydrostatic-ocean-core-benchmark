#!/bin/bash
# 3D cost across grid sizes on ONE node (RULES.md R7-1).
# Answers whether the 2D conclusion - that the 100x100 target grid is too
# small for a GPU - inverts once the vertical dimension and its batched
# tridiagonal solves are present.
set -uo pipefail
cd "$(dirname "$0")/.."
# Node-portable knobs: PY picks the interpreter, NVHPC_BIN the nvfortran
# runtime libraries the OpenACC binary needs. Defaults reproduce gpgpu.
NVHPC_BIN="${NVHPC_BIN:-$HOME/opt/nvhpc/Linux_x86_64/25.11/compilers/bin}"
export PATH="$NVHPC_BIN:$PATH"
PY="${PY:-./.venv/bin/python}"

# Refuse to produce timings on a contended shared node (R7-7).
source "$(dirname "$0")/require_idle_node.sh"
require_idle_node || exit 1

CASE="${CASE:-baroclinic_igw}"
NZ="${NZ:-30}"
STEPS="${STEPS:-20}"
REPEAT="${REPEAT:-5}"
CFL="${CFL:-2.0}"
SIZES="${*:-50 100 200 400}"
COMMON="--set solver.kind=pcg_jacobi --set grid.Lx=4.0e5 --set grid.Ly=4.0e5"
THREADS_LIST="${THREADS_LIST:-1 16 32}"
TRIDIAG="${TRIDIAG:-plane}"   # plane | column (spec S7.4 kernel variant)
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/matrix3d_${STAMP}"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "backend,threads,nx,nz,cells,n_steps,wall_s,mad_s,l2_rel_b,pcg_iters" > "$CSV"

# The Fortran writer pads with es24.16, so the value carries leading spaces;
# cut -d' ' would return an empty field. Strip everything up to the colon.
wall() { grep -o '"wall_s":[^,]*' "$1" | sed 's/.*: *//'; }

record() {  # record <backend> <threads> <json> <nx>
  "$PY" - "$3" "$1" "$2" "$4" "$NZ" >> "$CSV" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
backend, threads, nx, nz = sys.argv[2:6]
def num(k):
    v = d.get(k)
    return "nan" if v is None else f"{float(v):.6e}"
print(",".join([backend, threads, nx, nz, str(int(nx)**2*int(nz)),
                str(d.get("n_steps", 0)), num("wall_s"), num("wall_mad_s"),
                num("l2_rel_b"), str(d.get("solver_iterations", 0) or 0)]))
PY
}

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-0} case=$CASE nz=$NZ steps=$STEPS tridiag=$TRIDIAG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
hdr="nx cells numpy f_serial"
for t in $THREADS_LIST; do hdr="$hdr f_omp$t"; done
hdr="$hdr openacc cuda"
printf "\n"; printf "%11s" $hdr; printf "\n"

for nx in $SIZES; do
  tag="n${nx}"
  "$PY" -m tools.toml2nml $COMMON --case "$CASE" --nx "$nx" --nz "$NZ" \
      --cfl "$CFL" --steps "$STEPS" --out "$OUT/$tag.nml" --prefix "$OUT/$tag" \
      --n-repeat "$REPEAT" --n-warmup 1 > /dev/null
  sed -i 's/omp_min_points = .*/omp_min_points = 0/' "$OUT/$tag.nml"
  sed -i "s/tridiag_kernel = .*/tridiag_kernel = '${TRIDIAG}'/" "$OUT/$tag.nml"
  sed -i "s/pcg_sync = .*/pcg_sync = 'device'/" "$OUT/$tag.nml"

  np="skip"
  if [ "$nx" -le 200 ]; then
    np=$(NX="$nx" NZ="$NZ" STEPS="$STEPS" CASE="$CASE" CFL="$CFL" "$PY" - <<'PY'
import os, sys; sys.path.insert(0, '.')
from libs.utils.config import load_config
from libs.core.driver3d import simulate3d
cfg = load_config().with_overrides({'solver.kind': 'pcg_jacobi',
                                    'grid.Lx': 4.0e5, 'grid.Ly': 4.0e5})
m = simulate3d(cfg, int(os.environ['NX']), int(os.environ['NZ']),
               float(os.environ['CFL']), os.environ['CASE'],
               n_steps_override=int(os.environ['STEPS']), n_repeat=5)['metrics']
print(f"{m['wall_s']:.6e}")
PY
)
    echo "numpy_ref,1,$nx,$NZ,$((nx*nx*NZ)),$STEPS,$np,nan,nan,0" >> "$CSV"
  fi

  OMP_NUM_THREADS=1 libs/fortran/build/cfd_exp3d_serial "$OUT/$tag.nml" >/dev/null 2>&1
  s1=$(wall "$OUT/${tag}_metrics.json"); record fortran_serial 1 "$OUT/${tag}_metrics.json" "$nx"
  for t in $THREADS_LIST; do
    OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
      libs/fortran/build/cfd_exp3d "$OUT/$tag.nml" >/dev/null 2>&1
    eval "c$t=\$(wall $OUT/${tag}_metrics.json)"
    record fortran_omp "$t" "$OUT/${tag}_metrics.json" "$nx"
  done
  libs/fortran/build/cfd_exp3d_acc "$OUT/$tag.nml" >/dev/null 2>&1
  ac=$(wall "$OUT/${tag}_metrics.json"); record openacc 0 "$OUT/${tag}_metrics.json" "$nx"
  libs/cuda/build/cfd_exp3d_cuda "$OUT/$tag.nml" >/dev/null 2>&1
  cu=$(wall "$OUT/${tag}_metrics.json"); record cuda 0 "$OUT/${tag}_metrics.json" "$nx"

  row="$nx $((nx*nx*NZ)) $np $s1"
  for t in $THREADS_LIST; do eval "row=\"\$row \$c$t\""; done
  row="$row $ac $cu"
  printf "%11s" $row; printf "\n"
done
echo; echo "csv: $CSV"
recheck_idle_node || true
