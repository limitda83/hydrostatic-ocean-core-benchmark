#!/bin/bash
#PBS -N cfd3d_omp
#PBS -l select=1:ncpus=192
#PBS -l place=excl
#PBS -j oe
#PBS -V
#
# 3D OpenMP thread sweep on a geo85 compute node (192 cores, exclusive).
# geo85 has no GPU, so these belong to the CPU-scalability axis only and must
# never be paired with gpgpu GPU timings (RULES.md R7-1).
#
# submit: qsub -v NX=200,NZ=30 tools/pbs_omp3d_sweep.sh
set -euo pipefail
cd "${PBS_O_WORKDIR:-$HOME/cfd_exp}"
NX="${NX:-200}"; NZ="${NZ:-30}"; STEPS="${STEPS:-20}"; CFL="${CFL:-2.0}"
KERNELS="${KERNELS:-plane column}"
THREADS="${THREADS:-1 2 4 8 16 32 64 128 192}"
OUT="output/omp3d_nx${NX}_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"

echo "host=$(hostname) nx=${NX} nz=${NZ} steps=${STEPS} cfl=${CFL}"
lscpu | egrep "^Model name|^CPU\(s\)|^NUMA node\(s\)" || true

for kern in ${KERNELS}; do
  echo; echo "=== tridiag_kernel=${kern} ==="
  printf "%8s %12s %12s\n" threads "wall [s]" "speedup"
  base=""
  for t in ${THREADS}; do
    ./.venv/bin/python -m tools.toml2nml --set solver.kind=pcg_jacobi \
        --set grid.Lx=4.0e5 --set grid.Ly=4.0e5 --case baroclinic_igw \
        --nx "$NX" --nz "$NZ" --cfl "$CFL" --steps "$STEPS" \
        --out "$OUT/${kern}_t${t}.nml" --prefix "$OUT/${kern}_t${t}" \
        --n-repeat 5 --n-warmup 1 > /dev/null
    sed -i "s/omp_min_points = .*/omp_min_points = 0/" "$OUT/${kern}_t${t}.nml"
    sed -i "s/tridiag_kernel = .*/tridiag_kernel = '${kern}'/" "$OUT/${kern}_t${t}.nml"
    OMP_NUM_THREADS="$t" OMP_PROC_BIND=close OMP_PLACES=cores \
      ./libs/fortran/build/cfd_exp3d "$OUT/${kern}_t${t}.nml" > /dev/null 2>&1
    w=$(grep -o '"wall_s":[^,]*' "$OUT/${kern}_t${t}_metrics.json" | sed 's/.*: *//')
    [ -z "$base" ] && base="$w"
    printf "%8s %12s %11s\n" "$t" "$w" \
      "$(awk -v b="$base" -v x="$w" 'BEGIN{printf "%.2fx", b/x}')"
  done
done
echo; echo "results in ${OUT}"
