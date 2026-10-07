#!/bin/bash
#PBS -N cfd_exp_omp
#PBS -l select=1:ncpus=192
#PBS -l place=excl
#PBS -j oe
#PBS -V
#
# OpenMP thread sweep for the fortran_cpu backend on a geo85 compute node.
# RULES.md R7-7 requires exclusive node placement; -l place=excl provides it.
# geo85 has no GPU, so these numbers belong to the CPU-scalability axis only
# and must never be paired with gpgpu GPU timings (R7-1).
#
# submit:  qsub -v NX=512,CASE=igw,CFL=0.5 tools/pbs_omp_sweep.sh
set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME/cfd_exp}"
NX="${NX:-512}"
CASE="${CASE:-igw}"
CFL="${CFL:-0.5}"
THREADS="${THREADS:-1 2 4 8 16 32 64 128 192}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/omp_sweep_${CASE}_nx${NX}_${STAMP}"
mkdir -p "$OUT"

echo "host=$(hostname)  nx=${NX}  case=${CASE}  cfl=${CFL}"
lscpu | egrep "^Model name|^CPU\(s\)|^NUMA node\(s\)" || true

# One namelist per thread count so each run writes its own metrics file.
for t in ${THREADS}; do
  ./.venv/bin/python -m tools.toml2nml \
      --set solver.kind=pcg_jacobi --case "${CASE}" --nx "${NX}" --cfl "${CFL}" \
      --out "${OUT}/t${t}.nml" --prefix "${OUT}/t${t}" --n-repeat 5 --n-warmup 1 \
      > /dev/null
  echo "--- OMP_NUM_THREADS=${t} ---"
  OMP_NUM_THREADS="${t}" OMP_PROC_BIND=close OMP_PLACES=cores \
      ./libs/fortran/build/cfd_exp "${OUT}/t${t}.nml"
done

echo "results in ${OUT}"
