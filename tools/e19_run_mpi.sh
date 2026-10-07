#!/bin/bash
# E19 multi-node: MPI driver over the bundles of the latest e19 run directory (one GPU per node).
# Records host-staged and CUDA-aware (--device-mpi) exchanges; each run's state is compared with the 1-device state.
set -u
cd ${CFD_ROOT:-$HOME/cfd_exp}; export PATH=/usr/local/cuda/bin:$PATH; export PYTHONPATH=$PWD; PY=${PY:-python3}
HOST=${BENCH_HOST:-ktcloud}; HOSTS=${HOSTS:-main1,sub1}; NP=$(echo $HOSTS | tr ',' '\n' | wc -l); HALO=${HALO:-16}
OUT=${OUT:-$(ls -d output/e19_2* | grep -v CONTAM | tail -1)}; CSV=$(ls $OUT/e19_${HOST}_*.csv | tail -1); LOG=$OUT/run.log
MPIRUN="mpirun --allow-run-as-root -np $NP -H $HOSTS -wdir $PWD -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH"
for tok in ${TOKS:-DC TKE}; do for NX in ${GRIDS:-400 1000 2000}; do
  pre=$OUT/${tok}_${NX}; [ -f $pre.nml ] || { echo "no bundle $pre.nml" | tee -a $LOG; continue; }
  for mode in ${MODES:-host device}; do
    flag=""; [ $mode = device ] && flag="--device-mpi"; tag=${pre}_mpi${NP}_${mode}
    rm -f ${pre}_metrics.json ${pre}_state3d5.bin
    echo "=== $(date +%T) $tok nx=$NX mpi np=$NP hosts=$HOSTS mode=$mode" | tee -a $LOG
    $MPIRUN libs/cuda/build/cfd_exp3d5_cuda_mpi $pre.nml --halo $HALO $flag >> $LOG 2>&1; rc=$?
    status=ok; [ $rc -ne 0 ] && status="rc=$rc"
    [ -f ${pre}_metrics.json ] && mv ${pre}_metrics.json ${tag}_metrics.json
    [ -f ${pre}_state3d5.bin ] && mv ${pre}_state3d5.bin ${tag}_state3d5.bin
    ref=${pre}_nd1_state3d5.bin; [ -f $ref ] || ref=${pre}_single_state3d5.bin
    $PY - "$HOST" "mpi:$(echo $HOSTS | tr , +):$mode" "$NP" "$HALO" "$tok" "$NX" "$tag" "$ref" "$status" >> $CSV <<'PYEOF'
import json, sys, os, numpy as np
host, devs, nd, halo, tok, nx, tag, ref, status = sys.argv[1:10]
m = json.load(open(tag + "_metrics.json")) if os.path.exists(tag + "_metrics.json") else {}
diff = "nan"
if os.path.exists(tag + "_state3d5.bin") and os.path.exists(ref):
    a = np.fromfile(tag + "_state3d5.bin"); b = np.fromfile(ref); diff = f"{float(np.abs(a-b).max()):.3e}" if a.size == b.size else "size_mismatch"
f = lambda k: m.get(k, "nan")
print(",".join(map(str, [host, m.get("gpu", "NVIDIA H100 80GB HBM3"), devs, nd, halo, "lock_exchange" + ("_v06" if tok == "TKE" else ""), tok, "fb", nx, 30, f("n_steps"), f("wall_s"), f("wall_mad_s"), f("n_repeat"), f("device_bytes"), f("diverged"), diff, status])))
PYEOF
  done
done; done
echo "DONE mpi" | tee -a $LOG
