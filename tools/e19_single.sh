#!/bin/bash
# E19 reference: the plain single-device binary on the same bundles (1-card baseline without the multi driver's halo copies)
set -u
cd ${CFD_ROOT:-$HOME/cfd_exp}; export PATH=/usr/local/cuda/bin:$PATH; export PYTHONPATH=$PWD; PY=${PY:-./.venv/bin/python}
HOST=${BENCH_HOST:-gpgpu}; OUT=$(ls -d output/e19_2* | tail -1); CSV=$(ls $OUT/e19_${HOST}_*.csv | tail -1); LOG=$OUT/run.log; DEV=${DEV:-1}
for tok in DC TKE; do for NX in ${GRIDS:-400 1000 2000}; do
  pre=$OUT/${tok}_${NX}; tag=${pre}_single
  rm -f ${pre}_metrics.json ${pre}_state3d5.bin
  echo "=== $(date +%T) $tok nx=$NX single-device binary on GPU $DEV" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$DEV libs/cuda/build/cfd_exp3d5_cuda $pre.nml >> $LOG 2>&1; rc=$?
  status=ok; [ $rc -ne 0 ] && status="rc=$rc"; tail -3 $LOG | grep -qi "out of memory" && status=OOM
  [ -f ${pre}_metrics.json ] && mv ${pre}_metrics.json ${tag}_metrics.json
  [ -f ${pre}_state3d5.bin ] && mv ${pre}_state3d5.bin ${tag}_state3d5.bin
  $PY - "$HOST" "$DEV" 1 0 "$tok" "$NX" "$tag" "${pre}_nd1_state3d5.bin" "$status" >> $CSV <<'PYEOF'
import json, sys, os, numpy as np
host, devs, nd, halo, tok, nx, tag, ref, status = sys.argv[1:10]
m = json.load(open(tag + "_metrics.json")) if os.path.exists(tag + "_metrics.json") else {}
diff = "nan"
if os.path.exists(tag + "_state3d5.bin") and os.path.exists(ref):
    a = np.fromfile(tag + "_state3d5.bin"); b = np.fromfile(ref); diff = f"{float(np.abs(a-b).max()):.3e}" if a.size == b.size else "size_mismatch"
f = lambda k: m.get(k, "nan")
print(",".join(map(str, [host, m.get("gpu", "NVIDIA GeForce RTX 5090"), "single:" + devs, nd, halo, "lock_exchange" + ("_v06" if tok == "TKE" else ""), tok, "fb", nx, 30, m.get("n_steps", 50), f("wall_s"), f("wall_mad_s"), m.get("n_repeat", 5), f("device_bytes"), f("diverged"), diff, status])))
PYEOF
done; done
echo "DONE single" | tee -a $LOG
