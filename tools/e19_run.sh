#!/bin/bash
# E19 strong scaling on one node: fb, DC and DC+TKE, 400²/1000²/2000² x 30, 1 and 2 (and 4 if idle) RTX 5090.
set -u
cd ${CFD_ROOT:-$HOME/cfd_exp}; export PATH=/usr/local/cuda/bin:$PATH; export PYTHONPATH=$PWD; PY=${PY:-./.venv/bin/python}
HOST=${BENCH_HOST:-gpgpu}; STAMP=$(date +%Y%m%d-%H%M%S); OUT=output/e19_${STAMP}; mkdir -p $OUT
CSV=$OUT/e19_${HOST}_${STAMP}.csv; LOG=$OUT/run.log
echo "host,gpu,devices,n_devices,halo,case,physics,scheme,nx,nz,n_steps,wall_s,wall_mad_s,n_repeat,device_bytes,diverged,max_abs_diff_vs_1dev,status" > $CSV
DC="--set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=centered2 --set physics3d_v05.eos=teos10 --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=1e-3 --set physics3d.kappa=1e-3 --set physics3d_v05.pgf=keep --set grid.Lx=6.4e4 --set grid.Ly=6.4e3 --set physics.H=20.0"
TKE="--set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=up3_tvd --set physics3d_v06.closure=tke --set physics3d_v05.eos=teos10 --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6 --set physics3d.tau_x=0.05 --set physics3d_v05.pgf=keep --set grid.Lx=6.4e4 --set grid.Ly=6.4e3 --set physics.H=20.0"
DEVSETS=${DEVSETS:-"1 1,0"}; GRIDS=${GRIDS:-"400 1000 2000"}; HALO=${HALO:-16}
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > $OUT/nvidia_smi_before.txt
for tok in DC TKE; do sets="${!tok}"; for NX in $GRIDS; do
  pre=$OUT/${tok}_${NX}
  $PY tools/toml2nml.py --v05 $sets --set scheme.name=fb --case lock_exchange --nx $NX --nz 30 --cfl 0.5 --steps 50 --prefix $pre --n-repeat 5 --n-warmup 1 >/dev/null || { echo "bundle failed $tok $NX" | tee -a $LOG; continue; }
  ref=""
  for devs in $DEVSETS; do
    nd=$(echo $devs | tr ',' '\n' | wc -l); tag=${pre}_nd${nd}
    rm -f ${pre}_metrics.json ${pre}_state3d5.bin
    echo "=== $(date +%T) $tok nx=$NX devices=$devs" | tee -a $LOG
    libs/cuda/build/cfd_exp3d5_cuda_multi $pre.nml --ndev $nd --halo $HALO --devices $devs >> $LOG 2>&1; rc=$?
    status=ok; [ $rc -ne 0 ] && status="rc=$rc"; grep -q "out of memory" $LOG 2>/dev/null && [ $rc -ne 0 ] && status=OOM
    if [ -f ${pre}_metrics.json ]; then mv ${pre}_metrics.json ${tag}_metrics.json; fi
    if [ -f ${pre}_state3d5.bin ]; then mv ${pre}_state3d5.bin ${tag}_state3d5.bin; [ -z "$ref" ] && ref=${tag}_state3d5.bin; fi
    $PY - "$HOST" "$devs" "$nd" "$HALO" "$tok" "$NX" "$tag" "$ref" "$status" >> $CSV <<'PYEOF'
import json, sys, os, numpy as np
host, devs, nd, halo, tok, nx, tag, ref, status = sys.argv[1:10]
m = json.load(open(tag + "_metrics.json")) if os.path.exists(tag + "_metrics.json") else {}
diff = "nan"
if ref and os.path.exists(tag + "_state3d5.bin") and os.path.exists(ref):
    a = np.fromfile(tag + "_state3d5.bin"); b = np.fromfile(ref); diff = f"{float(np.abs(a-b).max()):.3e}" if a.size == b.size else "size_mismatch"
f = lambda k: m.get(k, "nan")
print(",".join(map(str, [host, m.get("gpu", "NVIDIA GeForce RTX 5090"), devs.replace(",", "+"), nd, halo, "lock_exchange" + ("_v06" if tok == "TKE" else ""), tok, "fb", nx, 30, m.get("n_steps", 50), f("wall_s"), f("wall_mad_s"), m.get("n_repeat", 5), f("device_bytes"), f("diverged"), diff, status])))
PYEOF
  done
done; done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > $OUT/nvidia_smi_after.txt
echo "DONE $CSV" | tee -a $LOG
