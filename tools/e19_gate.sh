#!/bin/bash
# E19 gate: multi-device state must be bit-identical to the single-device state (docs/03 S12)
set -u
cd ${CFD_ROOT:-$HOME/cfd_exp}; export PATH=/usr/local/cuda/bin:$PATH; export PYTHONPATH=$PWD; PY=${PY:-./.venv/bin/python}
OUT=output/e19_gate; mkdir -p $OUT
DC="--set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=centered2 --set physics3d_v05.eos=teos10 --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=1e-3 --set physics3d.kappa=1e-3 --set physics3d_v05.pgf=keep --set grid.Lx=6.4e4 --set grid.Ly=6.4e3 --set physics.H=20.0"
TKE="--set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=up3_tvd --set physics3d_v06.closure=tke --set physics3d_v05.eos=teos10 --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6 --set physics3d.tau_x=0.05 --set physics3d_v05.pgf=keep --set grid.Lx=6.4e4 --set grid.Ly=6.4e3 --set physics.H=20.0"
NX=${NX:-400}; STEPS=${STEPS:-50}; DEVS=${DEVS:-1,0}
for tok in DC TKE; do
  sets="${!tok}"; pre=$OUT/${tok}_${NX}
  $PY tools/toml2nml.py --v05 $sets --set scheme.name=fb --case lock_exchange --nx $NX --nz 30 --cfl 0.5 --steps $STEPS --prefix $pre --n-repeat 1 --n-warmup 0 >/dev/null || { echo "bundle failed $tok"; exit 1; }
  CUDA_VISIBLE_DEVICES=${DEVS%%,*} libs/cuda/build/cfd_exp3d5_cuda $pre.nml | grep wall; mv ${pre}_state3d5.bin ${pre}_single.bin
  for nd in ${NDS:-1 2}; do for H in 8 16; do
    [ $nd = 1 ] && [ $H = 8 ] && continue
    libs/cuda/build/cfd_exp3d5_cuda_multi $pre.nml --ndev $nd --halo $H --devices ${DEVS} 2>&1 | grep -E "wall|FATAL" ; mv ${pre}_state3d5.bin ${pre}_nd${nd}_h${H}.bin
  done; done
  $PY - "$pre" <<'PYEOF'
import numpy as np, sys, glob
pre=sys.argv[1]; ref=np.fromfile(pre+"_single.bin")
for f in sorted(glob.glob(pre+"_nd*_h*.bin")):
    a=np.fromfile(f); d=np.abs(a-ref); print(f"  {f.split('/')[-1]:28s} n={a.size} max|diff|={d.max():.3e} n_diff={(d>0).sum()} finite={np.isfinite(a).all()}")
PYEOF
done
