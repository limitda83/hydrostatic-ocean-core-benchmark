#!/usr/bin/env bash
#########################################################################
#  Module: gate3d5.sh                                                   #
#  Description: R2 gate for the spec v0.5 backends on this node: a      #
#               fixed set of physics x scheme x solver combinations,    #
#               each compared with the NumPy reference on a byte-       #
#               identical problem. Same combinations on every node.     #
#  Pipeline: build -> gate3d5.sh -> tier-2 sweeps                       #
#########################################################################
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
PY="${PY:-python3}"
NX=${NX:-32}; NZ=${NZ:-10}; STEPS=${STEPS:-10}
BINARIES=${BINARIES:-"libs/fortran/build/cfd_exp3d5_serial libs/fortran/build/cfd_exp3d5 libs/fortran/build/cfd_exp3d5_acc libs/cuda/build/cfd_exp3d5_cuda"}
BASE=(--set grid.Lx=4.0e5 --set grid.Ly=4.0e5 --set physics3d.N2=0.0 --set physics3d_v05.pgf=keep --set physics3d.nu=1e-3 --set physics3d.kappa=1e-3)

declare -a NAMES ARGS
add() { NAMES+=("$1"); ARGS+=("$2"); }
add "flat.theta.pcg"        "--case seamount_rest --set bathymetry.kind=flat --set solver.kind=pcg_jacobi"
add "seamount.theta.mg.closed" "--case seamount_rest --set bathymetry.kind=seamount --set solver.kind=multigrid --set domain.bc_x=closed --set domain.bc_y=closed"
add "seamount.split.closed"  "--case seamount_rest --set bathymetry.kind=seamount --set scheme.name=split_explicit --set domain.bc_x=closed"
add "seamount.theta1.rbgs"   "--case seamount_rest --set bathymetry.kind=seamount --set scheme.theta=1.0 --set solver.kind=pcg_rbgs"
add "rough.TS.teos10.mg"     "--case lock_exchange --set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=centered2 --set physics3d_v05.eos=teos10 --set solver.kind=multigrid --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0"
add "rough.TS.fb.upwind"     "--case lock_exchange --set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=upwind1 --set scheme.name=fb --set physics.f0=0.0"
add "v06.rough.TS.up3.mg"    "--case lock_exchange --set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=up3 --set physics3d_v05.eos=teos10 --set solver.kind=multigrid --set physics.f0=0.0 --set domain.bc_x=closed"
add "v06.seamount.tvd.tke.theta" "--case lock_exchange --set bathymetry.kind=seamount --set physics3d_v04.tracers=TS --set scheme.advection=up3_tvd --set physics3d_v06.closure=tke --set physics3d.tau_x=0.1 --set solver.kind=multigrid --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6"
add "v06.seamount.tke.split"  "--case lock_exchange --set bathymetry.kind=seamount --set physics3d_v04.tracers=TS --set scheme.advection=up3 --set physics3d_v06.closure=tke --set scheme.name=split_explicit --set physics3d.tau_x=0.1 --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6 --set domain.bc_x=closed"
add "v06.mxl_rec.tke.theta"  "--case lock_exchange --set bathymetry.kind=seamount --set physics3d_v04.tracers=TS --set scheme.advection=up3_tvd --set physics3d_v06.closure=tke --set physics3d_v06.mxl=recursive --set physics3d.tau_x=0.1 --set solver.kind=multigrid --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6"
add "v06.mxl_rec.tke.fb"     "--case lock_exchange --set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=up3 --set physics3d_v06.closure=tke --set physics3d_v06.mxl=recursive --set scheme.name=fb --set physics3d.tau_x=0.1 --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6 --set physics.f0=0.0"
add "seamount.split.seos.forcing" "--case lock_exchange --set bathymetry.kind=seamount --set physics3d_v04.tracers=TS --set scheme.advection=centered2 --set scheme.name=split_explicit --set physics3d_v05.eos=seos --set physics3d.tau_x=0.1 --set physics3d_v04.q_heat=100.0 --set physics3d.bottom_drag=1e-3 --set domain.bc_x=closed"

pass=0; fail=0; skip=0
printf '%-30s %-22s %s\n' case backend verdict
for b in $BINARIES; do
  [ -x "$b" ] || { skip=$((skip+1)); continue; }
  for i in "${!NAMES[@]}"; do
    # shellcheck disable=SC2086
    out=$("$PY" tools/compare_backends3d5.py --nx "$NX" --nz "$NZ" --steps "$STEPS" --binary "$b" \
            "${BASE[@]}" ${ARGS[$i]} 2>&1)
    if echo "$out" | grep -q "R2 GATE (3D v0.5): PASS"; then
      v="PASS $(echo "$out" | grep '^full run' | sed -E 's/.*rel L2: //; s/ +tol.*//')"; pass=$((pass+1))
    else
      v="FAIL: $(echo "$out" | grep -m1 -E 'FAIL|Error|error|FATAL' | cut -c1-100)"; fail=$((fail+1))
    fi
    printf '%-30s %-22s %s\n' "${NAMES[$i]}" "$(basename "$b")" "$v"
  done
done
echo "gate3d5: $pass pass, $fail fail, $skip backend(s) not built on this node"
if [ "$pass" -eq 0 ]; then
  # Every backend missing (ktcloud wipes /home/work between sessions) used to
  # leave pass=0 fail=0, and `[ 0 -eq 0 ]` reported R2 as satisfied by running
  # nothing at all (docs/90 N34).
  echo "FATAL: the gate executed 0 comparisons - nothing was verified." >&2
  exit 1
fi
[ "$fail" -eq 0 ]
