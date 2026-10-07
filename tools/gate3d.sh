#!/usr/bin/env bash
#########################################################################
#  Module: gate3d.sh                                                    #
#  Description: Run the R2 3D verification gate for every verification  #
#               case against every compiled backend present on this     #
#               node. One place holds the per-case configuration        #
#               overrides so a gate run is identical on every bench     #
#               node (RULES.md R2, R6).                                #
#  Pipeline: build -> gate3d.sh -> bench3d_matrix.sh                    #
#########################################################################
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.

NX=${NX:-32}
NZ=${NZ:-30}
STEPS=${STEPS:-20}

# Per-case configuration overrides. Every case needs the compiled backends'
# only elliptic solver (pcg_jacobi); the physics overrides are the conditions
# each analytic solution is derived under (docs/03 S7.6).
case_sets() {
  common="--set solver.kind=pcg_jacobi"
  case "$1" in
    barotropic3d)  echo "$common --set physics3d.N2=0.0 --set physics3d.nu=0.0 --set physics3d.kappa=0.0" ;;
    vdiffusion)    echo "$common --set physics3d.N2=0.0 --set physics3d.nu=1.0e-3" ;;
    baroclinic_igw) echo "$common --set physics3d.N2=1.0e-4 --set physics3d.nu=0.0 --set physics3d.kappa=0.0" ;;
    tracer_advect) echo "$common --set physics3d_v04.tracers=TS --set physics3d_v04.alpha_T=0.0 --set physics3d_v04.beta_S=0.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=0.0 --set physics3d.kappa=0.0 --set scheme.advection=centered2" ;;
    ekman)         echo "$common --set physics3d.N2=0.0 --set physics3d.nu=2.0 --set physics3d.tau_x=0.1 --set physics3d.kappa=0.0" ;;
  esac
}

CASES=${CASES:-"barotropic3d vdiffusion baroclinic_igw tracer_advect ekman"}
BINARIES=${BINARIES:-"libs/fortran/build/cfd_exp3d libs/fortran/build/cfd_exp3d_serial libs/fortran/build/cfd_exp3d_acc libs/cuda/build/cfd_exp3d_cuda"}

pass=0; fail=0; skip=0
printf '%-16s %-22s %s\n' case backend verdict
for c in $CASES; do
  sets=$(case_sets "$c")
  for b in $BINARIES; do
    [ -x "$b" ] || { skip=$((skip+1)); continue; }
    out=$(python3 tools/compare_backends3d.py --case "$c" --nx "$NX" --nz "$NZ" \
            --steps "$STEPS" --binary "$b" $sets 2>&1)
    if echo "$out" | grep -q "R2 GATE (3D): PASS"; then
      v=PASS; pass=$((pass+1))
    else
      v="FAIL: $(echo "$out" | grep -m1 -E 'FAIL|Error|error|rel' | cut -c1-90)"
      fail=$((fail+1))
    fi
    printf '%-16s %-22s %s\n' "$c" "$(basename "$b")" "$v"
  done
done
echo "gate3d: $pass pass, $fail fail, $skip backend(s) not built on this node"
[ "$fail" -eq 0 ]
