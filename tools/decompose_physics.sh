#!/usr/bin/env bash
#########################################################################
#  Module: decompose_physics.sh                                         #
#  Description: Splits the cost of the v0.6 "representative physics"    #
#               into its two parts on one device: the TKE closure        #
#               (column-local non-linear algebra + a per-step rebuild    #
#               of every coefficient) and the third-order advection      #
#               (wider stencil, and the TVD limiter's branches). Five    #
#               variants of the SAME case and grid, 50 fixed steps, so   #
#               the differences are cost only (docs/33 finding 2).       #
#  Pipeline: decompose_physics.sh -> tier2_sweep (TAG per variant) -> CSV#
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
NX="${NX:-400}"; NZ="${NZ:-30}"; STEPS="${STEPS:-50}"; REPEAT="${REPEAT:-5}"
BACKENDS="${BACKENDS:-gpu}"; SCHEMES="${SCHEMES:-theta0.5}"; SOLVERS="${SOLVERS:-multigrid}"
CFLS="${CFLS:-4}"
TKE_ON="--set physics3d_v06.closure=tke --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6 --set physics3d.tau_x=0.05"
run () {                       # run <tag> <extra --set flags>
  TAG="$1" EXTRA_SET="$2" NX=$NX NZ=$NZ STEPS=$STEPS REPEAT=$REPEAT BACKENDS="$BACKENDS" \
    CASES=lock_exchange SCHEMES="$SCHEMES" SOLVERS="$SOLVERS" CFLS="$CFLS" CFLS_FB=0.5 \
    bash tools/tier2_sweep.sh
}
run ".v05"          "--set scheme.advection=centered2 --set physics3d_v06.closure=none"
run ".up3"          "--set scheme.advection=up3 --set physics3d_v06.closure=none"
run ".up3tvd"       "--set scheme.advection=up3_tvd --set physics3d_v06.closure=none"
run ".tke"          "--set scheme.advection=centered2 $TKE_ON"
run ".tke_up3tvd"   "--set scheme.advection=up3_tvd $TKE_ON"
echo "=== decomposition done $(date -u +%FT%TZ)"
