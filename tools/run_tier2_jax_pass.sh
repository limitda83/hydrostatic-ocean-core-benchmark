#!/usr/bin/env bash
#########################################################################
#  Module: run_tier2_jax_pass.sh                                        #
#  Description: Second pass of a node's tier-2 orchestration for the    #
#               Python-GPU level: waits until the compiled-backend pass  #
#               has released the node, gates the JAX backend (R4: no     #
#               timing before the R2 gate on this device), then runs the #
#               node script for piece H only (SKIP_S=1 BACKENDS_H=jax):   #
#               the error of a configuration is backend-independent (R2), #
#               so JAX needs cost per step, not a second Pareto plane.    #
#  Pipeline: nohup tools/run_tier2_jax_pass.sh <gpgpu|ktcloud>          #
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
node="$1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
case "$node" in
  gpgpu)   export JAX_PY=./.venv-jax/bin/python PY=./.venv/bin/python
           # GPU 0 on this node is shared and usually busy; pin the idle one (R7-7, N22)
           export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" ;;
  ktcloud) export JAX_PY=/home/work/cello/.venv/bin/python PY=python3 ;;
esac
# anchored: a waiter whose command line merely mentions the script must not hold us (N18)
while pgrep -f "^bash tools/run_tier2_${node}.sh" >/dev/null; do sleep 60; done
echo "=== JAX gate on $node  $(date -u +%FT%TZ)"
gate=$(BINARIES=libs/jax/build/cfd_exp3d5_jax bash tools/gate3d5.sh 2>&1); echo "$gate" | tail -9
echo "$gate" | grep -q " 0 fail" || { echo "JAX gate FAILED on $node - no timings are taken (R4)"; exit 1; }
echo "=== JAX pass  $(date -u +%FT%TZ)"
SKIP_S=1 BACKENDS_H=jax bash "tools/run_tier2_${node}.sh"
