#!/bin/bash
#PBS -N cfd_tier2
#PBS -l select=1:ncpus=192
#PBS -l place=excl
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -V
#########################################################################
#  Module: pbs_tier2.sh                                                 #
#  Description: geo85 wrapper for the tier-2 sweep (docs/04 S6 step 6). #
#               One exclusive EPYC 9655 node (R7-7), CPU backends only  #
#               (R7-1: geo85 is the CPU representative). Every knob of  #
#               tier2_sweep.sh passes through qsub -v.                   #
#  Pipeline: qsub -v NX=400,STEPS=50 tools/pbs_tier2.sh -> tier2_sweep  #
#########################################################################
set -uo pipefail
cd "${PBS_O_WORKDIR:-$HOME/cfd_exp}"
echo "host=$(hostname) job=${PBS_JOBID:-none} start=$(date -u +%FT%TZ)"
lscpu | egrep "^Model name|^CPU\(s\)|^NUMA node\(s\)" || true
export BACKENDS=cpu PY=./.venv/bin/python
export THREADS="${THREADS:-1 8 32 64 128 192}"
export MAX_CPU_LOAD="${MAX_CPU_LOAD:-8}"      # an exclusive node is idle
# The node is ours alone, but the load average still carries the tail of the
# job that ran here a minute ago (three jobs died on load 130 at start, N13).
# Wait for it to decay instead of failing the guard.
settled=0
for _ in $(seq 1 40); do
  load=$(awk '{print $1}' /proc/loadavg)
  awk -v l="$load" -v m="$MAX_CPU_LOAD" 'BEGIN{exit !(l <= m)}' && { settled=1; break; }
  echo "load $load > $MAX_CPU_LOAD - waiting for the previous job's tail to decay"; sleep 30
done
if [ "$settled" -ne 1 ]; then
  # The old code fell through and measured anyway. On 2026-09-13 that produced a
  # sweep where 96 threads was SLOWER than 1 thread (100.7 s vs 27.8 s at 400^2)
  # because the node never became ours - 20 minutes of "load 90.00" then silence.
  # A contaminated number that looks plausible is worse than no number (R7-7).
  echo "FATAL: load stayed at $load (> $MAX_CPU_LOAD) for 20 minutes." >&2
  echo "       This node is not exclusively ours; refusing to measure." >&2
  exit 1
fi
# From here on the load IS this job (192 threads); the end-of-run recheck must
# not read our own threads as an intruder (it warned "load rose to 177" on a
# node nobody else could enter). Exclusive placement is the real guard.
export MAX_CPU_LOAD=250
bash tools/tier2_sweep.sh
echo "end=$(date -u +%FT%TZ)"
