#!/usr/bin/env bash
#########################################################################
#  Module: gpu_energy.sh                                                #
#  Description: Energy to solution for a GPU run. Time-to-solution      #
#               (R8) answers "how long"; a model centre's real          #
#               constraint is a power budget, so "how many joules"      #
#               is the second half of the same question - and the two   #
#               do not always rank configurations the same way.         #
#                                                                       #
#  usage: tools/gpu_energy.sh <label> <command ...>
#  NOTE: the command's whole process is integrated, so its timed section must
#  dominate its wall time. A run whose kernel is 40% of the process reports
#  60% startup and file I/O as kernel energy.                     #
#  Samples power.draw at 100 ms while the command runs and integrates.  #
#  Idle power is measured first and reported separately, because a      #
#  short kernel spends most of its wall time at idle draw and charging  #
#  that to the kernel would flatter long runs.                          #
#########################################################################
set -uo pipefail
LABEL="${1:?usage: gpu_energy.sh <label> <command ...>}"; shift
GPU="${CUDA_VISIBLE_DEVICES:-0}"
INTERVAL="${INTERVAL:-0.1}"
OUT="${OUT:-output/energy}"; mkdir -p "$OUT"
SAMPLES="$OUT/${LABEL}_power.csv"

# Idle draw as the MEDIAN of a three-second sample, not a single reading.
# A single reading taken just after another run catches the card still
# boosted: successive single samples on the same idle GPU read 141 W and
# 226 W, which made one kernel's energy-above-idle come out negative.
idle=$(nvidia-smi -i "$GPU" --query-gpu=power.draw \
         --format=csv,noheader,nounits -lms 100 2>/dev/null \
       | head -30 | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')

nvidia-smi -i "$GPU" --query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm \
    --format=csv,noheader,nounits -lms "$(awk -v i="$INTERVAL" 'BEGIN{print int(i*1000)}')" \
    > "$SAMPLES" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

start=$(date +%s.%N)
"$@"
rc=$?
end=$(date +%s.%N)
sleep 0.3
kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null

awk -F', ' -v lbl="$LABEL" -v idle="$idle" -v dt="$INTERVAL" \
    -v t0="$start" -v t1="$end" '
    $2 ~ /^[0-9.]+$/ { n++; sum += $2; if ($2 > peak) peak = $2 }
    END {
        wall = t1 - t0;
        if (n == 0) { print "no power samples"; exit }
        mean = sum / n;
        printf "%s wall=%.4f s  mean_power=%.1f W  peak=%.1f W  idle=%.1f W\n",
               lbl, wall, mean, peak, idle;
        printf "%s energy_total=%.2f J  energy_above_idle=%.2f J  samples=%d\n",
               lbl, mean * wall, (mean - idle) * wall, n;
    }' "$SAMPLES"
exit $rc
