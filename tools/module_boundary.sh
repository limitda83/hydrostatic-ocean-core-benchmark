#!/usr/bin/env bash
#########################################################################
#  Module: module_boundary.sh                                           #
#  Description: E17 driver - runs libs/cuda/build/module_boundary over  #
#               the grid x precision matrix and concatenates the CSV.   #
#               Grids and precisions match the physics decomposition    #
#               (docs/31 S3b) so the two join on (host, precision, nx). #
#  Pipeline: module_boundary.sh -> expr/E17/data -> module_boundary_report#
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BIN:-libs/cuda/build/module_boundary}"
NXS="${NXS:-100 400 1000}"
NZ="${NZ:-30}"
REPEAT="${REPEAT:-5}"
STATIC="${STATIC:-}"                 # set to 1 to send dz3/mask3/... every step
OUT="${OUT:-output/module_boundary_$(hostname -s).csv}"
[ -x "$BIN" ] || { echo "FATAL: $BIN not built (cd libs/cuda && make boundary)" >&2; exit 1; }
flag=""; [ -n "$STATIC" ] && flag="--static-every-step"
first=1
: > "$OUT"
for its in 8 4; do
  for nx in $NXS; do
    out=$("$BIN" --nx "$nx" --nz "$NZ" --itemsize "$its" --n-repeat "$REPEAT" $flag) || {
      echo "FAILED nx=$nx itemsize=$its" >&2; continue; }
    if [ "$first" = "1" ]; then echo "$out" >> "$OUT"; first=0
    else echo "$out" | tail -n +2 >> "$OUT"; fi
    echo "done nx=$nx itemsize=$its"
  done
done
echo "csv: $OUT"
