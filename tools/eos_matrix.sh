#!/usr/bin/env bash
#########################################################################
#  Module: eos_matrix.sh                                                #
#  Description: Cost of the equation of state (spec S10.6, RQ7) on      #
#               every backend built on this node. The EOS is the one    #
#               compute-bound kernel in a hydrostatic core, so this is  #
#               where the GPU-to-CPU ratio should be widest - the       #
#               opposite of every memory-bound kernel measured so far.  #
#  Pipeline: eos_bench -> eos_matrix.sh -> CSV -> docs/25               #
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
NVHPC_BIN="${NVHPC_BIN:-}"
[ -n "$NVHPC_BIN" ] && export PATH="$NVHPC_BIN:$PATH" && \
    export LD_LIBRARY_PATH="${NVHPC_BIN%/bin}/lib:${LD_LIBRARY_PATH:-}"
BACKENDS="${BACKENDS:-cpu gpu}"
want() { case " $BACKENDS " in *" $1 "*) return 0;; esac; return 1; }
source "$(dirname "$0")/require_idle_node.sh"
want cpu || MAX_CPU_LOAD=1e9
require_idle_node || exit 1

N="${N:-3000000}"
REPEAT="${REPEAT:-20}"
KINDS="${KINDS:-linear seos teos10}"
THREADS="${THREADS:-1 16}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/eos_${STAMP}"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
# R7-2: the bench prints median and MAD; both go into the CSV.
echo "backend,threads,n,eos,wall_s,wall_mad_s,n_repeat,checksum" > "$CSV"

# flop/cell counted from the source expressions (libs/core/eos.py).
EMIT_FAILED=0
emit() {  # emit <backend> <threads> <line>
    # A sed s/// that does not match returns the WHOLE line, so an absent field
    # (an older binary with no `mad=`) or a non-numeric one (`nan`, `Infinity`)
    # used to be written into the CSV as prose instead of failing. Parse the
    # line in one pass and refuse it if any field is missing (docs/90 N27).
    "${PY:-python3}" - "$1" "$2" "$REPEAT" "$3" >> "$CSV" <<'PYEOF'
import re, sys
backend, threads, repeat, line = sys.argv[1:5]
def grab(name, pattern):
    m = re.search(rf"(?:^|\s){name}=\s*({pattern})(?:\s|$)", line)
    if not m:
        sys.stderr.write(f"eos_matrix: field {name!r} not found in: {line!r}\n")
        raise SystemExit(1)
    return m.group(1)
num = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
row = [backend, threads, grab("n", r"\d+"), grab("eos", r"[A-Za-z0-9_.-]+"),
       grab("wall", num), grab("mad", num), repeat, grab("checksum", num)]
print(",".join(row))
PYEOF
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        # A refused row is a LOST measurement. Before the N27 fix a malformed
        # line was written as prose; now it is dropped - equally silent unless
        # we count it here and fail the run at the end.
        EMIT_FAILED=$((EMIT_FAILED + 1))
        echo "  !! REFUSED row: backend=$1 threads=$2 line=$3" >&2
    fi
    return $rc
}

echo "host=$(hostname) n=$N repeat=$REPEAT"
for k in $KINDS; do
  if want cpu; then
    for t in $THREADS; do
      line=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
             libs/fortran/build/eos_bench "$N" "$k" "$REPEAT" 2>/dev/null)
      [ -n "$line" ] && { echo "cpu t=$t  $line"; emit fortran_omp "$t" "$line"; }
    done
  fi
  if want gpu && [ -x libs/fortran/build/eos_bench_acc ]; then
    line=$(libs/fortran/build/eos_bench_acc "$N" "$k" "$REPEAT" 2>/dev/null)
    [ -n "$line" ] && { echo "acc      $line"; emit openacc 0 "$line"; }
  fi
  if want gpu && [ -x libs/cuda/build/eos_bench_cuda ]; then
    line=$(libs/cuda/build/eos_bench_cuda "$N" "$k" "$REPEAT" 2>/dev/null)
    [ -n "$line" ] && { echo "cuda     $line"; emit cuda 0 "$line"; }
  fi
done
echo "csv: $CSV"
if [ "$EMIT_FAILED" -ne 0 ]; then
    echo "FATAL: $EMIT_FAILED benchmark line(s) could not be parsed - the CSV is" >&2
    echo "       incomplete and must not be used (RULES.md R6)." >&2
    exit 1
fi
