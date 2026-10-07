#!/usr/bin/env bash
#########################################################################
#  Module: helm_matrix.sh                                               #
#  Description: The variable-coefficient Helmholtz matrix of spec       #
#               S10.5: bathymetry roughness x solver x barotropic CFL   #
#               x grid size, on every backend built on this node.       #
#               Every backend solves the SAME binary problem, so an     #
#               iteration count that differs is a bug, not a tuning     #
#               difference (R2).                                        #
#  Pipeline: write_helmholtz_case.py -> helm_matrix.sh -> CSV -> docs   #
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
NVHPC_BIN="${NVHPC_BIN:-}"
[ -n "$NVHPC_BIN" ] && export PATH="$NVHPC_BIN:$PATH" && \
    export LD_LIBRARY_PATH="${NVHPC_BIN%/bin}/lib:${LD_LIBRARY_PATH:-}"

SIZES="${SIZES:-128 256 512}"
CFLS="${CFLS:-2 8 32 128}"
TOPOS="${TOPOS:-flat rough0.05 rough0.20 seamount}"
SOLVERS="${SOLVERS:-pcg_jacobi pcg_rbgs multigrid}"
THREADS="${THREADS:-1 16}"
LX="${LX:-4.0e5}"
# Which backends to time on this node (R7-1: a node has ONE role).
# The KT Cloud H100 container shares its host CPU with other tenants, so
# its role is GPU-only; geo85 has no GPU and is CPU-only.
BACKENDS="${BACKENDS:-cpu gpu}"
want() { case " $BACKENDS " in *" $1 "*) return 0;; esac; return 1; }

source "$(dirname "$0")/require_idle_node.sh"
# A GPU-only node need not be quiet on the CPU side.
want cpu || MAX_CPU_LOAD=1e9
require_idle_node || exit 1
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="output/helm_${STAMP}"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
# R7-2: the dispersion travels with the number. wall_min stays a secondary
# column so the old min-of-3 data set can be compared against the new one.
echo "backend,threads,nx,topo,r_std,rx0,h_ratio,cfl,solver,iterations,wall_s,wall_mad_s,wall_min_s,n_repeat,attempts,residual,l2_vs_ref" > "$CSV"

record() {  # record <backend> <threads> <json> <nx> <topo> <cfl> <case.json>
    "$PY" - "$3" "$1" "$2" "$4" "$5" "$6" "$7" "${LAST_ATTEMPTS:-1}" >> "$CSV" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
backend, threads, nx, topo, cfl, casejson = sys.argv[2:8]   # argv[8] = attempts
c = json.load(open(casejson))
print(",".join([backend, threads, nx, topo,
                f"{c['r_std']:.6f}", f"{c['rx0']:.6f}", f"{c['h_ratio']:.4f}",
                cfl, m["solver"], str(m["iterations"]),
                f"{m['wall_s']:.6e}",
                f"{m.get('wall_mad_s', float('nan')):.6e}",
                f"{m.get('wall_min_s', float('nan')):.6e}",
                str(m.get("n_repeat", m.get("repeats", ""))),
                sys.argv[8],
                f"{m['residual']:.3e}",
                f"{m['l2_rel_vs_reference']:.3e}"]))
PYEOF
}


# A run whose MAD/median exceeds MAD_MAX was contended (gpgpu is a shared node,
# R7-7) and must not enter the table: 69 of 288 rows of the 2026-09-13 pass were
# above 2 %, and EIGHT of the sixteen "best CPU" cells of docs/25 S2 sat on such
# a row. Contention only inflates, so re-running and keeping the attempt with the
# smallest dispersion converges on the uncontended time. The attempt count and
# the final MAD both travel into the CSV, so a cell that never stabilised is
# visible rather than silently averaged in.
MAD_MAX="${MAD_MAX:-0.02}"
RETRY="${RETRY:-3}"
UNSTABLE=0
LAST_ATTEMPTS=1
# A run whose MAD/median exceeds MAD_MAX was contended (gpgpu is a shared node,
# R7-7) and must not enter the table: 69 of 288 rows of the first 2026-09-13 pass
# were above 2 %, and EIGHT of the sixteen "best CPU" cells of docs/25 S2 sat on
# such a row (docs/90 N28).
#
# Which attempt to keep. Selecting the attempt with the SMALLEST DISPERSION is a
# selection on the statistic we use to judge the row, and since the median is the
# ratio's denominator it biases the time. So the rule here is the SAME one
# tools/shared_node_min.py already states and defends for the tier-2 data:
# contention can only make a run slower, never faster, so the least contaminated
# estimate is the MINIMUM OF THE PER-ATTEMPT MEDIANS. The kept attempt carries
# its own MAD and the attempt count into the CSV, so a row that never stabilised
# is visible instead of being quietly averaged in.
run_stable() {   # run_stable <metrics.json> <command...>
    local mj="$1"; shift
    local best="" best_w="" best_mad="" attempt=0 w mad
    LAST_ATTEMPTS=0
    while [ "$attempt" -lt "$RETRY" ]; do
        attempt=$((attempt + 1))
        rm -f "$mj"
        "$@" > /dev/null 2>&1
        [ -f "$mj" ] || continue
        read -r w mad < <("$PY" -c "
import json, math, sys
m = json.load(open(sys.argv[1]))
w = float(m['wall_s'])
d = float(m.get('wall_mad_s', float('nan')))
# A non-finite or absent MAD must not win any comparison, so it is reported as
# a ratio above every threshold rather than as nan (nan loses every '<' test,
# which would freeze the first attempt in place).
r = (d / w) if (w > 0.0 and math.isfinite(d) and math.isfinite(w)) else 9.9
print(w if math.isfinite(w) else -1.0, f'{r:.6f}')" "$mj" 2>/dev/null) || { w=-1; mad=9.9; }
        "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.0 else 1)" "$w" || continue
        if [ -z "$best" ] || "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$w" "$best_w"; then
            best_w="$w"; best_mad="$mad"; best=$(cat "$mj")
        fi
        # Stop as soon as ANY attempt is stable: further attempts could only
        # lower the kept median, and the stability question is already answered.
        "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$mad" "$MAD_MAX" && break
    done
    LAST_ATTEMPTS="$attempt"
    [ -n "$best" ] || return 1
    printf '%s' "$best" > "$mj"
    "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$best_mad" "$MAD_MAX" || {
        UNSTABLE=$((UNSTABLE + 1))
        echo "  !! UNSTABLE after $attempt attempt(s): MAD/median=$best_mad > $MAD_MAX  ($mj)" >&2
    }
    return 0
}

# Every backend that does NOT go through run_stable must reset the counter, or
# record() would attach the previous CPU run's attempt count to a GPU row.
run_once() {     # run_once <metrics.json> <command...>
    local mj="$1"; shift
    LAST_ATTEMPTS=1
    rm -f "$mj"
    "$@" > /dev/null 2>&1
}

echo "host=$(hostname) sizes=[$SIZES] cfls=[$CFLS] topos=[$TOPOS]"
for nx in $SIZES; do
  for topo in $TOPOS; do
    kind="${topo%%[0-9.]*}"; r="${topo#$kind}"; r="${r:-0.1}"
    for cfl in $CFLS; do
      tag="${nx}_${topo}_c${cfl}"
      dir="$OUT/$tag"
      "$PY" tools/write_helmholtz_case.py --nx "$nx" --topo "$kind" \
          --r-target "$r" --cfl "$cfl" --Lx "$LX" --out "$dir" --reference \
          --skip-numpy-solvers --n-repeat "${HELM_REPEAT:-5}" \
          > /dev/null || { echo "case $tag failed"; continue; }
      for sv in $SOLVERS; do
        # Remove the metrics file before every run: a crashed backend would
        # otherwise be recorded with the previous backend's numbers.
        for t in $(want cpu && echo $THREADS); do
          OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
            run_stable "$dir/metrics_$sv.json" \
            libs/fortran/build/helmholtz_bench "$dir/case.nml" "$sv"
          [ -f "$dir/metrics_$sv.json" ] && \
            record "fortran_omp" "$t" "$dir/metrics_$sv.json" "$nx" "$topo" "$cfl" "$dir/case.json"
        done
        if want cpu && [ -x libs/fortran/build/helmholtz_bench_serial ]; then
          run_stable "$dir/metrics_$sv.json" \
            libs/fortran/build/helmholtz_bench_serial "$dir/case.nml" "$sv"
          [ -f "$dir/metrics_$sv.json" ] && \
            record "fortran_serial" 1 "$dir/metrics_$sv.json" "$nx" "$topo" "$cfl" "$dir/case.json"
        fi
        if want gpu && [ -x libs/fortran/build/helmholtz_bench_acc ]; then
          run_once "$dir/metrics_$sv.json" \
            libs/fortran/build/helmholtz_bench_acc "$dir/case.nml" "$sv"
          [ -f "$dir/metrics_$sv.json" ] && \
            record "openacc" 0 "$dir/metrics_$sv.json" "$nx" "$topo" "$cfl" "$dir/case.json"
        fi
        if want gpu && [ -x libs/cuda/build/helmholtz_bench_cuda ]; then
          run_once "$dir/metrics_cuda_$sv.json" \
            libs/cuda/build/helmholtz_bench_cuda "$dir/case.nml" "$sv"
          [ -f "$dir/metrics_cuda_$sv.json" ] && \
            record "cuda" 0 "$dir/metrics_cuda_$sv.json" "$nx" "$topo" "$cfl" "$dir/case.json"
        fi
      done
      # Keep the CSV, drop the binaries: a full sweep is tens of GB otherwise.
      rm -f "$dir"/*.bin
    done
  done
done
echo "csv: $CSV"
recheck_idle_node || true
if [ "$UNSTABLE" -ne 0 ]; then
    echo "WARNING: $UNSTABLE configuration(s) never reached MAD <= $MAD_MAX after $RETRY attempts."
    echo "         Those rows carry their final MAD and attempt count; exclude them from any"
    echo "         'best CPU' selection (RULES.md R7-7, docs/90 N28)."
fi
