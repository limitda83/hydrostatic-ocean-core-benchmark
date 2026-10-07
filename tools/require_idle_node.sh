#!/bin/bash
# Guard for RULES.md R7-7: refuse to produce timings on a contended node.
#
# gpgpu is shared. A run of docs/21's PCG experiment was silently corrupted by
# another user's job appearing on the target GPU mid-sweep - the numbers came
# out non-monotonic in CFL, which is physically impossible, and had to be
# discarded. Checking up front is cheaper than discovering it afterwards.
#
# usage: source tools/require_idle_node.sh   (uses CUDA_VISIBLE_DEVICES)
#        REQUIRE_IDLE=0 disables the check for a correctness-only run.

# A lock so a node never runs two of our own sweeps at once. The GPU check
# below asks whether SOMEONE ELSE is using the card; it cannot see our own
# earlier job, whose kernels peak at 23 MiB - under the 500 MiB threshold.
# Two concurrent sweeps on one H100 inflated each other's times twentyfold
# at identical iteration counts (docs/90 N9).
require_bench_lock() {
    [ "${REQUIRE_IDLE:-1}" = "0" ] && return 0
    # A fixed path, NOT $TMPDIR: two sweeps launched from shells with different
    # TMPDIR took different locks and ran side by side on one node (N22).
    local lock="${BENCH_LOCK:-$HOME/.cfd_exp_bench.lock}"
    if [ -e "$lock" ]; then
        local pid
        pid=$(cat "$lock" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "FATAL: another cfd_exp benchmark is already running (pid $pid)." >&2
            echo "       Two sweeps sharing one GPU inflate each other's timings" >&2
            echo "       at identical iteration counts (docs/90 N9)." >&2
            return 1
        fi
        echo "note: removing a stale bench lock from pid ${pid:-unknown}" >&2
        rm -f "$lock"
    fi
    echo $$ > "$lock"
    # shellcheck disable=SC2064
    trap "rm -f '$lock'" EXIT
    echo "bench lock taken: $lock (pid $$)"
}

# Median of three samples of (memory MiB, util %) for $CUDA_VISIBLE_DEVICES.
gpu_sample() {
    local gpu="${CUDA_VISIBLE_DEVICES:-0}" line k
    local -a mems=() utils=()
    if [ "${REQUIRE_GPU:-1}" = "0" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0 -1"; return 0
    fi
    for k in 1 2 3; do
        line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                          --format=csv,noheader,nounits | awk -F', ' -v g="$gpu" '$1==g')
        [ -z "$line" ] && { echo "FATAL: GPU $gpu not found" >&2; return 1; }
        mems+=("$(echo "$line" | awk -F', ' '{print $2}')")
        utils+=("$(echo "$line" | awk -F', ' '{print $3}')")
        [ "$k" -lt 3 ] && sleep 1
    done
    local mem util
    mem=$(printf '%s\n' "${mems[@]}" | sort -n | sed -n 2p)
    util=$(printf '%s\n' "${utils[@]}" | sort -n | sed -n 2p)
    case "$util" in ''|*[!0-9]*) util=-1 ;; esac
    echo "$mem $util"
}

# Wait until the GPU is free (another tenant's job, or the cello keepalive that
# preallocates 75% of the card, must not make a whole sweep exit - N21).
wait_for_gpu() {
    local limit="${WAIT_GPU_MIN:-0}" waited=0 mem util
    [ "${REQUIRE_GPU:-1}" = "0" ] && return 0
    while [ "$waited" -lt "$limit" ]; do
        read -r mem util <<<"$(gpu_sample)" || return 1
        if [ "$mem" -le "${MAX_GPU_MEM_MIB:-500}" ]; then return 0; fi
        echo "GPU busy (${mem} MiB, ${util}% util) - waiting ($waited/$limit min)" >&2
        sleep 60; waited=$((waited + 1))
    done
    return 0
}

# The load average still carries the tail of OUR OWN previous OpenMP run for
# minutes after it exits (exponential average). A ladder that starts one sweep
# per grid must wait for it to decay, not abort (N13 on geo85, N23 on gpgpu).
wait_for_load() {
    local limit="${WAIT_LOAD_MIN:-0}" waited=0 load max
    local ncpu; ncpu=$(nproc 2>/dev/null || echo 8)
    max="${MAX_CPU_LOAD:-$(awk -v n="$ncpu" 'BEGIN{printf "%.1f", (n/4.0 > 4.0 ? n/4.0 : 4.0)}')}"
    while [ "$waited" -lt "$limit" ]; do
        load=$(awk '{print $1}' /proc/loadavg)
        awk -v l="$load" -v m="$max" 'BEGIN{exit !(l <= m)}' && return 0
        echo "load $load > $max - waiting for the previous run's tail ($waited/$limit min)" >&2
        sleep 60; waited=$((waited + 1))
    done
    return 0
}

require_idle_node() {
    [ "${REQUIRE_IDLE:-1}" = "0" ] && return 0
    require_bench_lock || return 1
    wait_for_gpu || return 1
    wait_for_load || return 1
    local gpu="${CUDA_VISIBLE_DEVICES:-0}"
    local max_mem="${MAX_GPU_MEM_MIB:-500}"
    local max_util="${MAX_GPU_UTIL:-10}"
    # The default limit suits an 8-16 core bench node. A 112-core node is
    # not contended at load 8, so the limit scales with the core count when
    # MAX_CPU_LOAD is not set explicitly.
    local ncpu
    ncpu=$(nproc 2>/dev/null || echo 8)
    local max_load="${MAX_CPU_LOAD:-$(awk -v n="$ncpu" 'BEGIN{printf "%.1f", (n/4.0 > 4.0 ? n/4.0 : 4.0)}')}"
    export MAX_CPU_LOAD="$max_load"

    # Sample three times and take the median. The KT Cloud container's
    # nvidia-smi occasionally returns a wild value - one sample claimed
    # 61439 MiB in use on a GPU that was measured at 0 MiB five seconds
    # either side - and a single bad reading must not abort a sweep.
    local line mem util s1 s2 s3
    local -a mems=() utils=()
    local k
    # A CPU-only sweep (geo85 has no GPU at all) skips the GPU check.
    if [ "${REQUIRE_GPU:-1}" = "0" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
        mems=(0 0 0); utils=(-1 -1 -1)
    else
    for k in 1 2 3; do
        line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                          --format=csv,noheader,nounits \
                | awk -F', ' -v g="$gpu" '$1==g')
        if [ -z "$line" ]; then
            echo "FATAL: GPU $gpu not found" >&2; return 1
        fi
        mems+=("$(echo "$line" | awk -F', ' '{print $2}')")
        utils+=("$(echo "$line" | awk -F', ' '{print $3}')")
        [ "$k" -lt 3 ] && sleep 1
    done
    fi
    mem=$(printf '%s\n' "${mems[@]}" | sort -n | sed -n 2p)
    util=$(printf '%s\n' "${utils[@]}" | sort -n | sed -n 2p)
    # Some containers (KT Cloud H100) report utilisation as "[Not Found]".
    # Treat an unparseable value as unknown rather than letting the integer
    # comparison abort the script with a shell error.
    case "$util" in
        ''|*[!0-9]*) util=-1 ;;
    esac
    if [ "$mem" -gt "$max_mem" ] || { [ "$util" -ge 0 ] && [ "$util" -gt "$max_util" ]; }; then
        echo "FATAL: GPU $gpu is busy (${mem} MiB used, ${util}% util) - timings" >&2
        echo "       from a shared GPU are invalid (RULES.md R7-7)." >&2
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >&2
        return 1
    fi

    local load
    load=$(awk '{print $1}' /proc/loadavg)
    if awk -v l="$load" -v m="$max_load" 'BEGIN{exit !(l > m)}'; then
        echo "FATAL: CPU load average is ${load} (limit ${max_load}) - CPU timings" >&2
        echo "       from a contended node are invalid (RULES.md R7-7)." >&2
        return 1
    fi
    echo "node check: GPU $gpu ${mem} MiB / ${util}% util (-1 = unavailable),"\
         "load ${load} of limit ${max_load} - OK"
}

# The start-of-run check cannot see a job that arrives mid-sweep. Calling this
# afterwards turns that into a visible warning instead of a silent corruption.
recheck_idle_node() {
    [ "${REQUIRE_IDLE:-1}" = "0" ] && return 0
    # The GPU too: a tenant that arrives mid-sweep contaminates the timings.
    local gmem gutil
    if read -r gmem gutil <<<"$(gpu_sample)"; then
        if [ "$gmem" -gt "${MAX_GPU_MEM_MIB:-500}" ]; then
            echo "WARNING: GPU ${CUDA_VISIBLE_DEVICES:-0} shows ${gmem} MiB used AFTER the run -" >&2
            echo "         another process may have arrived mid-sweep (R7-7); cross-check." >&2
        fi
    fi
    local load
    load=$(awk '{print $1}' /proc/loadavg)
    if awk -v l="$load" -v m="${MAX_CPU_LOAD:-4.0}" 'BEGIN{exit !(l > m)}'; then
        echo "WARNING: CPU load rose to ${load} DURING the run - a job arrived" >&2
        echo "         mid-sweep, so these timings may be contaminated (R7-7)." >&2
        echo "         Cross-check against an independent run before reporting." >&2
        return 1
    fi
    echo "node recheck: load ${load} - still clean"
}
