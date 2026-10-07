#!/usr/bin/env bash
#########################################################################
#  Module: tier2_sweep.sh                                               #
#  Description: The tier-2 (model-level) sweep of docs/04 on one node:  #
#               scheme x solver x CFL x case x backend for the v0.5     #
#               full core, every run judged for accuracy against a      #
#               converged reference on the same grid (fb at CFL 0.1)    #
#               so the CSV carries both halves of R8 - error and time   #
#               to solution - for the Pareto plane.                     #
#                                                                       #
#  Node roles (docs/04 S3): BACKENDS=cpu on geo85 (PBS exclusive),      #
#  BACKENDS="cpu gpu" on gpgpu, BACKENDS=gpu on ktcloud.                 #
#  Two modes: the default runs each case to its physical horizon and    #
#  measures the error against the converged reference (piece S); with   #
#  STEPS=<n> it runs a fixed step count with no reference (piece H, the  #
#  byte-identical problem solved on every node - accuracy is settled by  #
#  piece S, only cost per step is compared).                             #
#  Pipeline: toml2nml --v05 -> cfd_exp3d5* -> state_error3d5 -> CSV     #
#########################################################################
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY="${PY:-python3}"
NVHPC_BIN="${NVHPC_BIN:-}"
[ -n "$NVHPC_BIN" ] && export PATH="$NVHPC_BIN:$PATH" && \
    export LD_LIBRARY_PATH="${NVHPC_BIN%/bin}/lib:${LD_LIBRARY_PATH:-}"
# Groups (cpu, gpu) or single backends (fortran_serial fortran_omp openacc cuda jax,
# and their fp32 twins fortran_serial_sp fortran_omp_sp openacc_sp cuda_sp - axis G).
BACKENDS="${BACKENDS:-cpu gpu}"
want() { case " $BACKENDS " in *" $1 "*) return 0;; esac; return 1; }
want_b() { want "$1" || want "$2"; }          # want_b <backend> <group>
JAX_PY="${JAX_PY:-python3}"; export JAX_PY
source "$(dirname "$0")/require_idle_node.sh"
want_b fortran_serial cpu || want_b fortran_omp cpu || MAX_CPU_LOAD=1e9
want_b openacc gpu || want_b cuda gpu || want_b jax gpu || want cuda_sp || want openacc_sp || want cuda_mixed || export REQUIRE_GPU=0
require_idle_node || exit 1

NX="${NX:-100}"; NZ="${NZ:-30}"
STEPS="${STEPS:-}"                                 # fixed-step mode (piece H)
CASES="${CASES:-baroclinic_igw basin_seiche lock_exchange}"
SCHEMES="${SCHEMES:-fb theta0.5 theta0.6 theta1.0 split}"
SOLVERS="${SOLVERS:-pcg_jacobi multigrid}"       # theta family only
CFLS="${CFLS:-0.5 2 8 32}"                        # theta and split
CFLS_FB="${CFLS_FB:-0.25 0.5 0.8}"                # explicit barotropic limit
REF_CFL="${REF_CFL:-0.1}"
THREADS="${THREADS:-1 16}"
CPU_SERIAL="${CPU_SERIAL:-1}"                      # 0 skips the serial build (big grids)
REPEAT="${REPEAT:-5}"
EXTRA_SET="${EXTRA_SET:-}"                         # extra --set flags for every bundle
TAG="${TAG:-}"                                     # appended to the case name in the CSV
                                                   # (physics decomposition: same case, different switches)
# The reference is a converged run, so any R2-gated backend may produce it;
# the fastest one available is used unless REFBIN says otherwise.
REFBIN="${REFBIN:-}"
# Resume: skip configurations already present in RESUME (a results.csv) and
# reuse REF_STATE_<case> / REF_METRICS_<case> files instead of recomputing the
# reference (the run was stopped, e.g. to add the divergence guard - N18).
RESUME="${RESUME:-}"
if [ -z "$REFBIN" ]; then
  for b in libs/cuda/build/cfd_exp3d5_cuda libs/fortran/build/cfd_exp3d5_acc \
           libs/fortran/build/cfd_exp3d5 libs/fortran/build/cfd_exp3d5_serial; do
    if [ -x "$b" ]; then
      case "$b" in *_cuda) want_b cuda gpu && { REFBIN=$b; break; } ;; *_acc) want_b openacc gpu && { REFBIN=$b; break; } ;; *) REFBIN=$b; break ;; esac
    fi
  done
fi
STAMP=$(date +%Y%m%d-%H%M%S)
# Two PBS jobs on a shared home can start in the same second: the pid and the
# compute node keep their bundles apart (they collided once - N12).
OUT="output/tier2_${STAMP}_$(hostname -s)_$$"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "host,backend,threads,case,scheme,theta,solver,cfl,nx,nz,n_steps,dt,wall_s,wall_mad_s,solver_iters,substeps,l2_eta,l2_u,l2_max,sim_s_per_wall_s" > "$CSV"
# R6: a run without a manifest is not reportable.
"$PY" tools/sweep_manifest.py "$OUT" --phase start >/dev/null 2>&1 || \
    echo "warning: manifest not written for $OUT (R6)" >&2
HOST="${BENCH_HOST:-$(hostname)}"

# Physics per case (docs/04 S2). Everything else comes from config/*.toml.
case_sets() {
  case "$1" in
    baroclinic_igw) echo "--set bathymetry.kind=seamount --set grid.Lx=4.0e5 --set grid.Ly=4.0e5 --set physics3d_v05.pgf=keep" ;;
    basin_seiche)  echo "--set bathymetry.kind=seamount --set domain.bc_x=closed --set domain.bc_y=closed --set physics3d.N2=0.0 --set physics3d.nu=0.0 --set physics3d.kappa=0.0 --set physics3d_v05.pgf=keep" ;;
    seamount_rest) echo "--set bathymetry.kind=seamount --set physics3d.N2=0.0 --set physics3d.nu=1e-3 --set physics3d.kappa=1e-3 --set physics3d_v05.pgf=keep" ;;
    # spec v0.6 (E14): case (3) with the TKE closure and TVD-limited third-order advection
    lock_exchange_v06) echo "--set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=up3_tvd --set physics3d_v06.closure=tke --set physics3d_v05.eos=teos10 --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6 --set physics3d.tau_x=0.05 --set physics3d_v05.pgf=keep --set grid.Lx=6.4e4 --set grid.Ly=6.4e3 --set physics.H=20.0" ;;
    lock_exchange_v06r) echo "--set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=up3_tvd --set physics3d_v06.closure=tke --set physics3d_v06.mxl=recursive --set physics3d_v05.eos=teos10 --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=1e-5 --set physics3d.kappa=1e-6 --set physics3d.tau_x=0.05 --set physics3d_v05.pgf=keep --set grid.Lx=6.4e4 --set grid.Ly=6.4e3 --set physics.H=20.0" ;;
    lock_exchange) echo "--set bathymetry.kind=rough --set physics3d_v04.tracers=TS --set scheme.advection=centered2 --set physics3d_v05.eos=teos10 --set physics3d_v04.A_h=10.0 --set physics3d_v04.K_h=10.0 --set physics.f0=0.0 --set physics3d.N2=0.0 --set physics3d.nu=1e-3 --set physics3d.kappa=1e-3 --set physics3d_v05.pgf=keep --set grid.Lx=6.4e4 --set grid.Ly=6.4e3 --set physics.H=20.0" ;;
  esac
}
scheme_sets() {  # <token> -> --set flags (theta value is derived separately:
                 # a variable set inside $(...) never reaches the caller)
  case "$1" in
    fb)      echo "--set scheme.name=fb" ;;
    split)   echo "--set scheme.name=split_explicit" ;;
    theta*)  echo "--set scheme.name=theta --set scheme.theta=${1#theta}" ;;
  esac
}
theta_of() { case "$1" in theta*) echo "${1#theta}";; *) echo 0;; esac; }
is_ts() { case "$1" in lock_exchange*) return 0;; *) return 1;; esac; }
case_of() { case "$1" in lock_exchange_v06|lock_exchange_v06r) echo lock_exchange;; *) echo "$1";; esac; }   # toml case behind a token

record() {  # record <backend> <threads> <prefix> <case> <scheme> <theta> <solver> <cfl> <refstate> <ts>
  local b="$1" t="$2" pre="$3" cs="$4" sch="$5" th="$6" sv="$7" cfl="$8" ref="$9" ts="${10}"
  [ -n "$TAG" ] && cs="${cs}${TAG}"
  if [ ! -f "${pre}_metrics.json" ]; then
    echo "MISSING RESULT: $b $cs $sch $sv cfl=$cfl nx=$NX (backend produced no metrics - see N20)" >&2
    echo "$HOST,$b,$t,$cs,$sch,$th,$sv,$cfl,$NX,$NZ,0,nan,nan,nan,0,0,nan,nan,nan,nan" >> "$CSV"
    return 1
  fi
  local tsflag=""; [ "$ts" = "1" ] && tsflag="--ts"
  local err='{}'
  if grep -q '"diverged": true' "${pre}_metrics.json" 2>/dev/null; then
    err='{"l2_rel_eta": Infinity, "l2_rel_u": Infinity, "l2_rel_max": Infinity}'
  elif [ -f "$ref" ]; then
    err=$("$PY" tools/state_error3d5.py "${pre}_state3d5.bin" "$ref" --nx "$NX" --nz "$NZ" $tsflag 2>/dev/null) || err='{}'
  fi
  "$PY" - "$pre" "$b" "$t" "$cs" "$sch" "$th" "$sv" "$cfl" "$err" "$HOST" "$NX" "$NZ" >> "$CSV" <<'PYEOF'
import json, sys
pre, b, t, cs, sch, th, sv, cfl, err, host, nx, nz = sys.argv[1:13]
m = json.load(open(pre + "_metrics.json")); e = json.loads(err)
def num(k, d=m):
    v = d.get(k); return "nan" if v is None else f"{float(v):.6e}"
ns, dt, w = int(m.get("n_steps", 0)), float(m.get("dt", 0)), float(m.get("wall_s", 0) or 0)
print(",".join([host, b, t, cs, sch, th, sv, cfl, nx, nz, str(ns), f"{dt:.6e}",
                num("wall_s"), num("wall_mad_s"), str(m.get("solver_iterations", 0)),
                str(m.get("barotropic_substeps", 0)), num("l2_rel_eta", e), num("l2_rel_u", e),
                num("l2_rel_max", e), f"{(ns*dt/w) if w > 0 else float('nan'):.6e}"]))
PYEOF
}

STEPFLAG=""; [ -n "$STEPS" ] && STEPFLAG="--steps $STEPS"
echo "host=$HOST nx=$NX nz=$NZ steps=${STEPS:-horizon} backends=[$BACKENDS] cases=[$CASES] schemes=[$SCHEMES] refbin=$REFBIN"
for cs in $CASES; do
  ts=0; is_ts "$cs" && ts=1
  rp="$OUT/${cs}_ref"
  if [ -z "$STEPS" ] && [ -n "$RESUME" ] && [ -f "$(dirname "$RESUME")/${cs}_ref_state3d5.bin" ]; then
    cp "$(dirname "$RESUME")/${cs}_ref_state3d5.bin" "${rp}_state3d5.bin"
    cp "$(dirname "$RESUME")/${cs}_ref_metrics.json" "${rp}_metrics.json" 2>/dev/null
    echo "reference $cs: reused from $(dirname "$RESUME")"
  elif [ -z "$STEPS" ]; then
    # ---- converged reference: fb at REF_CFL on the fastest gated backend
    "$PY" tools/toml2nml.py --v05 $(case_sets "$cs") $EXTRA_SET --set scheme.name=fb --case "$(case_of "$cs")" \
          --nx "$NX" --nz "$NZ" --cfl "$REF_CFL" --prefix "$rp" --n-repeat 1 --n-warmup 0 >/dev/null || { echo "ref bundle failed: $cs"; continue; }
    OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=close OMP_PLACES=cores "$REFBIN" "$rp.nml" >/dev/null 2>&1 || { echo "reference run failed: $cs"; continue; }
    echo "reference $cs: $(grep -o '"n_steps": [0-9]*' "${rp}_metrics.json") wall=$(grep -o '"wall_s": [0-9.e+-]*' "${rp}_metrics.json")"
  fi

  for sch in $SCHEMES; do
    SSET=$(scheme_sets "$sch"); THETA=$(theta_of "$sch")
    if [ "$sch" = "fb" ]; then cfls="$CFLS_FB"; else cfls="$CFLS"; fi
    if [ "${sch#theta}" != "$sch" ]; then solvers="$SOLVERS"; else solvers="none"; fi
    for sv in $solvers; do
      SV=""; [ "$sv" != "none" ] && SV="--set solver.kind=$sv"
      for cfl in $cfls; do
        tag="${cs}_${sch}_${sv}_c${cfl}"; pre="$OUT/$tag"
        if [ -n "$RESUME" ] && grep -q ",${cs},${sch},[^,]*,${sv},${cfl},${NX}," "$RESUME" 2>/dev/null; then
          grep ",${cs},${sch},[^,]*,${sv},${cfl},${NX}," "$RESUME" >> "$CSV"; echo "resumed $tag"; continue
        fi
        "$PY" tools/toml2nml.py --v05 $(case_sets "$cs") $EXTRA_SET $SSET $SV --case "$(case_of "$cs")" $STEPFLAG \
              --nx "$NX" --nz "$NZ" --cfl "$cfl" --prefix "$pre" --n-repeat "$REPEAT" --n-warmup 1 >/dev/null || { echo "bundle failed: $tag"; continue; }
        if want_b fortran_serial cpu && [ "$CPU_SERIAL" = "1" ] && [ -x libs/fortran/build/cfd_exp3d5_serial ]; then
          rm -f "${pre}_metrics.json"; libs/fortran/build/cfd_exp3d5_serial "$pre.nml" >/dev/null 2>&1
          record fortran_serial 1 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        if want_b fortran_omp cpu; then
          for t in $THREADS; do
            rm -f "${pre}_metrics.json"
            OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores libs/fortran/build/cfd_exp3d5 "$pre.nml" >/dev/null 2>&1
            record fortran_omp "$t" "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
          done
        fi
        if want_b openacc gpu && [ -x libs/fortran/build/cfd_exp3d5_acc ]; then
          rm -f "${pre}_metrics.json"; libs/fortran/build/cfd_exp3d5_acc "$pre.nml" >/dev/null 2>&1
          record openacc 0 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        if want_b cuda gpu && [ -x libs/cuda/build/cfd_exp3d5_cuda ]; then
          rm -f "${pre}_metrics.json"; libs/cuda/build/cfd_exp3d5_cuda "$pre.nml" >/dev/null 2>&1
          record cuda 0 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        if want_b jax gpu && [ -x libs/jax/build/cfd_exp3d5_jax ]; then
          rm -f "${pre}_metrics.json"; libs/jax/build/cfd_exp3d5_jax "$pre.nml" >/dev/null 2>&1
          record jax 0 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        # ---- fp32 twins (axis G). Same bundle, same iteration target; the
        # state file stays fp64 so the error against the reference is comparable.
        if want cuda_sp && [ -x libs/cuda/build/cfd_exp3d5_cuda_sp ]; then
          rm -f "${pre}_metrics.json"; libs/cuda/build/cfd_exp3d5_cuda_sp "$pre.nml" >/dev/null 2>&1
          record cuda_sp 0 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        # ---- mixed precision (expr/E18): fp64 core, fp32 closure. Same bundle;
        # the state file stays fp64 so the difference against fp64 is measurable.
        if want cuda_mixed && [ -x libs/cuda/build/cfd_exp3d5_cuda_mixed ]; then
          rm -f "${pre}_metrics.json"; libs/cuda/build/cfd_exp3d5_cuda_mixed "$pre.nml" >/dev/null 2>&1
          record cuda_mixed 0 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        if want openacc_sp && [ -x libs/fortran/build/cfd_exp3d5_acc_sp ]; then
          rm -f "${pre}_metrics.json"; libs/fortran/build/cfd_exp3d5_acc_sp "$pre.nml" >/dev/null 2>&1
          record openacc_sp 0 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        if want fortran_serial_sp && [ -x libs/fortran/build/cfd_exp3d5_serial_sp ]; then
          rm -f "${pre}_metrics.json"; libs/fortran/build/cfd_exp3d5_serial_sp "$pre.nml" >/dev/null 2>&1
          record fortran_serial_sp 1 "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
        fi
        if want fortran_omp_sp; then
          for t in $THREADS; do
            rm -f "${pre}_metrics.json"
            OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores libs/fortran/build/cfd_exp3d5_sp "$pre.nml" >/dev/null 2>&1
            record fortran_omp_sp "$t" "$pre" "$cs" "$sch" "$THETA" "$sv" "$cfl" "${rp}_state3d5.bin" "$ts"
          done
        fi
        rm -f "${pre}_init.bin" "${pre}_domain.bin" "${pre}_state3d5.bin"
        # Per-configuration GPU sample. A contending tenant that runs through
        # all five repeats inflates the median WITHOUT inflating the MAD, so a
        # low MAD is not evidence of an exclusive device (docs/90 N26). The
        # sample lets a contaminated row be identified after the fact.
        if [ "${REQUIRE_GPU:-1}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1; then
          printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$tag" \
            "$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')" \
            >> "$OUT/gpu_samples.log"
        fi
        # A multi-threaded run that is SLOWER than the single-thread run of the
        # same configuration means the node was not ours (2026-09-13: 96 threads
        # at 100.7 s against 1 thread at 27.8 s). Cheap to check, and it catches
        # the case an idle-at-start guard cannot.
        if [ -s "$CSV" ]; then
          "$PY" - "$CSV" <<'PYCHK' || true
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
g = defaultdict(dict)
for r in rows:
    try:
        w = float(r["wall_s"]); t = int(r["threads"])
    except (ValueError, KeyError):
        continue
    if w > 0 and r["backend"].startswith("fortran"):
        g[(r["case"], r["scheme"], r["solver"], r["nx"])][t] = w
for k, v in g.items():
    if 1 in v:
        for t, w in v.items():
            if t > 1 and w > v[1]:
                print(f"  !! THREAD INVERSION {k}: t{t} {w:.3f}s > t1 {v[1]:.3f}s "
                      f"- the node was not exclusive (R7-7)", file=sys.stderr)
PYCHK
        fi
        echo "done $tag $(date +%H:%M:%S)"
      done
    done
  done
  rm -f "${rp}_init.bin" "${rp}_domain.bin"
done
"$PY" tools/sweep_manifest.py "$OUT" --phase end >/dev/null 2>&1 || true
echo "csv: $CSV"
recheck_idle_node || true
