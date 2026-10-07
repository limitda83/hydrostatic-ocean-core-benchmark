#!/usr/bin/env python3
#########################################################################
#  Module: sweep_manifest                                               #
#  Description: Writes the R6 manifest of a tier-2 sweep directory:     #
#               git SHA and dirty flag, host and CPU/GPU model,         #
#               compiler and library versions with the flags the        #
#               binaries were built with, the environment that steers   #
#               the sweep (threads, pinned GPU, case and scheme sets),   #
#               the configuration tree with its SHA256, and the UTC     #
#               start/end. A sweep without one is not reportable.       #
#  Pipeline: tier2_sweep.sh (start and end) -> manifest.json            #
#########################################################################

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENV_KEYS = ["NX", "NZ", "STEPS", "REPEAT", "CASES", "SCHEMES", "SOLVERS", "CFLS", "CFLS_FB",
            "THREADS", "BACKENDS", "CPU_SERIAL", "EXTRA_SET", "REF_CFL", "BENCH_HOST",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
            "MAX_CPU_LOAD", "MAX_GPU_MEM_MIB", "WAIT_GPU_MIN", "WAIT_LOAD_MIN",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "PY", "JAX_PY", "RESUME"]


def run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              cwd=cwd, check=False).stdout.strip()
    except Exception:                                  # noqa: BLE001 - provenance is best effort
        return ""


def git_provenance(root: Path) -> dict:
    return {"sha": run(["git", "rev-parse", "HEAD"], root),
            "short": run(["git", "rev-parse", "--short", "HEAD"], root),
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
            "dirty": bool(run(["git", "status", "--porcelain"], root)),
            "describe": run(["git", "describe", "--always", "--dirty"], root)}


def host_info() -> dict:
    info = {"hostname": platform.node(), "system": platform.system(),
            "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    model = run(["bash", "-lc", "lscpu | sed -n 's/^Model name: *//p' | head -1"]) \
        or run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if model:
        info["cpu_model"] = model
    numa = run(["bash", "-lc", "lscpu | sed -n 's/^NUMA node(s): *//p' | head -1"])
    if numa:
        info["numa_nodes"] = numa
    gpus = run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader"])
    if gpus:
        info["gpus"] = [g.strip() for g in gpus.splitlines()]
    return info


def toolchain(root: Path) -> dict:
    """Compiler and library versions, and the flags each binary carries."""
    out: dict = {}
    for name, cmd in (("gfortran", ["gfortran", "--version"]),
                      ("nvfortran", ["nvfortran", "--version"]),
                      ("nvcc", ["nvcc", "--version"]),
                      ("mpirun", ["mpirun", "--version"])):
        v = run(cmd)
        if v:
            out[name] = v.splitlines()[0] if name != "nvfortran" else " ".join(v.split()[:3])
    # The literal compile command with every variable resolved, from a dry run
    # of the same target (make -n builds nothing). Conditional blocks inside the
    # Makefile make a textual scrape ambiguous - the gfortran and nvfortran
    # branches both define OPT/OMP/STD.
    for sub, targets in (("libs/fortran", ("all3d5", "serial3d5", "acc3d5")),
                         ("libs/cuda", ("all3d5",))):
        d = root / sub
        if not (d / "Makefile").exists():
            continue
        for t in targets:
            line = run(["bash", "-lc",
                        f"make -n {t} 2>/dev/null | grep -E '(nvcc|gfortran|nvfortran|ifx|ifort).*-o ' | head -1"], d)
            if line:
                out.setdefault("build_commands", {})[f"{sub}:{t}"] = " ".join(line.split())
    for b in ("libs/fortran/build/cfd_exp3d5", "libs/fortran/build/cfd_exp3d5_serial",
              "libs/fortran/build/cfd_exp3d5_acc", "libs/cuda/build/cfd_exp3d5_cuda"):
        p = root / b
        if p.exists():
            out.setdefault("binaries", {})[b] = {
                "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                "bytes": p.stat().st_size}
    jax_py = os.environ.get("JAX_PY")
    if jax_py:
        v = run([jax_py, "-c", "import jax; print(jax.__version__, jax.devices()[0].device_kind)"])
        if v:
            out["jax"] = v
    return out


def config_snapshot(root: Path) -> dict:
    """Hash the raw bytes: a stray non-UTF-8 byte in a config must not stop the
    manifest (it did on a node whose config carried a latin-1 character)."""
    cfgs = {}
    h = sha256()
    for p in sorted((root / "config").glob("*.toml")):
        raw = p.read_bytes()
        h.update(raw)
        cfgs[p.name] = raw.decode("utf-8", errors="replace")
    return {"sha256": h.hexdigest(), "files": cfgs}


def main() -> int:
    ap = argparse.ArgumentParser(description="write the R6 manifest of a sweep directory")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--phase", choices=["start", "end"], default="start")
    ap.add_argument("--retroactive", action="store_true",
                    help="the sweep already ran: mark the manifest as written after the fact")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "manifest.json"
    if args.phase == "end" and path.exists():
        m = json.loads(path.read_text())
        m["ended_utc"] = datetime.now(timezone.utc).isoformat()
        csv = args.out_dir / "results.csv"
        if csv.exists():
            m["rows"] = max(0, len(csv.read_bytes().splitlines()) - 1)
        path.write_text(json.dumps(m, indent=2, default=str))
        return 0
    manifest = {
        "kind": "tier2_sweep",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(args.out_dir),
        "git": git_provenance(args.root),
        "host": host_info(),
        "toolchain": toolchain(args.root),
        "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
        "config": config_snapshot(args.root),
        "rules": "RULES.md R6 (self-describing runs), R7 (timing protocol)",
    }
    if args.retroactive:
        # The sweep predates the manifest hook. Provenance is still the code and
        # binaries that produced it (same SHA, same build), but say so plainly.
        manifest["retroactive"] = True
        manifest["retroactive_note"] = ("written after the sweep; started_utc is the manifest "
                                        "time, not the run time - see the directory timestamp "
                                        "and results.csv mtime")
        csv = args.out_dir / "results.csv"
        if csv.exists():
            st = csv.stat()
            manifest["results_csv_mtime_utc"] = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
            manifest["rows"] = max(0, len(csv.read_bytes().splitlines()) - 1)
            manifest["env"].setdefault("_from_csv", "sweep env not captured at run time")
    path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
