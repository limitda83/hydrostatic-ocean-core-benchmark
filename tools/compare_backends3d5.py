#!/usr/bin/env python3
#########################################################################
#  Module: compare_backends3d5                                          #
#  Description: The R2 gate for the spec v0.5 compiled backends. Writes #
#               the v0.5 bundle (namelist, domain, initial state), runs #
#               the backend, and compares its final state with the      #
#               NumPy Stepper3DV05 on the identical discrete problem.   #
#  Pipeline: toml2nml --v05 -> cfd_exp3d5 -> compare_backends3d5        #
#########################################################################

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.core.model3d_v05 import Stepper3DV05               # noqa: E402
from libs.utils.config import load_config                    # noqa: E402
from tools.compare_backends3d import field_error             # noqa: E402
from tools.toml2nml import write_v05_bundle                  # noqa: E402

TOL_ONE_STEP = 1e-12
TOL_FULL_RUN = 1e-9


def read_state(path: Path, nx: int, ny: int, nz: int, ts: bool) -> dict:
    raw = np.fromfile(path, dtype=np.float64)
    n2, n3 = nx * ny, nx * ny * nz
    names = ["u", "v", "b"] + (["T", "S"] if ts else [])
    if raw.size != n2 + len(names) * n3:
        raise ValueError(f"{path}: expected {n2 + len(names) * n3} doubles, got {raw.size}")
    out = {"eta": raw[:n2].reshape(ny, nx)}
    off = n2
    for name in names:
        out[name] = raw[off:off + n3].reshape(nz, ny, nx)
        off += n3
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="R2 gate for the v0.5 backends")
    ap.add_argument("--config", default="config", type=Path)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--case", default="seamount_rest")
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--binary", type=Path, default=Path("libs/fortran/build/cfd_exp3d5"))
    ap.add_argument("--workdir", type=Path, default=Path("output/r2_gate3d5"))
    args = ap.parse_args()

    from main import _apply_overrides
    config = _apply_overrides(load_config(args.config), args.set)
    if not args.binary.exists():
        print(f"FAIL: backend binary not found: {args.binary}", file=sys.stderr)
        return 2
    args.workdir.mkdir(parents=True, exist_ok=True)

    verdicts = []
    for label, steps, tol in (("1 step", 1, TOL_ONE_STEP), ("full run", args.steps, TOL_FULL_RUN)):
        prefix = f"{args.workdir}/{label.replace(' ', '_')}"
        r = write_v05_bundle(prefix, config, args.nx, args.nz, args.case, args.cfl,
                             steps, 1, 0)
        proc = subprocess.run([str(args.binary), str(r["nml"])], capture_output=True,
                              text=True, check=False)
        if proc.returncode != 0:
            print(f"FAIL: backend exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}",
                  file=sys.stderr)
            return 2
        ts = r["physics"].tracers == "TS"
        got = read_state(Path(f"{prefix}_state3d5.bin"), args.nx, args.nx, args.nz, ts)

        ref = Stepper3DV05(r["domain"], r["physics"], r["params"], r["dt"]).integrate(
            r["state0"], steps)
        fields = ["u", "v", "b", "eta"] + (["T", "S"] if ts else [])
        scale = max(float(np.sqrt(np.sum(getattr(ref, f) ** 2))) for f in fields)
        d = {f: field_error(got[f], getattr(ref, f), scale) for f in fields}
        # `max` skips NaN (x > nan is False), so a NaN in any field but the
        # first silently returned a finite worst and the gate PASSED (N34).
        vals = list(d.values())
        worst = float("inf") if any(v != v for v in vals) else max(vals)
        verdicts.append((label, d, worst < tol))
        print(f"{label:<10} steps={steps:<5d} rel L2: "
              + " ".join(f"{f}={d[f]:.3e}" for f in fields)
              + f"  tol={tol:.0e}  {'PASS' if worst < tol else 'FAIL'}")

    (args.workdir / "r2_gate3d5.json").write_text(json.dumps(
        {"case": args.case, "nx": args.nx, "nz": args.nz, "binary": str(args.binary),
         "checks": [{"label": l, **v, "passed": ok} for l, v, ok in verdicts]}, indent=2))
    passed = all(ok for *_, ok in verdicts)
    print(f"\nR2 GATE (3D v0.5): {'PASS' if passed else 'FAIL'} ({args.binary} vs NumPy fp64 reference)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
