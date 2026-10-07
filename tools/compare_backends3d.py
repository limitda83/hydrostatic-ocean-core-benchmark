#!/usr/bin/env python3
#########################################################################
#  Module: compare_backends3d                                           #
#  Description: The R2 gate for the 3D backends. Runs a compiled 3D     #
#               backend and the NumPy reference on an identical         #
#               discrete problem and compares the final state.          #
#  Pipeline: toml2nml -> compiled 3D backend -> compare_backends3d      #
#########################################################################

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from libs.core.cases3d import build_case3d
from libs.core.driver3d import physics3d_from_config
from libs.core.grid import build_grid
from libs.core.model3d import Stepper3D
from libs.core.schemes import scheme_params_from_config
from libs.utils.config import load_config
from tools.toml2nml import resolve_run, write_namelist

TOL_ONE_STEP = 1e-12
TOL_FULL_RUN = 1e-9


def read_state3d(path: Path, nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
    """Read the stream dump: eta[ny,nx] then u, v, b each [nz,ny,nx].

    Fortran (nx,ny,nz) column-major has the same byte order as NumPy
    [nz,ny,nx] row-major (spec S0), so a plain reshape is correct.
    """
    raw = np.fromfile(path, dtype=np.float64)
    n2, n3 = nx * ny, nx * ny * nz
    if raw.size != n2 + 3 * n3:
        raise ValueError(f"{path}: expected {n2 + 3 * n3} doubles, got {raw.size}")
    off = 0
    eta = raw[off:off + n2].reshape(ny, nx); off += n2
    fields = {}
    for name in ("u", "v", "b"):
        fields[name] = raw[off:off + n3].reshape(nz, ny, nx)
        off += n3
    fields["eta"] = eta
    return fields


def field_error(a: np.ndarray, b: np.ndarray, state_scale: float) -> float:
    """Relative L2 where the reference field is meaningful, absolute-relative-
    to-state where it is not.

    baroclinic_igw has eta identically zero, so both backends produce only
    round-off noise there. Dividing that noise by its own norm compares two
    different round-off patterns and always fails - a defect in the metric,
    not in the code. Below 1e-11 of the state scale the error is normalised by
    the state instead, which is well defined and still catches real breakage.
    """
    num = float(np.sqrt(np.sum((a - b) ** 2)))
    den = float(np.sqrt(np.sum(b * b)))
    if den > 1e-11 * state_scale:
        return num / den
    return num / state_scale if state_scale > 0.0 else num


def main() -> int:
    ap = argparse.ArgumentParser(description="R2 gate for the 3D backends")
    ap.add_argument("--config", default="config", type=Path)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--case", default="barotropic3d")
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--nz", type=int, default=30)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--binary", type=Path,
                    default=Path("libs/fortran/build/cfd_exp3d"))
    ap.add_argument("--workdir", type=Path, default=Path("output/r2_gate3d"))
    args = ap.parse_args()

    from main import _apply_overrides
    config = _apply_overrides(load_config(args.config), args.set)
    if not args.binary.exists():
        print(f"FAIL: backend binary not found: {args.binary}", file=sys.stderr)
        return 2
    args.workdir.mkdir(parents=True, exist_ok=True)

    verdicts = []
    for label, steps, tol in (("1 step", 1, TOL_ONE_STEP),
                              ("full run", args.steps, TOL_FULL_RUN)):
        resolved = resolve_run(config, args.nx, args.cfl, args.case, steps, args.nz)
        tag = label.replace(" ", "_")
        prefix = f"{args.workdir}/{tag}"
        nml = write_namelist(Path(f"{prefix}.nml"), config, resolved, args.case,
                             prefix, 1, 0)
        proc = subprocess.run([str(args.binary), str(nml)], capture_output=True,
                              text=True, check=False)
        if proc.returncode != 0:
            print(f"FAIL: backend exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}",
                  file=sys.stderr)
            return 2

        got = read_state3d(Path(f"{prefix}_state3d.bin"), args.nx, args.nx, args.nz)

        grid = build_grid(config, nx=args.nx, ny=args.nx, nz=args.nz)
        physics = physics3d_from_config(config)
        params = scheme_params_from_config(config)
        case, _ = build_case3d(args.case, grid, physics, config)
        ref = Stepper3D(grid, physics, params, resolved["dt"]).integrate(
            case.initial(), steps)

        scale = max(float(np.sqrt(np.sum(f * f)))
                    for f in (ref.u, ref.v, ref.b, ref.eta))
        d = {name: field_error(got[name], getattr(ref, name), scale)
             for name in ("u", "v", "b", "eta")}
        worst = max(d.values())
        verdicts.append((label, d, worst < tol))
        print(f"{label:<10} steps={steps:<5d} rel L2: "
              f"u={d['u']:.3e} v={d['v']:.3e} b={d['b']:.3e} eta={d['eta']:.3e}  "
              f"tol={tol:.0e}  {'PASS' if worst < tol else 'FAIL'}")

    (args.workdir / "r2_gate3d.json").write_text(json.dumps(
        {"case": args.case, "nx": args.nx, "nz": args.nz,
         "binary": str(args.binary),
         "checks": [{"label": l, **v, "passed": ok} for l, v, ok in verdicts]},
        indent=2))
    passed = all(ok for *_, ok in verdicts)
    print(f"\nR2 GATE (3D): {'PASS' if passed else 'FAIL'} "
          f"({args.binary} vs NumPy fp64 reference)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
