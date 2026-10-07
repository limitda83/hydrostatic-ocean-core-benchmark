#!/usr/bin/env python3
#########################################################################
#  Module: compare_backends                                             #
#  Description: The R2 gate. Runs a compiled backend and the NumPy      #
#               reference on an identical discrete problem and compares #
#               the final states. No backend may be merged, and no      #
#               timing may be reported, until this passes (R2, R4).     #
#  Pipeline: toml2nml -> compiled backend -> compare_backends -> verdict#
#########################################################################

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from libs.core.schemes import State, Stepper
from libs.utils.config import load_config
from tools.toml2nml import resolve_run, write_namelist

# Tolerances from RULES.md R2 (fp64 backends).
TOL_ONE_STEP = 1e-12
TOL_FULL_RUN = 1e-9


def read_state_bin(path: Path, nx: int, ny: int) -> State:
    """Read the Fortran stream dump (eta, u, v), each nx*ny fp64.

    Fortran (nx,ny) column-major and NumPy [ny,nx] row-major share byte
    order (spec S0), so a plain reshape is correct - no transpose.
    """
    raw = np.fromfile(path, dtype=np.float64)
    n = nx * ny
    if raw.size != 3 * n:
        raise ValueError(f"{path}: expected {3 * n} doubles, got {raw.size}")
    eta = raw[0:n].reshape(ny, nx)
    u = raw[n:2 * n].reshape(ny, nx)
    v = raw[2 * n:3 * n].reshape(ny, nx)
    return State(u=u, v=v, eta=eta)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.sqrt(np.sum(b * b)))
    num = float(np.sqrt(np.sum((a - b) ** 2)))
    return num / den if den > 0.0 else num


def run_reference(config, nx: int, cfl: float, case_name: str,
                  n_steps_override: int | None = None) -> tuple[State, dict]:
    """Integrate the NumPy reference on exactly the resolved (dt, n_steps)."""
    from libs.core.cases import build_case
    resolved = resolve_run(config, nx, cfl, case_name)
    case, _ = build_case(case_name, resolved["grid"], resolved["physics"], config)
    stepper = Stepper(resolved["grid"], resolved["physics"], resolved["params"],
                      resolved["dt"])
    n_steps = n_steps_override or resolved["n_steps"]
    return stepper.integrate(case.initial(), n_steps), resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="R2 gate: compiled backend vs reference")
    parser.add_argument("--config", default="config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--case", default=None)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--binary", type=Path,
                        default=Path("libs/fortran/build/cfd_exp"))
    parser.add_argument("--workdir", type=Path, default=Path("output/r2_gate"))
    parser.add_argument("--n-repeat", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    from main import _apply_overrides
    config = _apply_overrides(load_config(args.config), args.set)
    case_name = args.case or config.get("case.name")
    if not args.binary.exists():
        print(f"FAIL: backend binary not found: {args.binary}\n"
              f"      build it with: make -C libs/fortran", file=sys.stderr)
        return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    verdicts: list[tuple[str, float, float, bool]] = []

    for label, steps_override, tol in (("1 step", 1, TOL_ONE_STEP),
                                       ("full run", None, TOL_FULL_RUN)):
        resolved = resolve_run(config, args.nx, args.cfl, case_name, args.steps)
        n_steps = steps_override or resolved["n_steps"]
        # Pin n_steps for the single-step probe by shrinking t_final to one dt.
        resolved_run = dict(resolved)
        if steps_override is not None:
            resolved_run["n_steps"] = 1
            resolved_run["t_final"] = resolved["dt"]

        tag = label.replace(" ", "_")
        prefix = f"{args.workdir}/{tag}"
        nml = write_namelist(Path(f"{prefix}.nml"), config, resolved_run,
                             case_name, prefix, args.n_repeat, 1)
        proc = subprocess.run([str(args.binary), str(nml)], capture_output=True,
                              text=True, check=False)
        if proc.returncode != 0:
            print(f"FAIL: backend exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}",
                  file=sys.stderr)
            return 2

        got = read_state_bin(Path(f"{prefix}_state.bin"), args.nx, args.nx)
        ref, _ = run_reference(config, args.nx, args.cfl, case_name, n_steps)
        d_eta = rel_l2(got.eta, ref.eta)
        d_uv = max(rel_l2(got.u, ref.u), rel_l2(got.v, ref.v))
        worst = max(d_eta, d_uv)
        verdicts.append((label, d_eta, d_uv, worst < tol))
        print(f"{label:<10} n_steps={n_steps:<6d} rel L2: eta={d_eta:.3e} "
              f"uv={d_uv:.3e}  tol={tol:.0e}  "
              f"{'PASS' if worst < tol else 'FAIL'}")

    summary = {"case": case_name, "nx": args.nx, "cfl": args.cfl,
               "binary": str(args.binary),
               "checks": [{"label": lbl, "l2_eta": e, "l2_uv": uv, "passed": ok}
                          for lbl, e, uv, ok in verdicts]}
    (args.workdir / "r2_gate.json").write_text(json.dumps(summary, indent=2))

    passed = all(ok for *_, ok in verdicts)
    print(f"\nR2 GATE: {'PASS' if passed else 'FAIL'} "
          f"({args.binary} vs NumPy fp64 reference)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
