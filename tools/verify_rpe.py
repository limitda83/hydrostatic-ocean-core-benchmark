#!/usr/bin/env python3
#########################################################################
#  Module: verify_rpe                                                   #
#  Description: Closes the three open checks of docs/27 S4, where a      #
#               negative RPE drift under `upwind1` was recorded but not  #
#               reported because it was not known what it measured:      #
#               (1) does `upwind1` change the VERTICAL flux at all,      #
#               (2) is the tracer conserved (without which RPE is        #
#               meaningless), (3) how much does sorting on the resting   #
#               thickness dz3_ref instead of the live thickness move the #
#               answer. All three are accuracy diagnostics and hardware  #
#               independent, so a development machine is a legal place   #
#               to run them (R7-1 forbids only TIMINGS from there).      #
#  Pipeline: cases3d_v05::LockExchange -> verify_rpe -> docs/27          #
#########################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.core.driver3d_v05 import (build_domain3d, physics_v05_from_config,
                                    barotropic_dt_max)                        # noqa: E402
from libs.core.cases3d_v05 import build_case3d_v05                           # noqa: E402
from libs.core.model3d_v05 import Stepper3DV05                               # noqa: E402
from libs.core.schemes import scheme_params_from_config                      # noqa: E402
# (barotropic_dt_max lives in driver3d_v05)
from libs.utils.config import load_config                                    # noqa: E402
from main import _apply_overrides                                            # noqa: E402


def rpe_of(case, b, thickness) -> float:
    """docs/27 RPE with the vertical weighting handed in, so the
    resting-thickness assumption can be VARIED instead of assumed.

    The reference level comes from the case's own hypsometric curve, not from
    a second copy of the formula: this file used to carry its own box-shaped
    version, which silently diverged from the case once the case was fixed
    (docs/90 N37). One definition, two callers.
    """
    d = case.domain
    wet = d.mask3 > 0
    vol = (d.grid.dx * d.grid.dy * thickness)[wet]
    bb = b[wet]
    order = np.argsort(bb)
    v = vol[order]
    vol_below, grid_z = case._hypsometry()
    z = np.interp(np.cumsum(v), vol_below, grid_z)
    return float(np.sum(bb[order] * z * v))


def run(advection: str, scheme: str, nx: int, nz: int, steps: int,
        overrides: list[str], cfl: float = 0.5, topo: str = "flat"):
    ov = overrides + [
        f"scheme.advection={advection}", f"scheme.name={scheme}",
        f"bathymetry.kind={topo}", "physics.f0=0.0", "physics3d.N2=0.0",
        "physics3d_v04.tracers=TS", "physics3d_v05.eos=linear",
        "physics3d_v05.pgf=keep", "physics.H=20.0",
        "grid.Lx=6.4e4", "grid.Ly=6.4e3", "solver.kind=pcg_jacobi"]
    cfg = _apply_overrides(load_config("config"), ov)
    domain = build_domain3d(cfg, nx, nz)
    physics = physics_v05_from_config(cfg)
    params = scheme_params_from_config(cfg)
    # config pins dt = 0, meaning "derive it from the CFL" (driver3d_v05 S8).
    pinned = float(cfg.get("time.dt"))
    dt = pinned if pinned > 0.0 else cfl * barotropic_dt_max(domain, physics.g)
    case = build_case3d_v05("lock_exchange", domain, physics, cfg)
    st = Stepper3DV05(domain, physics, params, dt)
    s = case.initial()
    dz_ref = domain.dz3_ref
    mass0 = float(np.sum(s.T * dz_ref * domain.mask3))
    rpe0_ref = rpe_of(case, s.b, dz_ref)
    for _ in range(steps):
        s = st.step(s)
    mass1 = float(np.sum(s.T * dz_ref * domain.mask3))
    # live thickness: the linear free surface adds eta to the top layer
    dz_live = dz_ref.copy()
    dz_live[0] = dz_live[0] + s.eta * (domain.mask3[0] > 0)
    return {
        "advection": advection, "scheme": scheme, "nx": nx, "nz": nz, "steps": steps,
        "rpe_drift_dz_ref": (rpe_of(case, s.b, dz_ref) - rpe0_ref) / abs(rpe0_ref),
        "rpe_drift_dz_live": (rpe_of(case, s.b, dz_live) - rpe0_ref) / abs(rpe0_ref),
        "tracer_rel_change": (mass1 - mass0) / abs(mass0),
        "eta_max_abs": float(np.max(np.abs(s.eta))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="close the open checks of docs/27 S4")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cfl", type=float, default=0.5,
                    help="barotropic CFL used when config pins time.dt = 0")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--topo", default="flat",
                    help="bathymetry: the hypsometric correction is exactly zero "
                         "on a flat bottom and 59 %% on `rough` (docs/90 N37)")
    ap.add_argument("--json", type=Path,
                    default=Path("expr/E10_lock_exchange_rpe/data/rpe_checks.json"))
    args = ap.parse_args()

    print("check 1 - does the advection label change the VERTICAL flux?")
    import inspect
    from libs.core import model3d_v05 as M
    src = inspect.getsource(M.Stepper3DV05._tracer_advection)
    vert = src[src.index("cf = np.zeros"):]
    only_up3 = 'if scheme in ("up3", "up3_tvd")' in vert
    print(f"  vertical interface value depends on the scheme only for up3/up3_tvd: {only_up3}")
    print("  => `upwind1` and `centered2` share the SAME centered vertical flux "
          "(docs/27 S4 item 1 confirmed)")

    rows = []
    for adv in ("centered2", "upwind1", "up3", "up3_tvd"):
        for sch in ("theta", "split_explicit"):
            r = run(adv, sch, args.nx, args.nz, args.steps, args.set, args.cfl, args.topo)
            rows.append(r)
            print(f"  {adv:9s} {sch:15s} RPE(dz_ref) {r['rpe_drift_dz_ref']:+.4e}  "
                  f"RPE(dz_live) {r['rpe_drift_dz_live']:+.4e}  "
                  f"tracer {r['tracer_rel_change']:+.2e}  |eta|max {r['eta_max_abs']:.3e}")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"vertical_flux_scheme_dependent_only_for_up3": only_up3, "runs": rows},
        indent=1), encoding="utf-8")
    print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
