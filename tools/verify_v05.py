#!/usr/bin/env python3
#########################################################################
#  Module: verify_v05                                                   #
#  Description: The spec v0.5 verification suite V5-1 .. V5-6 of        #
#               docs/03_discretization_spec.md S10.8. Every case either #
#               has an exact answer (V5-1, V5-4) or a diagnostic that   #
#               is provably zero for the exact equations, so the        #
#               measured number is the error itself.                    #
#  Pipeline: libs/core/* -> verify_v05 -> docs/24                       #
#########################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.core.bathymetry import build_bathymetry           # noqa: E402
from libs.core.cases3d import build_case3d                  # noqa: E402
from libs.core.domain import Domain                         # noqa: E402
from libs.core.driver3d_v05 import (build_domain3d,          # noqa: E402
                                    physics_v05_from_config,
                                    simulate3d_v05)
from libs.core.eos import _rho_teos10, rho_prime            # noqa: E402
from libs.core.grid import CGrid                            # noqa: E402
from libs.core.model3d import Physics3D, Stepper3D          # noqa: E402
from libs.core.model3d_v05 import PhysicsV05, Stepper3DV05  # noqa: E402
from libs.core.schemes import SchemeParams                  # noqa: E402
from libs.utils.config import load_config                   # noqa: E402

TOL_REDUCE = 1e-12          # V5-1: v0.5 must reproduce the verified v0.4 core
TOL_EOS = 1e-4              # V5-4: against the published check value


def _rel(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Relative error normalised by the state scale when the reference field
    is identically zero - the trap that made the 3D gate report 1.0 on an
    exactly-zero eta (docs/90 N3)."""
    den = float(np.linalg.norm(b))
    if den < 1e-11 * scale:
        den = scale
    return float(np.linalg.norm(a - b) / den)


def v5_1_reduce(cfg) -> dict:
    """V5-1: on flat / z-level / periodic / linear EOS the v0.5 stepper must
    reproduce the v0.2-v0.4 stepper it generalises.

    Gated on the theta legs only. The split-explicit legs carry a deliberate
    difference since N15 (see below) and are reported, not gated.
    """
    nx, nz, dt, n_steps = 32, 12, 600.0, 20
    grid = CGrid(nx=nx, ny=nx, Lx=4.0e5, Ly=4.0e5, nz=nz, depth=1000.0)
    H, mask = build_bathymetry(grid, "flat", 1000.0)
    domain = Domain(grid=grid, H=H, mask=mask, vcoord="zlevel")
    out = {}
    for scheme in ("theta", "split_explicit"):
        for case_name, phys in (
                ("barotropic3d", Physics3D(g=9.80616, f=1e-4, H=1000.0)),
                ("baroclinic_igw", Physics3D(g=9.80616, f=1e-4, H=1000.0,
                                             N2=1e-4, nu=1e-3, kappa=1e-3))):
            params = SchemeParams(name=scheme, theta=0.5, theta_cor=0.5,
                                  n_picard=2, solver="pcg_jacobi",
                                  rtol=1e-14, max_iter=5000)
            case, _ = build_case3d(case_name, grid, phys, cfg)
            s0 = case.initial()
            a = Stepper3D(grid, phys, params, dt)
            b = Stepper3DV05(domain, PhysicsV05(base=phys), params, dt)
            sa = sb = s0
            for _ in range(n_steps):
                sa, sb = a.step(sa), b.step(sb)
            scale = max(float(np.linalg.norm(sa.u)), 1e-30)
            err = max(_rel(getattr(sb, f), getattr(sa, f), scale)
                      for f in ("u", "v", "b", "eta"))
            out[f"{scheme}/{case_name}"] = err
    # Not every leg is an identity any more. N15 changed how v0.5 removes the
    # Coriolis term from the barotropic forcing - v0.4 takes the depth integral
    # of the 3D term, v0.5 applies the SAME 2D transport operator the barotropic
    # step adds back - and v0.4 was deliberately left on the old (wrong over
    # topography) form. So the split-explicit legs differ BY DESIGN and gating
    # them at 1e-12 would only re-report a decision already made. They are
    # measured and reported; the theta legs stay identities (docs/90 N35).
    identity = {k: v for k, v in out.items() if not k.startswith("split_explicit/")}
    known_diff = {k: v for k, v in out.items() if k.startswith("split_explicit/")}
    return {"name": "V5-1 reduce_v04", "tol": TOL_REDUCE,
            "worst": max(identity.values()), "detail": out,
            "known_difference_split_explicit": known_diff,
            "pass": max(identity.values()) < TOL_REDUCE}


def v5_4_eos() -> dict:
    """V5-4: polyTEOS10-bsq against the published check value, and the three
    EOS levels against each other in a regime where they must nearly agree."""
    check = float(_rho_teos10(np.array([10.0]), np.array([30.0]),
                              np.array([1000.0]))[0])
    published = 1027.45140
    err = abs(check - published) / published
    # Near the reference point the three levels must agree to ~1 kg/m3.
    T, S, z = np.array([10.0]), np.array([35.0]), np.array([0.0])
    lin = float(rho_prime("linear", T, S, z)[0])
    seos = float(rho_prime("seos", T, S, z)[0])
    return {"name": "V5-4 eos_check", "tol": TOL_EOS, "worst": err,
            "detail": {"teos10_check": check, "published": published,
                       "linear_anomaly_at_ref": lin, "seos_anomaly_at_ref": seos},
            "pass": err < TOL_EOS}


def v5_2_operator_order(cfg) -> dict:
    """V5-2: truncation order of the variable-coefficient operator.

    The full manufactured-solution machinery would need source-term hooks in
    the stepper. What actually needs verifying is narrower and can be measured
    directly: does the discrete D[K G[.]] converge to the continuous
    div(H grad eta) at second order over a SMOOTH but non-uniform H? A
    constant-coefficient test cannot see the error made by evaluating H at
    faces, which is the only new approximation topography introduces.

        H(x,y)   = H0 (1 + a sin(2 pi x/L) sin(2 pi y/L))
        eta(x,y) = cos(2 pi m x/L) cos(2 pi n y/L)

    both periodic and analytic, so the exact divergence is closed form.

    The gate is on the SECOND-ORDER rules, which is what verifies the operator
    itself. The production "min" rule's order is measured and reported rather
    than gated: it is first order by construction, and that is the result.
    """
    from libs.core.operators_masked import div_m, gradx_u_m, grady_v_m

    Lx = 4.0e5
    H0, a, m, n = 1000.0, 0.3, 2.0, 1.0
    kx = ky = 2.0 * np.pi / Lx
    sizes = (64, 128, 256, 512, 1024)

    def measure(rule: str) -> list[float]:
        out = []
        for nx in sizes:
            grid = CGrid(nx=nx, ny=nx, Lx=Lx, Ly=Lx, nz=1, depth=H0)
            one = np.ones(grid.shape)
            xe, ye = grid.coords("eta")
            H = H0 * (1.0 + a * np.sin(kx * xe) * np.sin(ky * ye))
            eta = np.cos(m * kx * xe) * np.cos(n * ky * ye)
            He = np.roll(H, -1, axis=-1)
            Hn = np.roll(H, -1, axis=-2)
            if rule == "min":
                Ku, Kv = np.minimum(H, He), np.minimum(H, Hn)
            elif rule == "mean":
                Ku, Kv = 0.5 * (H + He), 0.5 * (H + Hn)
            else:
                Ku, Kv = 2.0 * H * He / (H + He), 2.0 * H * Hn / (H + Hn)
            dHdx = H0 * a * kx * np.cos(kx * xe) * np.sin(ky * ye)
            dHdy = H0 * a * ky * np.sin(kx * xe) * np.cos(ky * ye)
            dedx = -m * kx * np.sin(m * kx * xe) * np.cos(n * ky * ye)
            dedy = -n * ky * np.cos(m * kx * xe) * np.sin(n * ky * ye)
            lap = -((m * kx) ** 2 + (n * ky) ** 2) * eta
            exact = dHdx * dedx + dHdy * dedy + H * lap
            num = div_m(Ku * gradx_u_m(eta, grid, one),
                        Kv * grady_v_m(eta, grid, one), grid, one)
            out.append(float(np.sqrt(np.mean((num - exact) ** 2))
                             / np.sqrt(np.mean(exact ** 2))))
        return out

    rows, orders = [], {}
    for rule in ("min", "mean", "harmonic"):
        e = measure(rule)
        orders[rule] = float(np.log2(e[-2] / e[-1]))
        for nx, v in zip(sizes, e):
            rows.append({"face_rule": rule, "nx": nx, "l2_rel": v})
    worst_second_order = min(orders["mean"], orders["harmonic"])
    return {"name": "V5-2 operator_order", "tol": 1.9,
            "worst": worst_second_order, "detail": rows,
            "orders": orders, "pass": worst_second_order > 1.9}


def v5_3_seamount(cfg, nx: int, nz: int, steps: int) -> dict:
    """V5-3: the resting seamount test across the three vertical coordinates
    and both pressure-gradient formulations. There is no pass/fail threshold
    in the literature - the number itself is the result."""
    base = cfg.with_overrides({"solver.kind": "pcg_jacobi",
                               "physics3d.N2": 0.0, "physics3d_v05.pgf": "keep"})
    rows = []
    for topo in ("flat", "seamount", "rough"):
        for vcoord in ("zlevel", "zstar", "sigma"):
            for corr in (True, False):
                c = base.with_overrides({"physics3d_v05.pgf_correction": corr})
                d = build_domain3d(c, nx, nz, vcoord=vcoord, bathymetry=topo)
                r = simulate3d_v05(c, "seamount_rest", d,
                                   physics_v05_from_config(c), 2.0,
                                   n_steps_override=steps)
                rows.append({"topo": topo, "vcoord": vcoord,
                             "pgf_correction": corr, "rx0": r["rx0"],
                             "max_u_cm_s": r["metric"],
                             "rms_u_cm_s": r["extra"]["rms_speed_cm_s"]})
    # The flat rows are identically zero by construction (no topography, no
    # horizontal buoyancy gradient), so a gate that looked only at them could
    # not fail - and the rows that carry the Beckmann-Haidvogel signal sat
    # outside it (docs/90 N34). Both halves are judged now:
    #   flat     -> must be machine zero (a regression guard)
    #   seamount -> the pressure-gradient CORRECTION must buy at least an order
    #               of magnitude over the uncorrected operator on the same grid
    flat = [r for r in rows if r["topo"] == "flat"]
    worst_flat = max(r["max_u_cm_s"] for r in flat) if flat else float("inf")
    ok_flat = bool(flat) and worst_flat < 1e-8
    gain = {}
    for vc in sorted({r["vcoord"] for r in rows if r["topo"] != "flat"}):
        on = [r for r in rows if r["topo"] != "flat" and r["vcoord"] == vc and r["pgf_correction"]]
        off = [r for r in rows if r["topo"] != "flat" and r["vcoord"] == vc and not r["pgf_correction"]]
        if on and off:
            a = max(r["max_u_cm_s"] for r in on)
            b = max(r["max_u_cm_s"] for r in off)
            gain[vc] = (b / a) if a > 0 else float("inf")
    ok_gain = bool(gain) and all(g >= 10.0 for g in gain.values())
    return {"name": "V5-3 seamount_rest", "tol": 1e-8,
            "worst": worst_flat, "detail": rows,
            "pgf_correction_gain": gain,
            "pass": bool(ok_flat and ok_gain)}


def v5_6_constancy(cfg, nx: int, nz: int, steps: int) -> dict:
    """V5-6: a uniform tracer under a moving free surface."""
    base = cfg.with_overrides({"solver.kind": "pcg_jacobi",
                               "physics3d.N2": 0.0, "physics3d.nu": 0.0,
                               "physics3d.kappa": 0.0,
                               "physics3d_v04.tracers": "TS",
                               "scheme.advection": "centered2"})
    rows = []
    for vcoord in ("zlevel", "zstar", "sigma"):
        d = build_domain3d(base, nx, nz, vcoord=vcoord, bathymetry="slope",
                           slope=0.5)
        r = simulate3d_v05(base, "zstar_constancy", d,
                           physics_v05_from_config(base), 2.0,
                           n_steps_override=steps)
        rows.append({"vcoord": vcoord, "spread": r["metric"],
                     "T_mean": r["extra"]["T_mean"],
                     "tracer_integral_drift":
                         r["extra"]["tracer_integral_drift"],
                     # the resting-thickness number is the one that invented a
                     # fake leak (docs/90 N31); the live one is the verdict
                     "tracer_integral_drift_live":
                         r["extra"].get("tracer_integral_drift_live")})
    worst = max(r["spread"] for r in rows)
    return {"name": "V5-6 zstar_constancy", "tol": 1e-9, "worst": worst,
            "detail": rows, "pass": worst < 1e-9}


def v5_5_flather(cfg, nx: int, nz: int) -> dict:
    """V5-5: reflection from the open boundary, against the closed-wall
    control which must reflect essentially everything."""
    base = cfg.with_overrides({"solver.kind": "pcg_jacobi",
                               "physics3d.N2": 0.0, "physics3d.nu": 0.0,
                               "physics3d.kappa": 0.0, "physics.f0": 0.0,
                               "scheme.name": "split_explicit"})
    rows = []
    for bc in ("closed", "open"):
        d = build_domain3d(base, nx, nz, vcoord="zlevel", bathymetry="flat",
                           bc=bc)
        r = simulate3d_v05(base, "flather_radiate", d,
                           physics_v05_from_config(base), 0.5)
        rows.append({"bc": bc, "reflected_fraction": r["metric"],
                     "n_steps": r["n_steps"]})
    op = [r for r in rows if r["bc"] == "open"][0]["reflected_fraction"]
    cl = [r for r in rows if r["bc"] == "closed"][0]["reflected_fraction"]
    return {"name": "V5-5 flather_radiate", "tol": 0.1,
            "worst": op, "detail": rows,
            "pass": op < 0.1 and op < 0.5 * cl}


def main() -> int:
    ap = argparse.ArgumentParser(description="spec v0.5 verification suite")
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config().with_overrides({"grid.Lx": 4.0e5, "grid.Ly": 4.0e5})

    wanted = set(args.only.split(",")) if args.only else None
    checks = {
        "v5_1": lambda: v5_1_reduce(cfg),
        "v5_2": lambda: v5_2_operator_order(cfg),
        "v5_3": lambda: v5_3_seamount(cfg, args.nx, args.nz, args.steps),
        "v5_4": v5_4_eos,
        "v5_5": lambda: v5_5_flather(cfg, args.nx, args.nz),
        "v5_6": lambda: v5_6_constancy(cfg, args.nx, args.nz, args.steps),
    }
    results = []
    for key, fn in checks.items():
        if wanted and key not in wanted:
            continue
        r = fn()
        results.append(r)
        verdict = "PASS" if r["pass"] else "FAIL"
        # V5-2 reports a convergence order, where larger is better; every
        # other case reports an error, where smaller is.
        rel = ">" if r["name"].startswith("V5-2") else "<"
        print(f"{verdict}  {r['name']:24s} worst={r['worst']:.4e} "
              f"tol {rel} {r['tol']:.1e}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"json: {args.json}")
    if not results:
        # `all([])` is True: a typo in --only used to exit 0 having run nothing.
        print("FATAL: no check ran - is --only naming a check that exists?")
        return 1
    if wanted:
        unknown = wanted - set(checks)
        if unknown:
            print(f"FATAL: unknown check(s): {', '.join(sorted(unknown))}")
            return 1
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
