#!/usr/bin/env python3
#########################################################################
#  Module: verify_advection                                             #
#  Description: Monotonicity / dispersion of the spec S11.3 face         #
#               reconstruction, SEPARATED from the time integrator.      #
#               The old V6-2 advected a top-hat with the model's         #
#               forward-Euler tracer step and read the overshoot as a    #
#               property of the scheme; it is not. Forward Euler with an #
#               upwind-biased third-order flux is unconditionally        #
#               amplifying, so that overshoot grows with the number of   #
#               steps (0.16 -> 1.65) and with the Courant number         #
#               (docs/90 N34). Here the same production reconstruction   #
#               is tested three ways, none of which can be contaminated  #
#               by the time scheme.                                      #
#  Pipeline: advection_v06.kappa_face -> verify_advection -> docs/03 S11 #
#########################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.core.advection_v06 import kappa_face                      # noqa: E402


def _faces(c: np.ndarray, limited: bool, vel_sign: float = 1.0,
           wet: np.ndarray | None = None, axis: int = -1) -> np.ndarray:
    """Production face values on a 1-D row.

    `wet` and `axis` are exposed because the model ALWAYS passes a real mask and
    uses both horizontal axes; a test that only ever calls this with wet=None on
    axis -1 leaves the wall-fallback branch and the y-operator unexercised
    (docs/90 N38).
    """
    shape = (1, 1, -1) if axis == -1 else (1, -1, 1)
    c3 = c.reshape(shape)
    vel = np.full_like(c3, vel_sign)
    w3 = None if wet is None else wet.reshape(shape).astype(float)
    return kappa_face(c3, vel, axis, w3, limited).ravel()


def face_boundedness(c: np.ndarray, limited: bool) -> dict:
    """A monotone reconstruction puts the face value BETWEEN its two cells.

    This is the definition of the limiter and involves no time stepping at
    all: no dt, no integrator, nothing to amplify. `f[i]` sits on the face
    between `c[i]` and `c[i+1]`.
    """
    out = {}
    for sign in (1.0, -1.0):
        f = _faces(c, limited, sign)
        lo = np.minimum(c, np.roll(c, -1))
        hi = np.maximum(c, np.roll(c, -1))
        span = float(hi.max() - lo.min()) or 1.0
        # A NaN face must be a violation, not a skipped entry: np.max ignores
        # nothing but Python's max() does, and that is exactly the bug this
        # project already fixed once in the R2 comparator (docs/90 N34). Do not
        # let the same hole back in through the test that guards against it.
        bad = ~np.isfinite(f)
        over = float("inf") if bad.any() else float(np.max(np.maximum(f - hi, 0.0))) / span
        under = float("inf") if bad.any() else float(np.max(np.maximum(lo - f, 0.0))) / span
        out["upwind" if sign > 0 else "downwind"] = {
            "over": over, "under": under, "n_nonfinite": int(bad.sum())}
    return out


def ssp_rk3_advect(c0: np.ndarray, limited: bool, courant: float,
                   crossings: float, sign: float = 1.0) -> tuple[np.ndarray, int]:
    """Constant-velocity periodic advection with SSP-RK3 in time.

    SSP-RK3 is a convex combination of forward-Euler stages, so it cannot
    amplify what a forward-Euler step at the same effective CFL would keep
    bounded. Any overshoot left is the SPATIAL reconstruction.
    """
    nx = c0.size
    dx = 1.0 / nx
    dt = courant * dx                        # u = 1
    n_steps = max(1, int(round(crossings / dt)))
    dt = crossings / n_steps

    def tend(c: np.ndarray) -> np.ndarray:
        # flux at face i is u * f[i]; the sign must reach BOTH the reconstruction
        # (which picks f_pos or f_neg) and the flux itself, or the downwind
        # branch of the production scheme is never evolved (docs/90 N38).
        flux = sign * _faces(c, limited, sign)
        return -(flux - np.roll(flux, 1)) / dx

    c = c0.copy()
    for _ in range(n_steps):
        c1 = c + dt * tend(c)
        c2 = 0.75 * c + 0.25 * (c1 + dt * tend(c1))
        c = (c + 2.0 * (c2 + dt * tend(c2))) / 3.0
    return c, n_steps


def main() -> int:
    ap = argparse.ArgumentParser(
        description="monotonicity of the S11.3 reconstruction, free of the time scheme")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--courant", type=float, default=0.4)
    ap.add_argument("--json", type=Path,
                    default=Path("expr/E14_v06_representative/data/advection_checks.json"))
    args = ap.parse_args()

    x = (np.arange(args.nx) + 0.5) / args.nx
    top = np.where((x > 0.3) & (x < 0.6), 1.0, 0.0)
    res: dict = {"nx": args.nx, "courant": args.courant}

    print("A0. the limiter FORMULA, on a profile with non-zero slope ratios")
    # A binary top-hat gives r = 0 or 0/0 everywhere, so Superbee's non-trivial
    # branches never run and the test cannot tell the limiter from plain
    # first-order upwind (verified: swapping in upwind1 passed). A monotone ramp
    # has r = 1 in its interior, where Superbee must reproduce the SECOND-ORDER
    # face value, not the upwind one.
    ramp = np.clip((x - 0.25) / 0.5, 0.0, 1.0)
    f_lim = _faces(ramp, True, 1.0)
    up1 = ramp                                   # first-order upwind face, u > 0
    interior = (x > 0.30) & (x < 0.70)
    d = np.roll(ramp, -1) - ramp
    second = ramp + 0.5 * d                      # r == 1 -> phi == 1 -> centred
    err_second = float(np.max(np.abs(f_lim - second)[interior]))
    gap_upwind = float(np.max(np.abs(f_lim - up1)[interior]))
    res["limiter_formula"] = {"err_vs_second_order": err_second,
                              "distance_from_upwind1": gap_upwind}
    print(f"   |f_limited - second order| = {err_second:.3e}   "
          f"|f_limited - upwind1| = {gap_upwind:.3e}")
    ok_formula = err_second < 1e-12 and gap_upwind > 1e-3
    print(f"   -> on a smooth ramp the limiter is second order, NOT upwind1: {ok_formula}")

    print("\nA. face-value boundedness - the limiter's definition, no time stepping")
    res["face"] = {}
    for name, lim in (("up3", False), ("up3_tvd", True)):
        b = face_boundedness(top, lim)
        res["face"][name] = b
        worst = (max(v["over"] for v in b.values()),
                 max(v["under"] for v in b.values()))
        print(f"   {name:8s} over {worst[0]:.3e}  under {worst[1]:.3e}")
    tvd_bounded = max(max(v["over"], v["under"])
                      for v in res["face"]["up3_tvd"].values()) <= 1e-14
    up3_unbounded = max(max(v["over"], v["under"])
                        for v in res["face"]["up3"].values()) > 1e-6
    print(f"   -> limited stays between its cells: {tvd_bounded}; "
          f"unlimited does not: {up3_unbounded}")

    print("\nA1. the wall fallback - the branch the model always takes, and this "
          "test never did")
    # Production calls kappa_face with a real mask: where the FAR stencil cell is
    # dry the face must fall back to first-order upwind (f_pos -> c, f_neg -> cp).
    # Every earlier call here passed wet=None, so a reversed condition, a wrong
    # roll offset or a wrong fallback cell would all have gone unnoticed.
    prof = 10.0 + np.sin(2.0 * np.pi * x) + 0.3 * np.cos(6.0 * np.pi * x)
    wet = np.ones_like(x)
    wet[20:24] = 0.0                              # a wall of four dry cells
    res["wall"] = {}
    ok_wall = True
    for sign, name, far_roll, fallback in ((1.0, "upwind", 1, prof),
                                           (-1.0, "downwind", -2, np.roll(prof, -1))):
        f_mask = _faces(prof, False, sign, wet)
        f_open = _faces(prof, False, sign, None)
        far_dry = np.roll(wet, far_roll) <= 0.0
        # where the far cell is dry: exactly the first-order value
        err_dry = float(np.max(np.abs(f_mask - fallback)[far_dry])) if far_dry.any() else 0.0
        # where it is wet: exactly the unmasked high-order value
        err_wet = float(np.max(np.abs(f_mask - f_open)[~far_dry]))
        # and the fallback must actually CHANGE something, or the test is empty
        moved = float(np.max(np.abs(f_mask - f_open)[far_dry])) if far_dry.any() else 0.0
        res["wall"][name] = {"n_dry_faces": int(far_dry.sum()), "err_at_dry": err_dry,
                             "err_at_wet": err_wet, "distance_from_open": moved}
        good = far_dry.sum() >= 3 and err_dry < 1e-14 and err_wet < 1e-14 and moved > 1e-3
        ok_wall = ok_wall and good
        print(f"   {name:9s} dry faces={int(far_dry.sum())}  "
              f"|f - upwind1| at dry = {err_dry:.2e}  "
              f"|f - unmasked| at wet = {err_wet:.2e}  fallback moved {moved:.2e}")
    print(f"   -> the far-cell-dry fallback is exactly first-order upwind: {ok_wall}")

    print("\nA2. the y operator - the model uses both axes, this test used one")
    fx = _faces(prof, False, 1.0, wet, axis=-1)
    fy = _faces(prof, False, 1.0, wet, axis=-2)
    err_axis = float(np.max(np.abs(fx - fy)))
    res["axis"] = {"max_abs_difference": err_axis}
    ok_axis = err_axis < 1e-14
    print(f"   |faces(axis=-1) - faces(axis=-2)| = {err_axis:.2e}  -> identical: {ok_axis}")

    print("\nB0. the hand-written SSP-RK3 must actually transport, at speed 1")
    # A no-op integrator (wrong coefficients, dt collapsing to 0) satisfies every
    # extremum criterion below - verified - so the integrator is checked against
    # a case with a known answer first: half a crossing must move a smooth sine
    # by exactly half a domain, i.e. negate it.
    sine = np.sin(2.0 * np.pi * x)
    half, nsteps = ssp_rk3_advect(sine, True, args.courant, 0.5)
    shifted = -sine                              # exact solution after 0.5 crossing
    amp = float(np.max(np.abs(half)))
    err_shift = float(np.max(np.abs(half - shifted))) / float(np.max(np.abs(sine)))
    res["transport"] = {"steps": nsteps, "amplitude": amp, "rel_error_vs_exact": err_shift}
    print(f"   steps={nsteps}  amplitude={amp:.4f} (1.0 = no damping)  "
          f"rel. error vs exact half-crossing = {err_shift:.3e}")
    ok_transport = err_shift < 0.05 and amp > 0.9
    print(f"   -> the integrator moves the field the right distance: {ok_transport}")

    print("\nB. SSP-RK3 in time - overshoot must not grow with the distance travelled")
    res["rk3"] = {}
    for name, lim in (("up3", False), ("up3_tvd", True)):
        row = []
        for cr in (1.0, 2.0, 4.0, 8.0):
            c, n = ssp_rk3_advect(top, lim, args.courant, cr)
            row.append({"crossings": cr, "steps": n,
                        "max": float(c.max()), "min": float(c.min())})
            print(f"   {name:8s} crossings={cr:4.1f} steps={n:5d}  "
                  f"max={c.max():+.4f}  min={c.min():+.4f}")
        res["rk3"][name] = row
    print("\nB1. order of accuracy on SMOOTH data, several resolutions")
    # Extrema of a top-hat at one resolution say nothing about accuracy. On a
    # smooth field the unlimited kappa scheme must converge at third order and
    # the limited one must not be better than first at the extrema it clips.
    res["order"] = {}
    for name, lim in (("up3", False), ("up3_tvd", True)):
        errs, ns = [], (32, 64, 128, 256)
        for n in ns:
            xx = (np.arange(n) + 0.5) / n
            s0 = np.sin(2.0 * np.pi * xx)
            c, _ = ssp_rk3_advect(s0, lim, args.courant, 1.0)
            errs.append(float(np.sqrt(np.mean((c - s0) ** 2))))
        ords = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
        res["order"][name] = {"nx": list(ns), "l2": errs, "orders": ords}
        print(f"   {name:8s} L2 = {['%.2e' % e for e in errs]}  orders = "
              f"{['%.2f' % o for o in ords]}")
    ok_order = res["order"]["up3"]["orders"][-1] > 2.6
    print(f"   -> unlimited converges at third order: {ok_order}")

    print("\nB2. negative velocity - reflection equivariance")
    # The earlier version reversed the profile and advected RIGHT, which is
    # advecting LEFT in disguise; for a top-hat that is not centred in the domain
    # the two land in different places and the check could never pass. The real
    # statement is equivariance: reflecting space AND the velocity must commute
    # with the scheme. That evolves the f_neg branch, which nothing else here did.
    reflect = lambda a: a[::-1].copy()                                # noqa: E731
    r_plus, _ = ssp_rk3_advect(top, False, args.courant, 0.25, +1.0)
    r_minus, _ = ssp_rk3_advect(reflect(top), False, args.courant, 0.25, -1.0)
    err_mirror = float(np.max(np.abs(r_plus - reflect(r_minus))))
    # and the negative velocity must actually transport, the other way
    res["mirror"] = {"max_abs_difference": err_mirror}
    ok_mirror = err_mirror < 1e-12
    print(f"   |R(+u) - reflect(R(-u) on reflected data)| = {err_mirror:.2e} "
          f"-> equivariant: {ok_mirror}")

    print("\nB3. a NON-integer crossing, judged against the exact solution")
    # Integer crossings return the profile to its start, so a scheme that does
    # not transport at all lands on the right answer by accident. A quarter
    # crossing of a smooth sine has a known exact answer everywhere.
    nq = 256
    xq = (np.arange(nq) + 0.5) / nq
    s0 = np.sin(2.0 * np.pi * xq)
    res["quarter"] = {}
    ok_quarter = True
    for sgn, tag in ((+1.0, "u>0"), (-1.0, "u<0")):
        got, _ = ssp_rk3_advect(s0, False, args.courant, 0.25, sgn)
        exact = np.sin(2.0 * np.pi * (xq - sgn * 0.25))
        rel = float(np.sqrt(np.mean((got - exact) ** 2)) / np.sqrt(np.mean(exact ** 2)))
        frozen = float(np.sqrt(np.mean((s0 - exact) ** 2)) / np.sqrt(np.mean(exact ** 2)))
        res["quarter"][tag] = {"rel_l2": rel, "rel_l2_if_frozen": frozen}
        ok_quarter = ok_quarter and rel < 0.02 and rel < 0.1 * frozen
        print(f"   {tag}  rel L2 vs exact = {rel:.3e}   (a frozen field would score "
              f"{frozen:.3e})")
    print(f"   -> transports the right distance in BOTH directions: {ok_quarter}")

    o = [r["max"] - 1.0 for r in res["rk3"]["up3"]]
    u = [-(r["min"]) for r in res["rk3"]["up3"]]
    tvd = res["rk3"]["up3_tvd"]
    tvd_mono = all(r["max"] <= 1.0 + 1e-12 and r["min"] >= -1e-12 for r in tvd)
    # Saturation, stated so it cannot be met by doing nothing. The old form
    # `o[-1] < 4*max(o[0],1e-12)` was satisfied by an integrator that froze the
    # field (all overshoots 0, 0 < 4e-12). Require BOTH: a real overshoot exists
    # at every distance, and it does not grow. Undershoot is checked too - an
    # amplification with a negative sign was invisible before.
    up3_present = min(o) > 1e-3 and min(u) > 1e-3
    up3_no_growth = o[-1] <= 1.05 * max(o) and u[-1] <= 1.05 * max(u)
    up3_saturates = bool(up3_present and up3_no_growth)
    res["saturation"] = {"over": o, "under": u,
                         "present": bool(up3_present), "no_growth": bool(up3_no_growth)}
    print(f"   -> limited creates no new extremum at any distance: {tvd_mono}")
    print(f"   -> unlimited overshoot saturates ({o[0]:.3f} -> {o[-1]:.3f}), "
          f"i.e. dispersion not amplification: {up3_saturates}")

    res["passed"] = bool(ok_formula and ok_wall and ok_axis and ok_transport
                         and ok_order and ok_mirror and ok_quarter and tvd_bounded
                         and up3_unbounded and tvd_mono and up3_saturates)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nADVECTION {'PASS' if res['passed'] else 'FAIL'}   json: {args.json}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
