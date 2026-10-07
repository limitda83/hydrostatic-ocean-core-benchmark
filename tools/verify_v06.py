#!/usr/bin/env python3
#########################################################################
#  Module: verify_v06                                                   #
#  Description: Verification suite of spec v0.6 (docs/03 S11.4):        #
#               V6-1 order of the third-order advection on a passive     #
#               sinusoid carried by a uniform flow (centered2 -> 2,      #
#               upwind1 -> 1, up3 -> 3); V6-2 monotonicity of the TVD   #
#               limiter on a top hat (and the overshoot of the          #
#               unlimited scheme); V6-3 the wind-driven mixed layer of   #
#               the TKE closure against the Kato-Phillips scaling        #
#               h = 1.05 u_* sqrt(t/N). V6-4 (bitwise reduction to v0.5) #
#               is the R2 gate against the compiled v0.5 backends.        #
#  Pipeline: model3d_v05 (v0.6) -> verify_v06 -> output/v06_verification #
#########################################################################

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.core.driver3d_v05 import build_domain3d, physics_v05_from_config   # noqa: E402
from libs.core.model3d import State3D                                         # noqa: E402
from libs.core.model3d_v05 import Stepper3DV05                                # noqa: E402
from libs.core.schemes import scheme_params_from_config                       # noqa: E402
from libs.utils.config import load_config                                     # noqa: E402
from main import _apply_overrides                                             # noqa: E402

PASSIVE = ["bathymetry.kind=flat", "domain.bc_x=periodic", "domain.bc_y=periodic",
           "physics3d_v04.tracers=TS", "physics3d_v04.alpha_T=0.0", "physics3d_v04.beta_S=0.0",
           "physics.f0=0.0", "physics3d.N2=0.0", "physics3d.nu=0.0", "physics3d.kappa=0.0",
           "physics3d_v05.pgf=keep", "scheme.name=fb", "grid.Lx=1.0e5", "grid.Ly=1.0e4",
           "physics.H=100.0"]


def _setup(overrides: list[str], nx: int, ny_override: int | None, nz: int):
    cfg = _apply_overrides(load_config("config"), overrides)
    domain = build_domain3d(cfg, nx, nz)
    if ny_override is not None and domain.grid.ny != ny_override:
        # keep the aspect ratio square cells: Ly = Lx * ny / nx is set by the caller
        pass
    physics = physics_v05_from_config(cfg)
    params = scheme_params_from_config(cfg)
    return cfg, domain, physics, params


def _advect(scheme: str, nx: int, profile: str, courant: float = 0.2, crossings: float = 1.0):
    ny, nz = 4, 4
    ov = PASSIVE + [f"scheme.advection={scheme}", f"grid.Ly={1.0e5 * ny / nx:.6e}"]
    cfg, domain, physics, params = _setup(ov, nx, ny, nz)
    g = domain.grid
    u0 = 1.0
    dt = courant * g.dx / u0
    n_steps = int(round(crossings * g.Lx / u0 / dt))
    dt = crossings * g.Lx / u0 / n_steps
    st = Stepper3DV05(domain, physics, params, dt)
    x, _ = g.coords("eta")
    if profile == "sine":
        T0 = 10.0 + np.sin(2.0 * np.pi * x / g.Lx)
    else:
        T0 = 10.0 + np.where((x > 0.3 * g.Lx) & (x < 0.6 * g.Lx), 1.0, 0.0)
    T = np.broadcast_to(T0, g.shape3d).copy()
    s = State3D(u=np.full(g.shape3d, u0), v=np.zeros(g.shape3d), b=np.zeros(g.shape3d),
                eta=np.zeros(g.shape), t=0.0, T=T, S=np.full(g.shape3d, 35.0))
    s = st.integrate(s, n_steps)
    exact = T                                            # a whole number of crossings
    err = np.sqrt(np.mean((s.T - exact) ** 2)) / np.sqrt(np.mean((exact - 10.0) ** 2))
    return err, s.T, exact, n_steps


def v6_1(schemes=("centered2", "upwind1", "up3", "up3_tvd"), sizes=(32, 64, 128, 256)) -> dict:
    """Order of the discrete advection OPERATOR: DIV(u C) applied to a sinusoid
    carried by a uniform flow against the analytic u dC/dx. The time stepping
    of the tracer is forward Euler by spec, so an integrated test would show
    order 1 for every scheme; the operator test isolates the face interpolation."""
    out = {}
    ny, nz = 4, 4
    for sc in schemes:
        errs = []
        for n in sizes:
            ov = PASSIVE + [f"scheme.advection={sc}", f"grid.Ly={1.0e5 * ny / n:.6e}"]
            cfg, domain, physics, params = _setup(ov, n, ny, nz)
            g = domain.grid
            st = Stepper3DV05(domain, physics, params, 1.0)
            x, _ = g.coords("eta")
            k = 2.0 * np.pi / g.Lx
            c = np.broadcast_to(10.0 + np.sin(k * x), g.shape3d).copy()
            u = np.full(g.shape3d, 1.0)
            w = np.zeros((nz + 1,) + g.shape)
            div = st._tracer_advection(c, u, np.zeros_like(u), w)
            exact = np.broadcast_to(k * np.cos(k * x), g.shape3d)
            errs.append(float(np.sqrt(np.mean((div - exact) ** 2)) / np.sqrt(np.mean(exact ** 2))))
        orders = [float(np.log(errs[i - 1] / errs[i]) / np.log(sizes[i] / sizes[i - 1])) for i in range(1, len(sizes))]
        out[sc] = {"sizes": list(sizes), "errors": errs, "orders": orders}
        print(f"V6-1 {sc:10s} errors {['%.3e' % e for e in errs]} orders {['%.2f' % o for o in orders]}")
    passed = out["up3"]["orders"][-1] >= 2.8 and out["centered2"]["orders"][-1] >= 1.8 \
        and out["upwind1"]["orders"][-1] >= 0.8
    out["passed"] = bool(passed)
    return out


def v6_2(nx: int = 128) -> dict:
    """V6-2: monotonicity of the S11.3 reconstruction.

    The 2026-09-11 version advected a top-hat with the MODEL's forward-Euler
    tracer step and read the overshoot off the result. That number was not a
    property of the scheme: forward Euler with an upwind-biased third-order
    flux amplifies, and the measured overshoot grew with both the step count
    and the Courant number (C=0.05 -> 0.19, C=0.4 -> 3.3e+05). The test now
    delegates to `tools/verify_advection.py`, which exercises the SAME
    production reconstruction with the time integrator taken out of the
    question (docs/90 N34, N36).
    """
    import subprocess
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    out = root / "expr/E14_v06_representative/data/advection_checks.json"
    r = subprocess.run([sys.executable, str(root / "tools/verify_advection.py"),
                        "--nx", str(nx), "--json", str(out)],
                       cwd=root, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(root)})
    print(r.stdout.rstrip())
    if r.returncode != 0 and r.stderr:
        print(r.stderr.rstrip())
    data = json.loads(out.read_text()) if out.exists() else {"passed": False}
    return {"face": data.get("face"), "rk3": data.get("rk3"),
            "passed": bool(data.get("passed"))}


def v6_3(hours: float = 30.0, dt: float = 60.0, mxl: str = "integral") -> dict:
    nx, ny, nz, H = 4, 4, 50, 50.0
    tau = 0.1
    ov = ["bathymetry.kind=flat", "domain.bc_x=periodic", "domain.bc_y=periodic",
          "physics3d_v04.tracers=TS", "physics3d_v04.alpha_T=2.0e-4", "physics3d_v04.beta_S=0.0",
          "physics.f0=0.0", "physics3d.N2=0.0", "physics3d.nu=1.0e-5", "physics3d.kappa=1.0e-6",
          f"physics3d.tau_x={tau}", "physics3d_v05.pgf=keep", "physics3d_v05.eos=linear",
          "physics3d_v06.closure=tke", f"physics3d_v06.mxl={mxl}",
          "scheme.name=fb", "scheme.advection=none",
          f"physics.H={H}", "grid.Lx=4.0e3", "grid.Ly=4.0e3"]
    cfg, domain, physics, params = _setup(ov, nx, ny, nz)
    g = domain.grid
    N = 0.01
    dTdz = N ** 2 / (physics.g * physics.alpha_T)
    z = (np.arange(nz) + 0.5)[:, None, None] * g.dz      # depth below the surface, positive down
    T = 20.0 - dTdz * z * np.ones(g.shape3d)             # warm above cold: N^2 > 0
    st = Stepper3DV05(domain, physics, params, dt)
    s = State3D(u=np.zeros(g.shape3d), v=np.zeros(g.shape3d), b=np.zeros(g.shape3d),
                eta=np.zeros(g.shape), t=0.0, T=T, S=np.full(g.shape3d, 35.0))
    s = State3D(u=s.u, v=s.v, b=st.buoyancy_of(s.T, s.S), eta=s.eta, t=0.0, T=s.T, S=s.S)
    ustar = np.sqrt(tau / physics.rho0)
    n_steps = int(hours * 3600 / dt)
    rows = []
    dT_tot = dTdz * H
    for n in range(1, n_steps + 1):
        s = st.step(s)
        if n % int(3600 / dt) == 0:
            prof = s.T[:, 0, 0]
            # mixed-layer depth = the interface of maximum temperature gradient
            # (the entrainment interface of the laboratory experiment)
            k = int(np.argmin(np.diff(prof)))
            h = float((k + 1) * g.dz)
            t = n * dt
            h_kp = 1.05 * ustar * np.sqrt(t / N)
            rows.append({"hour": t / 3600, "h": h, "h_kp": float(h_kp), "ratio": h / h_kp})
    checked = [r for r in rows if 6.0 <= r["hour"] <= hours]
    ok = all(0.75 <= r["ratio"] <= 1.25 for r in checked)
    for r in rows[::3]:
        print(f"V6-3[{mxl}] t={r['hour']:5.1f} h  h={r['h']:6.2f} m  Kato-Phillips {r['h_kp']:6.2f} m  ratio {r['ratio']:.2f}")
    print(f"V6-3[{mxl}] {'PASS' if ok else 'FAIL'} (ratio within 0.75-1.25 for t = 6..{hours:g} h), "
          f"tridiagonal solves {st.tridiagonal_solves}, solver rebuilds {st.solver_rebuilds}")
    return {"rows": rows, "passed": bool(ok), "ustar": float(ustar), "N": N, "mxl": mxl}


def v6_4(nz: int = 60) -> dict:
    """V6-4: the S11.2 mixing-length axis, measured on the operator itself.

    V6-3 (Kato-Phillips) shows the closure is alive but cannot separate the two
    `mxl` forms - its +-25 % window is five times wider than the ~5 % they
    differ by in that experiment. The place they actually differ is the
    limiter-dominated surface layer, and that is measured directly here, with
    no model and no time integration (docs/90 N37).
    """
    import subprocess
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    out = root / "expr/E16_mixing_length/data/mixing_length_checks.json"
    r = subprocess.run([sys.executable, str(root / "tools/verify_mixing_length.py"),
                        "--nz", str(nz), "--json", str(out)],
                       cwd=root, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(root)})
    print(r.stdout.rstrip())
    if r.returncode != 0 and r.stderr:
        print(r.stderr.rstrip())
    data = json.loads(out.read_text()) if out.exists() else {"passed": False}
    return {**data, "passed": bool(data.get("passed"))}


def main() -> int:
    ap = argparse.ArgumentParser(description="spec v0.6 verification (V6-1..V6-3)")
    ap.add_argument("--only", default="v6_1,v6_2,v6_3,v6_4")
    ap.add_argument("--mxl", default="integral", help="V6-3 mixing length (S11.2 axis)")
    ap.add_argument("--json", type=Path, default=Path("output/v06_verification.json"))
    args = ap.parse_args()
    res = {}
    for name in args.only.split(","):
        res[name] = (v6_3(mxl=args.mxl) if name == "v6_3"
                     else {"v6_1": v6_1, "v6_2": v6_2, "v6_4": v6_4}[name]())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(res, indent=1, default=float))
    ok = all(v.get("passed", False) for v in res.values())
    print("V6 suite:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
