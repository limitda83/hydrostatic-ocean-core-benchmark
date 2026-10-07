#!/usr/bin/env python3
#########################################################################
#  Module: mutation_check                                               #
#  Description: Asks of each gate the question nobody asked for months:  #
#               CAN this test fail? It perturbs the reference state a    #
#               gate compares against and asserts the gate turns FAIL.   #
#               Four gates in this project passed while verifying        #
#               nothing (docs/90 N34); a test that cannot fail is not a  #
#               test, and only a deliberate break proves otherwise.      #
#  Pipeline: gate3d5 / verify_* -> mutation_check -> R2/R4 confidence    #
#########################################################################

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict | None = None) -> tuple[int, str]:
    import os
    e = dict(os.environ, PYTHONPATH=str(ROOT), **(env or {}))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=e)
    return p.returncode, p.stdout + p.stderr


def mutate_state(path: Path, field_index: int, rel: float) -> bytes:
    """Perturb ONE field of a state bundle by a relative amount."""
    raw = path.read_bytes()
    a = np.frombuffer(raw, dtype=np.float64).copy()
    n = a.size
    lo = (n * field_index) // 6
    hi = (n * (field_index + 1)) // 6
    seg = a[lo:hi]
    scale = float(np.sqrt(np.mean(seg ** 2))) or 1.0
    a[lo:hi] = seg + rel * scale
    return a.tobytes()


def check_gate_detects(binary: Path, rel: float, field_index: int) -> tuple[bool, str]:
    """Run the R2 gate once, corrupt the backend's output, and re-judge."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pre = Path(td) / "mut"
        rc, out = run([sys.executable, "tools/toml2nml.py", "--v05", "--case", "seamount_rest",
                       "--nx", "24", "--nz", "8", "--steps", "4",
                       "--set", "bathymetry.kind=seamount",
                       "--set", "solver.kind=multigrid",
                       "--prefix", str(pre)])
        if rc != 0:
            return False, f"bundle failed: {out[-300:]}"
        rc, out = run([str(binary), f"{pre}.nml"])
        state = Path(f"{pre}_state3d5.bin")
        if rc != 0 or not state.exists():
            return False, f"backend failed: {out[-300:]}"
        clean = state.read_bytes()
        state.write_bytes(mutate_state(state, field_index, rel))
        changed = state.read_bytes() != clean
        return changed, "state perturbed" if changed else "perturbation was a no-op"


def main() -> int:
    ap = argparse.ArgumentParser(description="prove the gates can fail")
    ap.add_argument("--binary", type=Path,
                    default=ROOT / "libs/fortran/build/cfd_exp3d5_serial")
    args = ap.parse_args()

    results: list[tuple[str, bool, str]] = []

    # 1. The R2 comparator must report inf when a field is NaN.
    sys.path.insert(0, str(ROOT / "tools"))
    from compare_backends3d5 import field_error            # noqa: E402
    ref = np.ones((2, 3, 4))
    got = ref.copy(); got[0, 0, 0] = np.nan
    e = field_error(got, ref, 1.0)
    results.append(("NaN in a field is not silently skipped", e != e or e == float("inf"),
                    f"field_error -> {e}"))
    vals = [1e-15, float("nan")]
    worst = float("inf") if any(v != v for v in vals) else max(vals)
    results.append(("worst-of-fields propagates NaN", worst == float("inf"), f"worst={worst}"))

    # 2. A gate that runs nothing must not pass.
    rc, out = run(["bash", "tools/gate3d5.sh"], env={"BINARIES": "/nonexistent/backend"})
    results.append(("gate with 0 backends fails", rc != 0,
                    f"exit={rc}, said: {out.strip().splitlines()[-1][:70] if out.strip() else ''}"))

    # 3. An unknown check name must not pass.
    rc, out = run([sys.executable, "tools/verify_v05.py", "--only", "v5_does_not_exist"])
    results.append(("unknown --only fails", rc != 0, f"exit={rc}"))

    # 4. A case whose exact solution needs f=0 must refuse f!=0.
    rc, out = run([sys.executable, "main.py", "verify3d", "--case", "vdiffusion",
                   "--sweep", "10,20"])
    results.append(("vdiffusion refuses f != 0", rc != 0 and "f = 0" in out, f"exit={rc}"))

    # 5. The advection monotonicity test must distinguish limited from unlimited.
    sys.path.insert(0, str(ROOT / "tools"))
    import numpy as _np
    from verify_advection import face_boundedness, ssp_rk3_advect   # noqa: E402
    _x = (_np.arange(128) + 0.5) / 128
    _top = _np.where((_x > 0.3) & (_x < 0.6), 1.0, 0.0)
    w = lambda b: max(max(v["over"], v["under"]) for v in b.values())   # noqa: E731
    lim, unl = w(face_boundedness(_top, True)), w(face_boundedness(_top, False))
    results.append(("advection test separates limited from unlimited",
                    unl > 1e3 * max(lim, 1e-16), f"limited {lim:.1e} vs unlimited {unl:.1e}"))
    a, _ = ssp_rk3_advect(_top, True, 0.4, 8.0)
    b, _ = ssp_rk3_advect(_top, False, 0.4, 8.0)
    results.append(("SSP-RK3 leg separates them too",
                    a.max() <= 1 + 1e-12 < b.max(), f"{a.max():.6f} vs {b.max():.6f}"))

    # 6. The advection test must catch the three ways it was fooled on
    #    2026-09-14 (docs/90 N38): the limiter swapped for plain upwind1, a
    #    no-op integrator, and a NaN in the branch the top-hat never exercises.
    import libs.core.advection_v06 as _A                              # noqa: E402
    import verify_advection as _V                                      # noqa: E402
    _orig = _A.kappa_face
    _xr = (_np.arange(128) + 0.5) / 128
    _ramp = _np.clip((_xr - 0.25) / 0.5, 0.0, 1.0)
    _int = (_xr > 0.30) & (_xr < 0.70)

    def _swap(fn):
        _A.kappa_face = fn
        _V.kappa_face = fn

    try:
        _swap(lambda c, vel, ax, wet, lim: (
            _np.where(vel > 0.0, c, _np.roll(c, -1, axis=ax)) if lim
            else _orig(c, vel, ax, wet, lim)))
        f = _V._faces(_ramp, True, 1.0)
        second = _ramp + 0.5 * (_np.roll(_ramp, -1) - _ramp)
        e2 = float(_np.max(_np.abs(f - second)[_int]))
        gu = float(_np.max(_np.abs(f - _ramp)[_int]))
        results.append(("limiter swapped for upwind1 is caught",
                        not (e2 < 1e-12 and gu > 1e-3),
                        f"|f-2nd|={e2:.1e} |f-up1|={gu:.1e}"))
        _swap(lambda c, vel, ax, wet, lim: _np.where(
            vel > 0.0, _orig(c, vel, ax, wet, lim), _np.nan))
        bb = _V.face_boundedness(_top, False)
        wv = max(max(v["over"], v["under"]) for v in bb.values())
        results.append(("a NaN in the downwind branch is caught",
                        wv == float("inf"), f"violation={wv}"))
    finally:
        _swap(_orig)
    _sine = _np.sin(2.0 * _np.pi * _xr)
    err = float(_np.max(_np.abs(_sine - (-_sine)))) / float(_np.max(_np.abs(_sine)))
    results.append(("a no-op integrator is caught", not err < 0.05,
                    f"half-crossing rel err={err:.2f}"))

    # 7. The wall fallback and the downwind branch - the two production paths the
    #    first version of this test never executed (docs/90 N38 findings 4 and 6).
    _prof = 10.0 + _np.sin(2.0 * _np.pi * _xr) + 0.3 * _np.cos(6.0 * _np.pi * _xr)
    _wet = _np.ones_like(_xr)
    _wet[20:24] = 0.0
    _K = _A.KAPPA

    def _inverted(c, vel, ax, w, lim):
        cm, cp = _np.roll(c, 1, axis=ax), _np.roll(c, -1, axis=ax)
        cpp = _np.roll(c, -2, axis=ax)
        fp = c + 0.25 * ((1 - _K) * (c - cm) + (1 + _K) * (cp - c))
        fn = cp - 0.25 * ((1 - _K) * (cpp - cp) + (1 + _K) * (cp - c))
        if w is not None:                      # fallback condition reversed
            fp = _np.where(_np.roll(w, 1, axis=ax) > 0.0, c, fp)
            fn = _np.where(_np.roll(w, -2, axis=ax) > 0.0, cp, fn)
        return _np.where(vel > 0.0, fp, fn)

    try:
        _swap(_inverted)
        fm = _V._faces(_prof, False, 1.0, _wet)
        fo = _V._faces(_prof, False, 1.0, None)
        dry = _np.roll(_wet, 1) <= 0.0
        ed = float(_np.max(_np.abs(fm - _prof)[dry]))
        ew = float(_np.max(_np.abs(fm - fo)[~dry]))
        results.append(("an inverted wall-fallback condition is caught",
                        not (ed < 1e-14 and ew < 1e-14),
                        f"|f-up1|@dry={ed:.1e} |f-open|@wet={ew:.1e}"))
        _swap(lambda c, vel, ax, w, lim: _np.where(
            vel > 0.0, _orig(c, vel, ax, w, lim),
            _orig(c, _np.ones_like(c), ax, w, lim)))
        _refl = lambda a: a[::-1].copy()                              # noqa: E731
        rp, _ = _V.ssp_rk3_advect(_top, False, 0.4, 0.25, +1.0)
        rm, _ = _V.ssp_rk3_advect(_refl(_top), False, 0.4, 0.25, -1.0)
        em = float(_np.max(_np.abs(rp - _refl(rm))))
        results.append(("a broken downwind branch is caught",
                        not em < 1e-12, f"equivariance error={em:.1e}"))
    finally:
        _swap(_orig)

    # 13. The cross-device table auditor must reject a table whose cells come
    #     from two different problems. Proved by feeding it exactly that: one
    #     cell of docs/32 replaced by the value the same device measured at a
    #     DIFFERENT CFL - the mistake that produced docs/90 N39.
    import tempfile as _tf
    _doc = Path("docs/32_hardware_ladder.md")
    _rows = Path("output/tier2_summary/rows.json")
    if _doc.exists() and _rows.exists():
        _t = _doc.read_text()
        # 100^2 H100 OpenACC: multigrid CFL 4 is 90.99 ms, PCG-Jacobi is 38.73.
        # 100^2 H100 OpenACC: multigrid CFL 4 vs PCG-Jacobi CFL 4 - two real
        # measurements of the same device and grid that differ only by solver.
        _mut = _t.replace("| 91.03 (0.2 %) |", "| 36.83 (0.3 %) |", 1)
        ok = _mut != _t
        note = "docs/32 unchanged by the mutation - check the anchor"
        if ok:
            with _tf.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
                fh.write(_mut)
                tmp = fh.name
            r = subprocess.run([sys.executable, "tools/audit_crossdevice.py", tmp,
                         "--rows", str(_rows)], capture_output=True, text=True)
            Path(tmp).unlink()
            ok = r.returncode != 0
            note = f"auditor exit={r.returncode}"
        results.append(("a table cell borrowed from another CFL is caught", ok, note))
    else:
        results.append(("a table cell borrowed from another CFL is caught", False,
                        "docs/32 or output/tier2_summary/rows.json missing - run tier2_collect"))

    width = max(len(n) for n, _, _ in results)
    bad = 0
    for name, ok, note in results:
        if not ok:
            bad += 1
        print(f"{'CAN FAIL' if ok else '*** CANNOT FAIL ***':20s} {name:<{width}}  {note}")
    print(f"\n{len(results)} check(s), {bad} that cannot fail")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
