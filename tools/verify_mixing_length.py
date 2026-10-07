#!/usr/bin/env python3
#########################################################################
#  Module: verify_mixing_length                                         #
#  Description: Verifies the two forms of the S11.2 mixing length        #
#               DIRECTLY, on a prescribed (e, N^2) column - no model, no #
#               time integration. V6-3 (Kato-Phillips deepening) shows   #
#               the closure is alive but cannot separate the two forms:  #
#               its +-25 % window is five times wider than the 5 %       #
#               difference they produce there (docs/90 N34). The axis    #
#               needs a test that can see it, and the place they differ  #
#               is the limiter-dominated surface layer, not the          #
#               stratified interior where both reduce to sqrt(2e/N^2).   #
#  Pipeline: closure.{gaspar,recursive}_lengths -> here -> docs/03 S11.2 #
#########################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.core.closure import (TKEParams, gaspar_lengths, interface_geometry,   # noqa: E402
                               recursive_lengths)


def column(nz: int, dz: float, e_surf: float, e_deep: float,
           n2_mix: float, n2_pyc: float, h_mix: float):
    """A prescribed column: strong turbulence over a weakly stratified mixed
    layer, weak turbulence under a stratified pycnocline."""
    shape = (nz, 1, 1)
    dz3 = np.full(shape, dz)
    mask3 = np.ones(shape)
    z = (np.arange(nz) + 0.5)[:, None, None] * dz
    n2c = np.where(z < h_mix, n2_mix, n2_pyc) * np.ones(shape)
    n2 = np.zeros((nz + 1, 1, 1))
    n2[1:nz] = 0.5 * (n2c[:-1] + n2c[1:])
    zi = (np.arange(nz + 1))[:, None, None] * dz
    e = np.where(zi < h_mix, e_surf, e_deep)
    zs, zb, wet, dzi = interface_geometry(dz3, mask3)
    return e, n2, dzi, zs, zb, wet


def lengths(e, n2, dzi, zs, zb, wet, p, form):
    up, dn = (recursive_lengths if form == "recursive" else gaspar_lengths)(
        e, n2, dzi, zs, zb, wet, p)
    return np.maximum(p.l_min, np.sqrt(up * dn)).ravel()


def main() -> int:
    ap = argparse.ArgumentParser(description="the S11.2 mixing-length axis, measured directly")
    ap.add_argument("--nz", type=int, default=60)
    ap.add_argument("--json", type=Path,
                    default=Path("expr/E16_mixing_length/data/mixing_length_checks.json"))
    args = ap.parse_args()
    p = TKEParams()
    res: dict = {"nz": args.nz}

    print("A. uniformly stratified interior: BOTH forms must give sqrt(2e/N^2)")
    dz, e0, n2_0 = 1.0, 1.0e-4, 4.0e-4
    nz = args.nz
    e, n2, dzi, zs, zb, wet = column(nz, dz, e0, e0, n2_0, n2_0, -1.0)
    analytic = np.sqrt(2.0 * e0 / n2_0)
    far = slice(nz // 3, 2 * nz // 3)            # away from both walls
    res["analytic"] = {}
    for form in ("integral", "recursive"):
        l = lengths(e, n2, dzi, zs, zb, wet, p, form)[far]
        err = float(np.max(np.abs(l - analytic)) / analytic)
        res["analytic"][form] = err
        print(f"   {form:10s} max rel. error vs sqrt(2e/N^2) = {err:.3e}")
    ok_analytic = all(v < 2e-2 for v in res["analytic"].values())

    print("\nB. limiter-dominated surface layer: the two forms must DIFFER")
    e, n2, dzi, zs, zb, wet = column(nz, 1.25, 2.0e-3, 1.0e-4, 1.0e-6, 4.0e-4, 10.0)
    li = lengths(e, n2, dzi, zs, zb, wet, p, "integral")
    lr = lengths(e, n2, dzi, zs, zb, wet, p, "recursive")
    m = wet.ravel() > 0
    rel = np.abs(li[m] - lr[m]) / np.maximum(li[m], 1e-12)
    res["difference"] = {"median": float(np.median(rel)), "max": float(rel.max()),
                         "n_over_10pct": int(np.sum(rel > 0.10))}
    print(f"   relative difference: median {np.median(rel) * 100:.1f} %, "
          f"max {rel.max() * 100:.1f} %, interfaces over 10 % = {res['difference']['n_over_10pct']}")
    ok_differ = rel.max() > 0.10 and res["difference"]["n_over_10pct"] >= 3

    print("\nC. both forms stay physical everywhere")
    res["bounds"] = {}
    for name, l in (("integral", li), ("recursive", lr)):
        ok = bool(np.all(l[m] >= p.l_min - 1e-15) and
                  np.all(l[m] <= (zs.ravel()[m] + zb.ravel()[m] + 2 * p.z_0) + 1e-9))
        res["bounds"][name] = ok
        print(f"   {name:10s} l_min <= l <= wall distance: {ok}")
    ok_bounds = all(res["bounds"].values())

    res["passed"] = bool(ok_analytic and ok_differ and ok_bounds)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nMIXING LENGTH {'PASS' if res['passed'] else 'FAIL'}   json: {args.json}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
