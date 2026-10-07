#!/usr/bin/env python3
#########################################################################
#  Module: write_helmholtz_case                                         #
#  Description: Write a variable-coefficient free-surface Helmholtz     #
#               problem (spec S10.5) to plain binary, so that every     #
#               backend solves the IDENTICAL discrete system. The       #
#               bathymetry generator stays in one language: a spectral  #
#               field reimplemented in Fortran and CUDA would be three  #
#               chances to disagree, and R2 would have no ground truth. #
#  Pipeline: bathymetry/domain -> write_helmholtz_case -> backends      #
#########################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.core.bathymetry import build_bathymetry      # noqa: E402
from libs.core.domain import Domain                    # noqa: E402
from libs.core.grid import CGrid                       # noqa: E402
from libs.core.solvers_var import HelmholtzVar         # noqa: E402
from libs.utils.config import load_config              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="write a Helmholtz benchmark case")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nz", type=int, default=30)
    ap.add_argument("--topo", default="rough")
    ap.add_argument("--r-target", type=float, default=0.1)
    ap.add_argument("--vcoord", default="zlevel")
    ap.add_argument("--bc", default="periodic")
    ap.add_argument("--cfl", type=float, default=2.0)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--n-repeat", type=int, default=5,
                    help="timed repeats (R7-2 minimum 5)")
    ap.add_argument("--n-warmup", type=int, default=1,
                    help="discarded warm-up runs")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--Lx", type=float, default=0.0,
                    help="domain size [m]; 0 => config grid.Lx")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", action="store_true",
                    help="also solve it here and store the reference solution")
    ap.add_argument("--skip-numpy-solvers", action="store_true",
                    help="store the reference but skip the per-solver NumPy "
                         "iteration counts. They are only a cross-check of what "
                         "the compiled backends report, and rbgs at nx=1024 "
                         "takes tens of thousands of NumPy iterations.")
    args = ap.parse_args()

    cfg = load_config()
    Lx = args.Lx if args.Lx > 0.0 else float(cfg.get("grid.Lx"))
    H0 = float(cfg.get("physics.H"))
    g = float(cfg.get("physics.g"))
    grid = CGrid(nx=args.nx, ny=args.nx, Lx=Lx, Ly=Lx, nz=args.nz, depth=H0)
    H, mask = build_bathymetry(grid, args.topo, H0, r_target=args.r_target,
                               seed=args.seed)
    dom = Domain(grid=grid, H=H, mask=mask, vcoord=args.vcoord,
                 bc_x=args.bc, bc_y=args.bc)

    # The Helmholtz coefficient at the same CFL the model would run at.
    c_max = np.sqrt(g * float(np.max(H[mask > 0])))
    dt_baro = 1.0 / (c_max * np.sqrt(1.0 / grid.dx**2 + 1.0 / grid.dy**2))
    dt = args.cfl * dt_baro
    coef = g * args.theta**2 * dt**2

    # Depth-integrated face coefficients; with nu = 0 the effective depth is
    # the water column itself, which is the configuration the benchmark uses.
    Ku = np.sum(dom.face_thickness(dom.dz3_ref)[0], axis=0)
    Kv = np.sum(dom.face_thickness(dom.dz3_ref)[1], axis=0)

    rng = np.random.default_rng(args.seed)
    rhs = rng.standard_normal(grid.shape) * mask
    rhs -= rhs.mean()
    rhs *= mask

    args.out.mkdir(parents=True, exist_ok=True)
    # Fortran (nx,ny) column-major has the same byte order as NumPy [ny,nx].
    for name, arr in (("ku", Ku), ("kv", Kv), ("mask", mask), ("rhs", rhs),
                      ("depth", H)):
        arr.astype(np.float64).tofile(args.out / f"{name}.bin")

    meta = {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "dx": grid.dx,
            "dy": grid.dy, "coef": coef, "dt": dt, "cfl": args.cfl,
            "theta": args.theta, "rtol": args.rtol, "max_iter": args.max_iter,
            "topo": args.topo, "r_target": args.r_target, "vcoord": args.vcoord,
            "bc": args.bc, **dom.summary()}

    if args.reference:
        sol, rep = HelmholtzVar(grid, coef, Ku, Kv, mask, kind="pcg_jacobi",
                                rtol=min(args.rtol, 1e-13),
                                max_iter=args.max_iter).solve(rhs)
        sol.astype(np.float64).tofile(args.out / "eta_ref.bin")
        meta["reference_iterations"] = rep.iterations
        meta["reference_converged"] = rep.converged
        for kind in ([] if args.skip_numpy_solvers
                     else ("pcg_jacobi", "pcg_rbgs", "rbgs", "multigrid")):
            _, r = HelmholtzVar(grid, coef, Ku, Kv, mask, kind=kind,
                                rtol=args.rtol, max_iter=args.max_iter).solve(rhs)
            meta[f"numpy_iters_{kind}"] = r.iterations
            meta[f"numpy_converged_{kind}"] = r.converged

    (args.out / "case.json").write_text(json.dumps(meta, indent=2, default=float))
    # A namelist so the Fortran backend needs no JSON parser. Fortran wants
    # a 'd' exponent for a double literal; '%.17e' round-trips exactly.
    def fd(value: float) -> str:
        return f"{float(value):.17e}".replace("e", "d")

    with open(args.out / "case.nml", "w") as fh:
        fh.write("&helmholtz\n")
        fh.write(f"  nx = {grid.nx}\n  ny = {grid.ny}\n")
        fh.write(f"  dx = {fd(grid.dx)}\n  dy = {fd(grid.dy)}\n")
        fh.write(f"  coef = {fd(coef)}\n")
        fh.write(f"  rtol = {fd(args.rtol)}\n")
        fh.write(f"  max_iter = {args.max_iter}\n")
        # R7-2 asks for at least five repeats; a contended shared node needs
        # more before the median settles (docs/90 N28), so it is a knob.
        fh.write(f"  n_repeat = {args.n_repeat}\n")
        fh.write(f"  n_warmup = {args.n_warmup}\n")
        fh.write(f"  datadir = '{args.out.resolve()}'\n")
        fh.write("/\n")
    print(json.dumps(meta, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
