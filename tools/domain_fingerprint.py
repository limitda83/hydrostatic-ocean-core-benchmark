#!/usr/bin/env python3
#########################################################################
#  Module: domain_fingerprint                                           #
#  Description: Prints what the bathymetry generator produces for a     #
#               case on THIS machine - depth range and a hash of the    #
#               depth field - through the same path toml2nml uses, but  #
#               without writing a bundle. Run it on every node: a       #
#               byte-identical problem needs an identical fingerprint,  #
#               and the 2026-09-16 audit found the `rough` depth of     #
#               lock_exchange differing ~50x between two dates.          #
#  Pipeline: config -> bathymetry_from_config -> fingerprint            #
#########################################################################

from __future__ import annotations

import argparse
import hashlib
import platform
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from libs.core.bathymetry import bathymetry_from_config  # noqa: E402
from libs.core.grid import CGrid  # noqa: E402
from libs.utils.config import load_config  # noqa: E402
from main import _apply_overrides  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="lock_exchange")
    ap.add_argument("--nx", type=int, nargs="+", default=[400, 1000])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dt", type=float, default=None, help="if given, also print n_split by the stepper rule")
    a = ap.parse_args()
    config = _apply_overrides(load_config("config"), a.set)
    print(f"host={socket.gethostname()} {platform.machine()} numpy={np.__version__} case={a.case} "
          f"kind={config.get('bathymetry.kind')} seed={config.get('bathymetry.seed', 20260911)}")
    for nx in a.nx:
        grid = CGrid(nx=nx, ny=nx, Lx=float(config.get("grid.Lx")), Ly=float(config.get("grid.Ly")),
                     nz=30, depth=float(config.get("physics.H")))
        H, mask = bathymetry_from_config(grid, config)
        h = np.ascontiguousarray(H, dtype=np.float64)
        hmax = float(h[mask > 0].max())
        line = (f"nx={nx:5d}  hmin={h[mask > 0].min():12.6f}  hmax={hmax:12.6f}  "
                f"wet={int((mask > 0).sum())}  md5(h)={hashlib.md5(h.tobytes()).hexdigest()[:12]}")
        if a.dt is not None:   # mod_model3d_v05.f90 stepper5_init, verbatim
            c = np.sqrt(float(config.get("physics.g", 9.81)) * hmax)
            dt_baro = 1.0 / (c * np.sqrt(1.0 / grid.dx ** 2 + 1.0 / grid.dy ** 2))
            line += f"  dt_baro={dt_baro:.5f}  n_split(dt={a.dt})={max(1, int(np.ceil(1.2 * a.dt / dt_baro)))}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
