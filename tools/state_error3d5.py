#!/usr/bin/env python3
#########################################################################
#  Module: state_error3d5                                               #
#  Description: Relative L2 error between two v0.5 state binaries       #
#               (eta, u, v, b[, T, S]) at the same physical time - the  #
#               accuracy half of "time to solution at fixed error"      #
#               (R8) for the tier-2 sweeps of docs/04. The reference    #
#               is a converged run on the same grid, so the error       #
#               measured is the TIME discretisation error of the        #
#               configuration, which is what separates the schemes.     #
#  Pipeline: tier2_sweep.sh -> state_error3d5 -> CSV                    #
#########################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_backends3d import field_error          # noqa: E402
from tools.compare_backends3d5 import read_state          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="L2 error between two v0.5 states")
    ap.add_argument("state", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--ts", action="store_true", help="states carry T and S")
    args = ap.parse_args()
    a = read_state(args.state, args.nx, args.nx, args.nz, args.ts)
    r = read_state(args.reference, args.nx, args.nx, args.nz, args.ts)
    fields = ["eta", "u", "v", "b"] + (["T", "S"] if args.ts else [])
    scale = max(float(np.sqrt(np.sum(r[f] ** 2))) for f in fields)
    out = {f"l2_rel_{f}": field_error(a[f], r[f], scale) for f in fields}
    out["l2_rel_max"] = max(out.values())
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
