#!/usr/bin/env python3
#########################################################################
#  Module: performance_index                                            #
#  Description: Build the comprehensive performance index from the      #
#               measurement CSVs. The index is deliberately a           #
#               DECOMPOSITION rather than a single score:               #
#                                                                       #
#      time per step = iterations/step  x  cost per iteration           #
#                      \_ algorithm _/     \_ implementation+hardware _/#
#                                                                       #
#               The left factor depends on scheme, solver, topography   #
#               and time step and is identical on every machine; the    #
#               right factor depends on backend, hardware and grid      #
#               size and is identical for every physical problem. A     #
#               model developer can therefore combine a measured left   #
#               factor with a measured right factor to predict a        #
#               configuration nobody ran - which is the only form of    #
#               performance result that transfers.                      #
#  Pipeline: helm_matrix / eos_matrix CSVs -> performance_index -> docs #
#########################################################################

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Peak fp64 throughput and memory bandwidth, measured (docs/13) or from the
# vendor where measurement is impossible. Used only for roofline ratios.
HARDWARE = {
    "rtx5090": {"bw_gb_s": 1580.0, "fp64_tflops": 1.98, "kind": "gpu"},
    "h100":    {"bw_gb_s": 3350.0, "fp64_tflops": 34.0, "kind": "gpu"},
    "epyc9655": {"bw_gb_s": 614.0, "fp64_tflops": 5.5, "kind": "cpu"},
    "xeon8480": {"bw_gb_s": 614.0, "fp64_tflops": 7.2, "kind": "cpu"},
}

# fp64 array-passes per cell per iteration, counted kernel by kernel from
# libs/cuda/src/helmholtz_bench.cu. Neighbour reads are assumed to hit cache,
# which is what a 2D stencil on a modern cache does; the count is therefore a
# lower bound on traffic and the bandwidth fractions below are a lower bound
# on utilisation.
#
#   apply      read p, ku, kv, msk + write ap             5
#   dot(p,ap)  read p, ap                                 2
#   axpy2      read p, ap, x, r + write x, r              6
#   dot(r,r)   read r                                     1
#   jacobi     read dinv, r + write z                     3
#   dot(r,z)   read r, z                                  2
#   p update   read z, p + write p                        3
#                                                        --
#                                                        22
# The RBGS preconditioner replaces the 3-pass Jacobi with a fill plus four
# colour sweeps, each reading x, b, ku, kv, msk, dinv and writing x.
ARRAYS_PER_PCG_ITER = {"pcg_jacobi": 22, "pcg_rbgs": 22 - 3 + 1 + 4 * 7}


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        hardware = p.parent.name if p.name == "results.csv" else p.stem
        with open(p) as fh:
            for r in csv.DictReader(fh):
                r["_source"] = str(p)
                rows.append(r)
    return rows


def _f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def algorithmic_factor(rows: list[dict]) -> dict:
    """Iterations per solve as a function of (solver, topo, cfl, nx) only.

    Every backend must agree here - the same discrete system is being
    solved - so a spread larger than zero is a portability bug, and the
    spread is reported rather than averaged away.
    """
    seen = defaultdict(set)
    for r in rows:
        key = (r["solver"], r["topo"], r["cfl"], r["nx"])
        seen[key].add(int(r["iterations"]))
    out = {}
    for key, iters in sorted(seen.items()):
        out["|".join(key)] = {"iterations": sorted(iters)[0],
                              "backend_spread": len(iters) - 1,
                              "rx0": None}
    for r in rows:
        k = "|".join((r["solver"], r["topo"], r["cfl"], r["nx"]))
        if out[k]["rx0"] is None:
            out[k]["rx0"] = _f(r, "rx0")
    return out


def implementation_factor(rows: list[dict], hardware: str) -> dict:
    """Cost per iteration [ns] and the fraction of peak bandwidth it reaches.

    Dividing the wall time by the iteration count removes the problem's
    difficulty, leaving a number that describes only the backend and the
    machine. It is the factor that transfers between studies.
    """
    hw = HARDWARE.get(hardware, {})
    out = {}
    for r in rows:
        it = int(r["iterations"])
        if it <= 0:
            continue
        nx = int(r["nx"])
        cells = nx * nx
        ns = _f(r, "wall_s") / it * 1e9
        key = (r["backend"], r.get("threads", "0"), r["solver"], r["nx"])
        out.setdefault(key, []).append(ns)
    summary = {}
    for (backend, threads, solver, nx), vals in sorted(out.items()):
        cells = int(nx) ** 2
        ns = statistics.median(vals)
        arrays = ARRAYS_PER_PCG_ITER.get(solver, 24)
        bytes_moved = cells * arrays * 8
        gb_s = bytes_moved / (ns * 1e-9) / 1e9
        summary["|".join((backend, str(threads), solver, nx))] = {
            "ns_per_iteration": ns,
            "ns_per_cell_per_iteration": ns / cells,
            "achieved_gb_s": gb_s,
            "bandwidth_fraction": (gb_s / hw["bw_gb_s"]) if hw else None,
            "samples": len(vals),
        }
    return summary


def predict(alg: dict, impl: dict, solver: str, topo: str, cfl: str, nx: str,
            backend: str, threads: str = "0") -> float | None:
    """Time to one elliptic solve, predicted from the two factors."""
    a = alg.get("|".join((solver, topo, cfl, nx)))
    i = impl.get("|".join((backend, threads, solver, nx)))
    if not a or not i:
        return None
    return a["iterations"] * i["ns_per_iteration"] * 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description="comprehensive performance index")
    ap.add_argument("csv", nargs="+", type=Path,
                    help="helm_matrix results.csv files, one per node")
    ap.add_argument("--hardware", action="append", default=[],
                    metavar="PATH=NAME", help="map a csv path to a hardware key")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--validate", action="store_true",
                    help="check the prediction against every measured row")
    args = ap.parse_args()

    hw_of = {}
    for spec in args.hardware:
        path, _, name = spec.partition("=")
        hw_of[path] = name

    per_node = {}
    all_rows = []
    for p in args.csv:
        rows = read_rows([p])
        name = hw_of.get(str(p), p.parent.name)
        per_node[name] = rows
        all_rows.extend(rows)

    alg = algorithmic_factor(all_rows)
    bad = {k: v for k, v in alg.items() if v["backend_spread"]}
    print(f"algorithmic factor: {len(alg)} configurations, "
          f"{len(bad)} with a backend disagreement")
    for k, v in list(bad.items())[:10]:
        print(f"  DISAGREE {k}: spread {v['backend_spread']}")

    impl = {}
    for name, rows in per_node.items():
        impl[name] = implementation_factor(rows, name)
        print(f"implementation factor [{name}]: {len(impl[name])} entries")

    if args.validate:
        errs = []
        for name, rows in per_node.items():
            for r in rows:
                t = predict(alg, impl[name], r["solver"], r["topo"], r["cfl"],
                            r["nx"], r["backend"], r.get("threads", "0"))
                if t and _f(r, "wall_s") > 0:
                    errs.append(abs(t - _f(r, "wall_s")) / _f(r, "wall_s"))
        if errs:
            errs.sort()
            print(f"prediction error vs {len(errs)} measured rows: "
                  f"median {errs[len(errs)//2]:.3f}, "
                  f"p90 {errs[int(0.9*len(errs))]:.3f}, max {errs[-1]:.3f}")

    out = {"algorithmic": alg, "implementation": impl}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float))
        print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
