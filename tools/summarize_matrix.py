#!/usr/bin/env python3
#########################################################################
#  Module: summarize_matrix                                             #
#  Description: Turn the benchmark matrix CSV into the two tables the   #
#               report needs: per-backend cost across grid sizes        #
#               (Part A) and the error-vs-wallclock Pareto (Part B).    #
#               Non-finite entries are shown as DIVERGED rather than    #
#               silently dropped (RULES.md R12).                       #
#  Pipeline: bench_matrix.sh -> results.csv -> summarize_matrix         #
#########################################################################

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

COLUMNS = ["numpy_ref", "fortran_serial", "omp_1t", "omp_16t", "omp_32t",
           "openacc", "cuda"]
LABELS = {"numpy_ref": "NumPy", "fortran_serial": "F-serial", "omp_1t": "F-omp1",
          "omp_16t": "F-omp16", "omp_32t": "F-omp32", "openacc": "OpenACC",
          "cuda": "CUDA"}


def backend_key(row: dict[str, str]) -> str:
    if row["backend"] == "fortran_omp":
        return f"omp_{row['threads']}t"
    return row["backend"]


def as_float(value: str) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def part_a(rows: list[dict[str, str]]) -> None:
    table: dict[int, dict[str, str]] = defaultdict(dict)
    steps = ""
    for r in rows:
        table[int(r["nx"])][backend_key(r)] = r["wall_s"]
        steps = r["n_steps"]

    print(f"=== Part A: wall time [s] for {steps} steps, cfl=2.0, theta=0.5 ===")
    print(f"{'nx':>6}{'points':>10}" + "".join(f"{LABELS[c]:>11}" for c in COLUMNS))
    for nx in sorted(table):
        line = f"{nx:>6}{nx * nx:>10}"
        for c in COLUMNS:
            v = as_float(table[nx].get(c, ""))
            line += f"{v:>11.4f}" if v is not None else f"{'-':>11}"
        print(line)

    print(f"\n{'nx':>6}  speedup vs single-core serial CPU")
    print(f"{'':>6}" + "".join(f"{LABELS[c]:>11}" for c in COLUMNS))
    for nx in sorted(table):
        base = as_float(table[nx].get("fortran_serial", ""))
        line = f"{nx:>6}"
        for c in COLUMNS:
            v = as_float(table[nx].get(c, ""))
            line += f"{base / v:>10.2f}x" if (v and base) else f"{'-':>11}"
        print(line)


def part_b(rows: list[dict[str, str]]) -> None:
    table: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for r in rows:
        key = (r["scheme"], r["theta"], r["cfl"])
        table[key][backend_key(r)] = r["wall_s"]
        table[key]["l2"] = r["l2_rel_eta"]
        table[key]["steps"] = r["n_steps"]
        table[key]["pcg"] = r["pcg_iters"]

    nxs = sorted({r["nx"] for r in rows})
    print(f"\n=== Part B: error vs wall time, nx={','.join(nxs)}, "
          f"fixed physical time ===")
    print(f"{'scheme':>10}{'cfl':>7}{'steps':>7}{'L2(eta)':>13}"
          f"{'omp16 [s]':>12}{'CUDA [s]':>12}{'pcg/solve':>11}")
    for (scheme, theta, cfl) in sorted(table, key=lambda k: (k[0], k[1], float(k[2]))):
        e = table[(scheme, theta, cfl)]
        l2 = as_float(e.get("l2", ""))
        steps = int(e.get("steps", 0) or 0)
        pcg = as_float(e.get("pcg", "")) or 0.0
        cpu = as_float(e.get("omp_16t", ""))
        gpu = as_float(e.get("cuda", ""))
        name = f"{scheme}{theta}" if scheme == "theta" else scheme
        l2s = f"{l2:.4e}" if l2 is not None else "DIVERGED"
        print(f"{name:>10}{float(cfl):>7.1f}{steps:>7}{l2s:>13}"
              f"{cpu if cpu is not None else float('nan'):>12.4f}"
              f"{gpu if gpu is not None else float('nan'):>12.4f}"
              f"{pcg / max(1, steps * 2):>11.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="summarise the benchmark matrix")
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    a = [r for r in rows if r["part"] == "A"]
    b = [r for r in rows if r["part"] == "B"]
    if a:
        part_a(a)
    if b:
        part_b(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
