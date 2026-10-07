#!/usr/bin/env python3
#########################################################################
#  Module: module_boundary_report                                       #
#  Description: Joins the two halves of E17 - the closure's own cost    #
#               (decomposition runs, docs/31 S3c: `.tke` minus `.v05`)  #
#               and the cost of moving that closure's data through the  #
#               host (module_boundary). Rewritten 2026-09-15 after the  #
#               external review (docs/90 N44): the old join keyed on     #
#               (host, precision, nx, nz) only, took the device from the #
#               FILENAME, and let a static-every-step run overwrite the  #
#               ordinary one. Every axis now travels with the row, the   #
#               device comes from the CSV, and a closure difference that #
#               is not resolved above its own MAD is refused.            #
#  Pipeline: decompose_physics.sh + module_boundary -> this -> docs/38   #
#########################################################################

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PREC_OF = {"cuda": "fp64", "cuda_sp": "fp32", "openacc": "fp64", "openacc_sp": "fp32"}
# The GPU name the CUDA runtime reports -> the host label the decomposition CSVs use.
HOST_OF_GPU = {"NVIDIA GeForce RTX 5090": "gpgpu", "NVIDIA H100 80GB HBM3": "ktcloud",
               "NVIDIA H100": "ktcloud"}
DEVICE = {"gpgpu": "RTX 5090", "ktcloud": "H100"}
CASE = "C_host_to_dev_roundtrip_pinned"      # the one comparison E17 exists for


def closure_cost(paths: list[Path]) -> dict[tuple, dict]:
    """ms/step of the TKE closure = wall(.tke) - wall(.v05), keyed on EVERY axis
    of the run: host, precision, case stem, scheme, solver, cfl, nx, nz, n_steps.
    Two rows pair only if they agree on all of them."""
    runs: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for p in paths:
        for r in csv.DictReader(p.open()):
            try:
                wall = float(r["wall_s"]); mad = float(r["wall_mad_s"]); n = int(r["n_steps"])
            except (KeyError, ValueError):
                continue
            if wall <= 0 or n <= 0 or "." not in r["case"]:
                continue
            stem, tag = r["case"].split(".", 1)
            key = (r["host"].split(".")[0], PREC_OF.get(r["backend"], r["backend"]), stem,
                   r["scheme"], r["solver"], float(r["cfl"]), int(r["nx"]), int(r["nz"]), n)
            runs[key][tag] = {"wall": wall, "mad": mad}
    out = {}
    for key, v in runs.items():
        if "v05" not in v or "tke" not in v:
            continue
        n = key[-1]
        diff = v["tke"]["wall"] - v["v05"]["wall"]
        # MAD of a difference of two independent medians, conservatively.
        unc = v["tke"]["mad"] + v["v05"]["mad"]
        out[key] = {"ms": 1e3 * diff / n, "unc_ms": 1e3 * unc / n,
                    "resolved": diff > 0 and diff > 2.0 * unc}
    return out


def boundary(paths: list[Path]) -> dict[tuple, dict]:
    """median seconds of the host round trip, keyed (host, precision, nx, nz,
    static_every_step). The host is read from the CSV's `gpu` column - never
    from the file name."""
    out: dict[tuple, dict] = {}
    for p in paths:
        for r in csv.DictReader(p.open()):
            if r["case"] != CASE:
                continue
            host = HOST_OF_GPU.get(r["gpu"])
            if host is None:
                raise SystemExit(f"{p}: unknown GPU '{r['gpu']}' - add it to HOST_OF_GPU")
            prec = "fp64" if int(r["itemsize"]) == 8 else "fp32"
            key = (host, prec, int(r["nx"]), int(r["nz"]), int(r["static_every_step"]))
            if key in out:
                raise SystemExit(f"{p}: duplicate boundary row for {key} - refusing to guess")
            out[key] = {"s": float(r["median_s"]), "mad_s": float(r["mad_s"]),
                        "bytes": int(r["bytes_in"]) + int(r["bytes_out"]), "gpu": r["gpu"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E17 report: host round trip vs closure cost")
    ap.add_argument("--decompose", type=Path, nargs="+", required=True)
    ap.add_argument("--boundary", type=Path, nargs="+", required=True)
    ap.add_argument("--scheme", default="fb", help="closure denominator from this scheme only")
    ap.add_argument("--static", type=int, default=0, choices=(0, 1),
                    help="which boundary variant to report (0: static arrays uploaded once)")
    args = ap.parse_args()

    cc = closure_cost(args.decompose)
    bb = boundary(args.boundary)
    if not cc:
        print("WARNING: no (.v05, .tke) pair found - the join is wrong, not the data.")
    if not bb:
        print("WARNING: no boundary rows parsed.")

    print(f"| device | precision | grid | closure in place [ms/step] (±) | "
          f"host round trip [ms/step] | **round trip / closure** |")
    print("|---|---|---:|---:|---:|---:|")
    rows = []
    for bkey in sorted(bb, key=lambda k: (k[0], k[1], k[2])):
        host, prec, nx, nz, static = bkey
        if static != args.static:
            continue
        cands = [(k, v) for k, v in cc.items()
                 if k[0] == host and k[1] == prec and k[6] == nx and k[7] == nz and k[3] == args.scheme]
        if len(cands) > 1:
            raise SystemExit(f"closure cost for {host} {prec} {nx}² is ambiguous across "
                             f"{[c[0][2:6] for c in cands]} - pass one decomposition set")
        c_ms = 1e3 * bb[bkey]["s"]
        if not cands:
            print(f"| {DEVICE[host]} | {prec} | {nx}²×{nz} | — | {c_ms:.2f} | — |")
            continue
        ck, cv = cands[0]
        if not cv["resolved"]:
            print(f"| {DEVICE[host]} | {prec} | {nx}²×{nz} | **미해상** ({cv['ms']:.2f} ± {cv['unc_ms']:.2f}) | "
                  f"{c_ms:.2f} | **폐기** |")
            continue
        print(f"| {DEVICE[host]} | {prec} | {nx}²×{nz} | {cv['ms']:.2f} (±{cv['unc_ms']:.2f}) | "
              f"{c_ms:.2f} | **{c_ms / cv['ms']:.1f}×** |")
        rows.append((host, prec, nx, cv["ms"], c_ms))

    print("\n### 어디서부터 경계가 폐쇄보다 비싼가")
    by = defaultdict(list)
    for host, prec, nx, cl, c_ms in rows:
        by[(host, prec)].append((nx, c_ms / cl))
    for (host, prec), v in sorted(by.items()):
        v.sort()
        trend = " → ".join(f"{nx}²: {r:.1f}×" for nx, r in v)
        over = [nx for nx, r in v if r > 1.0]
        verdict = f"**{over[0]}² 부터**" if over else "측정한 격자 안에서는 없음"
        print(f"- **{DEVICE[host]} {prec}** — {trend} → {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
