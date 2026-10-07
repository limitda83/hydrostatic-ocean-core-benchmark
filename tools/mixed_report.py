#!/usr/bin/env python3
#########################################################################
#  Module: mixed_report                                                 #
#  Description: E18 - puts the three precision variants of the same     #
#               CUDA core side by side: fp64, whole-fp32 (cuda_sp) and  #
#               mixed (fp64 core, fp32 closure). Two products: (1) the  #
#               closure's own cost per precision from the fb           #
#               decomposition (.tke - .v05), with the built-in sanity   #
#               check that .v05 - which has no closure - must agree     #
#               between mixed and fp64; (2) whole-step time from the    #
#               ladder rows. Every join pins host, case, scheme,        #
#               solver, cfl, nx, nz and n_steps (N44).                  #
#  Pipeline: decompose_physics.sh + tier2_sweep(.mixed) -> this -> docs #
#########################################################################

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PREC = {"cuda": "fp64", "cuda_sp": "fp32", "cuda_mixed": "mixed"}


def load(paths):
    for p in paths:
        with p.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    w = float(r["wall_s"]); n = int(r["n_steps"])
                except (KeyError, ValueError):
                    continue
                if w <= 0 or n <= 0 or r["backend"] not in PREC:
                    continue
                yield r, w, n


def decomposition(paths):
    runs = defaultdict(dict)
    for r, w, n in load(paths):
        if "." not in r["case"]:
            continue
        stem, tag = r["case"].split(".", 1)
        key = (r["host"].split(".")[0], stem, r["scheme"], r["solver"], float(r["cfl"]),
               int(r["nx"]), int(r["nz"]), n)
        runs[key][(PREC[r["backend"]], tag)] = (w, float(r["wall_mad_s"]))
    print("## 폐쇄 자체의 비용 — `.tke` − `.v05`, ms/스텝")
    print("| 장치 | 격자 | fp64 | fp32 | **mixed** | mixed/fp64 | fp32/fp64 | `.v05` mixed/fp64 (건전성, 1.00 이어야) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(runs):
        v = runs[key]; n = key[-1]
        def cl(p):
            if (p, "tke") in v and (p, "v05") in v:
                return 1e3 * (v[(p, "tke")][0] - v[(p, "v05")][0]) / n
            return None
        c64, c32, cmx = cl("fp64"), cl("fp32"), cl("mixed")
        if cmx is None:
            continue
        san = (v[("mixed", "v05")][0] / v[("fp64", "v05")][0]) if ("fp64", "v05") in v else float("nan")
        f = lambda x: "—" if x is None else f"{x:.2f}"
        print(f"| {key[0]} | {key[5]}²×{key[6]} | {f(c64)} | {f(c32)} | **{f(cmx)}** | "
              f"{'—' if c64 is None else f'{cmx/c64:.2f}×'} | {'—' if c64 is None or c32 is None else f'{c32/c64:.2f}×'} | "
              f"{san:.3f}" + (" ⚠" if abs(san - 1) > 0.03 else "") + " |")


def ladder(paths):
    rows = defaultdict(dict)
    for r, w, n in load(paths):
        key = (r["host"].split(".")[0], r["case"].split(".")[0], r["scheme"], r["solver"], float(r["cfl"]),
               int(r["nx"]), int(r["nz"]), n)
        rows[key][PREC[r["backend"]]] = (1e3 * w / n, 100 * float(r["wall_mad_s"]) / w, int(r["solver_iters"]))
    print("\n## 스텝 전체 — ms/스텝 (MAD %), 솔버 반복수")
    print("| 장치 | 문제 | 격자 | fp64 | fp32 | **mixed** | mixed/fp64 | 반복수 fp64/fp32/mixed |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for key in sorted(rows):
        v = rows[key]
        if "mixed" not in v:
            continue
        f = lambda p: "—" if p not in v else f"{v[p][0]:.2f} ({v[p][1]:.1f} %)"
        it = "/".join(str(v[p][2]) if p in v else "—" for p in ("fp64", "fp32", "mixed"))
        ratio = f"{v['mixed'][0]/v['fp64'][0]:.2f}×" if "fp64" in v else "—"
        print(f"| {key[0]} | {key[1]} {key[2]}/{key[3]} CFL {key[4]:g} | {key[5]}² | {f('fp64')} | {f('fp32')} | **{f('mixed')}** | {ratio} | {it} |")


def main() -> int:
    ap = argparse.ArgumentParser(description="E18 mixed-precision report")
    ap.add_argument("--decompose", type=Path, nargs="*", default=[])
    ap.add_argument("--ladder", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    if a.decompose:
        decomposition(a.decompose)
    if a.ladder:
        ladder(a.ladder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
