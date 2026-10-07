#########################################################################
#  Module: e19_table                                                     #
#  Description: Turn the E19 strong-scaling CSV(s) into the manuscript   #
#               table rows (LaTeX + Markdown): per grid and physics      #
#               level, the 1-card reference (plain single-device binary  #
#               measured in the same session), the N-card times, the     #
#               parallel speed-up, and the ratio against the EPYC node   #
#               baseline of the hardware ladder.                         #
#  Pipeline: expr/E19_multinode_scaling/data/*.csv -> stdout            #
#########################################################################
from __future__ import annotations
import argparse, csv, logging, math
from pathlib import Path

# EPYC 9655 best OpenMP configuration [ms/step], fb CFL 0.5, 50 steps (paper/caf Table 3)
EPYC_MS = {("DC", 400): 24.89, ("DC", 1000): 137.65, ("DC", 2000): 664.44,
           ("TKE", 400): 79.22, ("TKE", 1000): 390.74, ("TKE", 2000): 3016.88}


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def ms_per_step(r: dict) -> float:
    try:
        return 1e3 * float(r["wall_s"]) / float(r["n_steps"])
    except (ValueError, ZeroDivisionError):
        return math.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--gpu-label", default="RTX 5090")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = load(a.csv)
    bad = [r for r in rows if r["max_abs_diff_vs_1dev"] not in ("nan", "0.000e+00")]
    if bad:
        logging.error(f"{len(bad)} rows differ from the 1-device state - gate violated, refusing")
        return 1
    key = lambda r: (r["host"], r["physics"], int(r["nx"]))
    combos = sorted({key(r) for r in rows}, key=lambda k: (k[0], k[1] != "DC", k[2]))
    tex, md = [], []
    for host, phys, nx in combos:
        single = [r for r in rows if key(r) == (host, phys, nx) and r["devices"].startswith("single")]
        # the driver run on one device is an overhead check, not a reported column: skip n_devices == 1 non-single rows
        multi = sorted([r for r in rows if key(r) == (host, phys, nx) and not r["devices"].startswith("single") and int(r["n_devices"]) > 1],
                       key=lambda r: (int(r["n_devices"]), r["devices"]))
        t1 = ms_per_step(single[0]) if single and single[0]["status"] == "ok" else math.nan
        cells_tex, cells_md = [], []
        for r in multi:
            n = int(r["n_devices"])
            if r["status"] != "ok":
                cells_tex.append(f"{r['status']}"); cells_md.append(r["status"]); continue
            t = ms_per_step(r); sp = t1 / t if t1 == t1 else math.nan
            cells_tex.append(f"{t:.2f} ({sp:.2f})" if sp == sp else f"{t:.2f} (--)")
            cells_md.append(f"{t:.2f} ({sp:.2f})" if sp == sp else f"{t:.2f} (–)")
        best = min((ms_per_step(r) for r in multi + single if r["status"] == "ok"), default=math.nan)
        epyc = EPYC_MS.get((phys, nx), math.nan)
        vs = epyc / best if best == best else math.nan
        t1s = f"{t1:.2f}" if t1 == t1 else (single[0]["status"] if single else "--")
        labels = " / ".join(r["devices"].split(":")[-1] if r["devices"].startswith("mpi") else f"{r['n_devices']} cards" for r in multi)
        tex.append(f"{host} & ${nx}^2$ & {phys.replace('TKE', 'DC+TKE')} & {t1s} & " + " & ".join(cells_tex) + f" & {epyc:.2f} & {vs:.1f}\\\\  % {labels}")
        md.append(f"| {host} | ${nx}^2$ | {phys.replace('TKE', 'DC+TKE')} | {t1s} | " + " | ".join(cells_md) + f" | {epyc:.2f} | {vs:.1f} | {labels} |")
    print("% LaTeX rows: host & grid & physics & 1 card [ms] & N cards [ms] (speed-up vs 1 card) ... & EPYC best [ms] & best/EPYC")
    print("\n".join(tex)); print(); print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
