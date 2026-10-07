#!/usr/bin/env python3
#########################################################################
#  Module: tier2_collect                                                #
#  Description: Merges the tier-2 CSVs of the three nodes (docs/04 S6   #
#               step 7) into the two products of the design: the        #
#               PARETO PLANE of piece S (error against the converged    #
#               reference vs simulated seconds per wall second, one     #
#               point per scheme x solver x CFL x case, GPU backends at  #
#               400^2 x 30) and the HARDWARE LADDER of piece H (cost     #
#               per step for backend x device x grid on the byte-       #
#               identical 50-step problem). Speed-ups follow R8-1 and   #
#               R7: the baseline is the best CPU configuration OF THE    #
#               GPU'S OWN NODE when that node measured one, else it is   #
#               named as a cross-node baseline. Every ratio names both   #
#               ends and `ktcloud` CPU time is never one of them.        #
#  Pipeline: tier2_sweep.sh (3 nodes) -> tier2_collect -> docs/30       #
#########################################################################

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

NUM = ["threads", "theta", "cfl", "nx", "nz", "n_steps", "dt", "wall_s", "wall_mad_s",
       "solver_iters", "substeps", "l2_eta", "l2_u", "l2_max", "sim_s_per_wall_s"]
DEVICE = {"gpgpu": "RTX 5090", "ktcloud": "H100", "geo85": "EPYC 9655"}
# A host maps to its GPU above; its CPU is a DIFFERENT device and R8-1 requires
# the baseline to name the part that actually ran. Labelling `gpgpu`'s 16-thread
# CPU rows "RTX 5090" would put the card's name on the Threadripper's time.
CPU_DEVICE = {"gpgpu": "Threadripper PRO 9955WX", "ktcloud": "Xeon Platinum 8480+",
              "geo85": "EPYC 9655"}
CPU_BACKENDS = ("fortran_serial", "fortran_omp", "fortran_serial_sp", "fortran_omp_sp")


def precision_of(backend: str) -> str:
    """R9: fp32 is a separate axis and must not share a column with fp64."""
    return "fp32" if backend.endswith("_sp") else "fp64"
# A re-measurement carries a sweep TAG on the case token so both campaigns can
# live in one directory. `.r2` and `.stab` are RE-MEASUREMENTS of rows this
# project had already flagged as contaminated - not new cases. Keying on the
# raw token put them in a group of their own, so the clean values never reached
# the tables and docs/32 went on quoting the rows the project had rejected
# (docs/90 N39).
# `.accr`: the H100 OpenACC column, re-measured 2026-09-15 with the binary that
# finally passed a gate on that node (docs/90 N42). Like the others, it must
# REPLACE the row it re-measured, not sit beside it (R13-4).
REMEASURE_TAGS = (".r2", ".stab", ".accr")


def canon_case(case: str) -> str:
    for t in REMEASURE_TAGS:
        if case.endswith(t):
            return case[: -len(t)]
    return case


def is_remeasure(case: str) -> bool:
    return case.endswith(REMEASURE_TAGS)


def canon_host(host: str) -> str:
    """The node identity of a PBS compute node is a manifest detail, not an axis.

    `geo85` hands out node02..node05, all the same 96C EPYC 9655 under
    `place=excl`, and the two campaigns did not land on the same one. Keying on
    the reported hostname stopped a re-measurement from ever meeting the row it
    re-measured.
    """
    h = str(host).split(".")[0]
    if h == "geo85" or h.startswith("node"):
        return "geo85"
    if h.startswith("localhost"):
        return "gpgpu"
    return h


SCHEME_LABEL = {"fb": "forward-backward", "theta0.5": "theta 0.5", "theta0.6": "theta 0.6",
                "theta1.0": "theta 1.0", "split": "split-explicit"}


def load(paths: list[Path]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    dupes: list = []
    failed: list = []
    for p in paths:
        with p.open() as f:
            for r in csv.DictReader(f):
                for k in NUM:
                    try:
                        r[k] = float(r[k])
                    except (KeyError, ValueError):
                        r[k] = float("nan")
                if r["wall_s"] != r["wall_s"] or r["wall_s"] <= 0:
                    # A run that produced no time is a RESULT (OOM at 2000^2 on a
                    # 32 GB card, a backend that refused the problem). Dropping it
                    # made the configuration look as if it had never been tried.
                    r["failed"] = True
                    failed.append((r["host"], r["backend"], r["case"], r["scheme"],
                                   int(r["nx"]), int(r["nz"])))
                    continue
                # A run that hit max_iter (2000) on average every step blew up before
                # the divergence guard existed (docs/90 N18/N19): its time is not a
                # cost per step. Flag it as diverged (l2 = inf) instead of dropping it.
                if r["n_steps"] > 0 and r["solver_iters"] / r["n_steps"] >= 2000.0:
                    r["l2_max"] = r["l2_eta"] = r["l2_u"] = float("inf")
                    r["diverged"] = True
                # EVERY axis of the matrix goes in the key. `nz` was missing, and
                # the mxl experiment varies it (15/30/60): 84 rows were silently
                # overwritten, taking two of the three nz points of docs/36 S2-1
                # with them (docs/90 N33).
                # The re-measurement campaign renamed the case and landed on a
                # different PBS node; canonicalise both BEFORE the key is built
                # so a clean row meets the contaminated row it replaces.
                r["remeasured"] = is_remeasure(r["case"])
                r["case"] = canon_case(r["case"])
                r["host"] = canon_host(r["host"])
                key = (r["host"], r["backend"], int(r["threads"]), r["case"], r["scheme"],
                       r["solver"], r["cfl"], int(r["nx"]), int(r["nz"]), int(r["n_steps"]))
                prev = rows.get(key)
                if prev is not None:
                    dupes.append(key)
                    # Provenance beats file order: a re-measurement is never
                    # overwritten by the original run it superseded, whatever
                    # order the files were passed in.
                    if prev.get("remeasured") and not r["remeasured"]:
                        continue
                    if prev.get("remeasured") == r["remeasured"]:
                        # Two equally valid measurements of the same problem (the
                        # ladder sweep and the mxl sweep both cover some points).
                        # "Later file wins" made the published number depend on
                        # ARGV ORDER - the H100 CUDA cell moved 2.6 % when an
                        # unrelated file was added. Contention and interference
                        # can only make a run slower (N26), so take the minimum:
                        # deterministic, and the same rule shared_node_min uses.
                        if prev["wall_s"] <= r["wall_s"]:
                            continue
                rows[key] = r
    if failed:
        print(f"note: {len(failed)} run(s) produced no time and are reported as "
              f"failures, not omissions:")
        for f in sorted(set(failed)):
            print(f"  FAILED {f[0]} {f[1]} {f[2]} {f[3]} nx={f[4]} nz={f[5]}")
    if dupes:
        print(f"note: {len(dupes)} configuration(s) appeared more than once. A "
              f"re-measurement wins; otherwise the FASTEST of the duplicates wins "
              f"(deterministic, independent of argument order). "
              f"Distinct keys: {len(set(dupes))}")
    return list(rows.values())


# geo85's compute nodes report their own hostname; gpgpu reports
# "localhost.localdomain" unless BENCH_HOST is set. Both are CPU-scalability
# hosts for the purposes of picking a CPU baseline.
CPU_NODE_PREFIXES = ("geo85", "node")


def cpu_time_allowed(host: str) -> bool:
    """R7-1: `ktcloud` shares its host CPU with other tenants, so its CPU time
    goes in no report - not as a column, not as a baseline."""
    return canon_host(host) != "ktcloud"


def warn_empty(what: str, produced: list, source: list) -> None:
    """A filter that leaves nothing renders as an empty table, and an empty
    table reads as "there was no data" rather than "the filter is wrong".
    Three tables in this file were empty for months for exactly that reason
    (docs/90 N33), so silence is no longer allowed.
    """
    if source and not produced:
        print(f"WARNING: {what} produced 0 rows from {len(source)} input rows - "
              f"the filter is probably wrong, not the data.")


def case_level(case: str) -> tuple[str, str | None]:
    """(stem, physics level) for a case token that may carry a sweep TAG.

    `lock_exchange_v06.mxl` -> ("lock_exchange", "v06");  `lock_exchange.mxl`
    -> ("lock_exchange", "v05").  The `_v06r` variant of the mixing-length axis
    is a v0.6 run and pairs with the v0.5 run of the same stem.
    """
    stem = case.split(".", 1)[0]
    if stem.endswith("_v06r"):
        # v0.6 physics with the O(nz) mixing length: a DIFFERENT variant from
        # `_v06`, so it gets its own level and never overwrites it.
        return stem[: -len("_v06r")], "v06_recursive"
    if stem.endswith("_v06"):
        return stem[: -len("_v06")], "v06_integral"
    return stem, "v05"


def is_cpu_node(host: str) -> bool:
    h = str(host).split(".")[0]
    return any(h.startswith(p) for p in CPU_NODE_PREFIXES)


def fmt(x: float, unit: str = "") -> str:
    if x != x:
        return "—"
    if unit == "s":
        return f"{x:.3g} s"
    if unit == "e":
        return f"{x:.2e}"
    return f"{x:.3g}"


def pareto_plane(rows: list[dict], tol: float) -> dict:
    """Piece S: horizon runs with a finite error (diverged rows are excluded)."""
    s_rows = [r for r in rows if r["l2_max"] == r["l2_max"] and r["l2_max"] != float("inf")]
    out = {"tables": {}, "front": {}, "fastest_at_tol": {}}
    by_case = defaultdict(list)
    for r in s_rows:
        by_case[(r["case"], r["host"], r["backend"])].append(r)
    for (case, host, backend), rs in sorted(by_case.items()):
        rs.sort(key=lambda r: (r["scheme"], r["solver"], r["cfl"]))
        lines = [f"### {case} — {DEVICE.get(host, host)} · {backend} (nx={int(rs[0]['nx'])}, nz={int(rs[0]['nz'])})",
                 "", "| scheme | solver | CFL | steps | solver iters/step | wall (median±MAD) | L2 error | sim s / wall s |",
                 "|---|---|---:|---:|---:|---|---:|---:|"]
        for r in rs:
            ips = r["solver_iters"] / r["n_steps"] if r["n_steps"] else float("nan")
            lines.append(f"| {r['scheme']} | {r['solver']} | {r['cfl']:g} | {int(r['n_steps'])} | "
                         f"{fmt(ips)} | {fmt(r['wall_s'], 's')} ± {fmt(r['wall_mad_s'], 's')} | "
                         f"{fmt(r['l2_max'], 'e')} | {fmt(r['sim_s_per_wall_s'])} |")
        out["tables"][f"{case}|{host}|{backend}"] = "\n".join(lines)
        # Pareto front: no other point is both more accurate and faster.
        pts = [(r["l2_max"], r["wall_s"], r) for r in rs]
        front = [r for e, w, r in pts if not any((e2 <= e and w2 < w) or (e2 < e and w2 <= w)
                                                  for e2, w2, _ in pts)]
        out["front"][f"{case}|{host}|{backend}"] = [
            {"scheme": r["scheme"], "solver": r["solver"], "cfl": r["cfl"], "l2": r["l2_max"],
             "wall_s": r["wall_s"]} for r in sorted(front, key=lambda r: r["wall_s"])]
        # The winner at each tolerance - which is the paper's point: the ranking
        # changes with the error you are willing to accept (R8).
        winners = []
        for t in (1e-1, 1e-2, tol, 1e-4):
            ok = [r for r in rs if r["l2_max"] <= t]
            if ok:
                best = min(ok, key=lambda r: r["wall_s"])
                winners.append({"tol": t, "scheme": best["scheme"], "solver": best["solver"],
                                "cfl": best["cfl"], "wall_s": best["wall_s"], "l2": best["l2_max"]})
        if winners:
            out["fastest_at_tol"][f"{case}|{host}|{backend}"] = winners
    return out


def hardware_ladder(rows: list[dict]) -> dict:
    """Piece H: fixed-step runs (l2 nan). Cost per step per (nx, scheme, solver)."""
    h_rows = [r for r in rows if r["l2_max"] != r["l2_max"]]
    out = {"tables": {}, "speedups": []}
    diverged = [r for r in rows if r["l2_max"] == float("inf")]
    if diverged:
        out["diverged"] = [f"{r['host']}/{r['backend']} {r['case']} {r['scheme']}/{r['solver']} CFL {r['cfl']:g} nx={int(r['nx'])}" for r in diverged]
    by_prob = defaultdict(list)
    for r in h_rows:
        by_prob[(r["case"], r["scheme"], r["solver"], r["cfl"], int(r["nx"]),
                 int(r["nz"]), int(r["n_steps"]), precision_of(r["backend"]))].append(r)
    for (case, scheme, solver, cfl, nx, nz, n_steps, prec), rs in sorted(
            by_prob.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][4], kv[0][5], kv[0][7])):
        cells = nx * nx * nz
        # iteration counts must agree for a cross-node comparison (docs/04 S5)
        iters = {int(r["solver_iters"]) for r in rs}
        rs.sort(key=lambda r: r["wall_s"])
        lines = [f"### {case} · {scheme}/{solver} · CFL {cfl:g} · {nx}²×{nz} · **{prec}** ({n_steps} steps"
                 + (f", solver iters {iters.pop()}" if len(iters) == 1 else f", ITERS DIFFER {sorted(iters)}") + ")",
                 "", "| device | backend | threads | ms / step | Mcell·step / s | median ± MAD |",
                 "|---|---|---:|---:|---:|---|"]
        for r in rs:
            ms = 1e3 * r["wall_s"] / n_steps
            dev = (CPU_DEVICE if r["backend"] in CPU_BACKENDS else DEVICE).get(r["host"], r["host"])
            lines.append(f"| {dev} | {r['backend']} | "
                         f"{int(r['threads']) if r['backend'] in CPU_BACKENDS else '—'} | {ms:.3g} | "
                         f"{cells * n_steps / r['wall_s'] / 1e6:.4g} | {fmt(r['wall_s'], 's')} ± {fmt(r['wall_mad_s'], 's')} |")
        out["tables"][f"{case}|{scheme}|{solver}|{cfl}|{nx}|{nz}|{prec}"] = "\n".join(lines)
        # The CSV records `hostname`, which on the PBS cluster is the COMPUTE
        # node (node02, ...), never "geo85". Matching the literal string made
        # every speed-up row vanish without an error (docs/90 N33).
        cpu = [r for r in rs if r["backend"] in CPU_BACKENDS and cpu_time_allowed(r["host"])]
        gpu = [r for r in rs if r["backend"] not in CPU_BACKENDS]
        if cpu and gpu:
            for g in gpu:
                # R7: a CPU<->GPU ratio belongs to ONE node. The baseline is the
                # best CPU configuration OF THE GPU'S OWN NODE whenever that node
                # measured one; only when it did not does the ratio cross nodes,
                # and then it says so and names the CPU (R8-1). Hard-coding the
                # EPYC label gave every RTX 5090 row a baseline from a machine it
                # never ran beside, while its own 16-core CPU sat in the same
                # table (docs/90 N39).
                same = [r for r in cpu if r["host"] == g["host"]]
                if same:
                    pool, where = same, "same-node"
                else:
                    # Both baselines of one row must come from ONE machine. Taking
                    # "best CPU" from the fastest node and "serial" from whichever
                    # node happened to have a serial row put an EPYC and a
                    # Threadripper in the same sentence.
                    host = min(cpu, key=lambda r: r["wall_s"])["host"]
                    pool, where = [r for r in cpu if r["host"] == host], "cross-node"
                best_cpu = min(pool, key=lambda r: r["wall_s"])
                serial = [r for r in pool if r["backend"] == "fortran_serial"]
                out["speedups"].append({
                    # Every axis of the problem travels with the ratio. Without
                    # them a speed-up row cannot say WHICH problem it belongs to,
                    # and a table built from them can silently mix two (docs/90 N39).
                    "case": case, "cfl": cfl, "nz": nz, "n_steps": n_steps,
                    "precision": prec,
                    "nx": nx, "scheme": scheme, "solver": solver, "gpu_host": g["host"],
                    "gpu_device": DEVICE.get(g["host"], g["host"]),
                    "gpu_backend": g["backend"], "gpu_ms_per_step": 1e3 * g["wall_s"] / n_steps,
                    "baseline": where,
                    "best_cpu": f"{best_cpu['backend']} x{int(best_cpu['threads'])} on "
                                f"{CPU_DEVICE.get(best_cpu['host'], best_cpu['host'])}",
                    "speedup_vs_best_cpu": best_cpu["wall_s"] / g["wall_s"],
                    "serial_cpu": (f"fortran_serial on {CPU_DEVICE.get(serial[0]['host'], serial[0]['host'])}"
                                   if serial else None),
                    "speedup_vs_serial": (serial[0]["wall_s"] / g["wall_s"]) if serial else None})
    return out


def physics_cost(rows: list[dict]) -> list[dict]:
    """E14: cost of the representative physics - v0.6 (TKE closure + TVD UP3)
    over v0.5 for the same case, grid, scheme, solver, backend and device,
    from the fixed-step ladder (piece H)."""
    h_rows = [r for r in rows if r["l2_max"] != r["l2_max"]]     # inf (diverged) excluded
    pair = defaultdict(dict)
    for r in h_rows:
        # The sweep appends a TAG to the case token (`lock_exchange_v06.mxl`),
        # so `endswith("_v06")` was never true and every v0.6 run fell into the
        # v0.5 slot - the table silently disappeared (docs/90 N33).
        stem, level = case_level(r["case"])
        if level is None:
            continue
        key = (r["host"], r["backend"], int(r["threads"]), r["scheme"], r["solver"],
               r["cfl"], int(r["nx"]), int(r["nz"]), int(r["n_steps"]), stem)
        pair[key][level] = r
    out = []
    for key, v in sorted(pair.items()):
        if "v05" not in v:
            continue
        a = v["v05"]
        for level in ("v06_integral", "v06_recursive"):
            if level not in v:
                continue
            b = v[level]
            ms_a = 1e3 * a["wall_s"] / a["n_steps"]
            ms_b = 1e3 * b["wall_s"] / b["n_steps"]
            out.append({"host": key[0], "backend": key[1], "threads": key[2], "scheme": key[3],
                        "solver": key[4], "nx": key[6], "nz": key[7], "case": key[9],
                        "mxl": "integral" if level.endswith("integral") else "recursive",
                        "ms_v05": ms_a, "ms_v06": ms_b,
                        # per-STEP ratio: the table prints it next to two ms/step
                        # numbers, so it must be their quotient
                        "ratio": ms_b / ms_a,
                        "iters_v05": int(a["solver_iters"]), "iters_v06": int(b["solver_iters"])})
    return out

def mxl_cost(rows: list[dict]) -> list[dict]:
    """The S11.2 mixing-length axis: the O(nz^2) potential-energy budget of
    Gaspar et al. (1990) against the two O(nz) sweeps of NEMO nn_mxl=2, for
    the same closure, case, grid, scheme, solver, backend and device. The
    profile of docs/35 makes this the arithmetic of the single largest kernel,
    so the ratio is expected to depend on the device's fp64 prescription."""
    h_rows = [r for r in rows if r["l2_max"] != r["l2_max"]]     # inf (diverged) excluded
    pair = defaultdict(dict)
    for r in h_rows:
        c = r["case"]
        if "_v06" not in c:
            continue
        form = "recursive" if "_v06r" in c else "integral"
        base = c.replace("_v06r", "_v06")
        key = (r["host"], r["backend"], int(r["threads"]), r["scheme"], r["solver"],
               r["cfl"], int(r["nx"]), int(r["nz"]), int(r["n_steps"]), base)
        pair[key][form] = r
    out = []
    for key, v in sorted(pair.items()):
        if "integral" in v and "recursive" in v:
            a, b = v["integral"], v["recursive"]
            ms_a = 1e3 * a["wall_s"] / a["n_steps"]
            ms_b = 1e3 * b["wall_s"] / b["n_steps"]
            out.append({"host": key[0], "backend": key[1], "threads": key[2], "scheme": key[3],
                        "solver": key[4], "nx": key[6], "nz": key[7], "case": key[9],
                        "ms_integral": ms_a, "ms_recursive": ms_b,
                        "ratio": ms_a / ms_b,
                        "iters_integral": int(a["solver_iters"]), "iters_recursive": int(b["solver_iters"])})
    return out


def derived_times(rows: list[dict]) -> list[dict]:
    """Time to solution of the backends that did not run piece S (docs/26).

    t_B(config) = t_cuda(config) x [cost per step of B / cost per step of CUDA]
    measured in piece H on the same device, grid, scheme and solver. The
    ratio is taken at piece H's CFL; for the theta family the solver
    iteration count differs with CFL, so the ratio is per-iteration-neutral
    only to the extent that B and CUDA scale alike - the 1.1 % validation of
    docs/26 covers exactly that assumption.
    """
    s_rows = [r for r in rows if r["l2_max"] == r["l2_max"] and r["backend"] == "cuda"]
    h_rows = [r for r in rows if r["l2_max"] != r["l2_max"]]
    ratio = {}
    for h in h_rows:
        key = (h["host"], int(h["nx"]), h["scheme"], h["solver"])
        ratio.setdefault(key, {})[(h["backend"], int(h["threads"]))] = h["wall_s"]
    out = []
    for sr in s_rows:
        key = (sr["host"], int(sr["nx"]), sr["scheme"], sr["solver"])
        table = ratio.get(key, {})
        base = table.get(("cuda", 0))
        if base is None:
            continue
        for (b, t), w in table.items():
            if b == "cuda":
                continue
            out.append({"case": sr["case"], "scheme": sr["scheme"], "solver": sr["solver"], "cfl": sr["cfl"],
                        "nx": int(sr["nx"]), "host": sr["host"], "backend": b, "threads": t,
                        "l2": sr["l2_max"], "wall_s_derived": sr["wall_s"] * w / base,
                        "from": f"cuda {sr['wall_s']:.4g} s x ({b} x{t} / cuda = {w / base:.3g} in piece H)"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="merge tier-2 CSVs into the Pareto plane and hardware ladder")
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output/tier2_summary"))
    ap.add_argument("--tol", type=float, default=1e-3, help="fixed-error threshold (relative L2)")
    args = ap.parse_args()
    rows = load(args.csv)
    s = pareto_plane(rows, args.tol)
    h = hardware_ladder(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    (args.out / "pareto.json").write_text(json.dumps({"front": s["front"], "fastest_at_tol": s["fastest_at_tol"]}, indent=1))
    (args.out / "ladder.json").write_text(json.dumps(h["speedups"], indent=1))
    d = derived_times(rows)
    (args.out / "derived.json").write_text(json.dumps(d, indent=1))
    warn_empty("physics_cost (v0.5 vs v0.6 pairing)", physics_cost(rows), rows)
    pc = physics_cost(rows)
    (args.out / "physics_cost.json").write_text(json.dumps(pc, indent=1))
    warn_empty("mxl_cost (integral vs recursive pairing)", mxl_cost(rows), rows)
    mx = mxl_cost(rows)
    (args.out / "mxl_cost.json").write_text(json.dumps(mx, indent=1))
    md = ["# Tier-2 summary", "", f"{len(rows)} rows from {len(args.csv)} files.", "",
          "## Piece S — Pareto plane (error vs time to solution)", ""]
    for k in sorted(s["tables"]):
        md += [s["tables"][k], ""]
        ws = s["fastest_at_tol"].get(k) or []
        if ws:
            md += ["Fastest configuration at each tolerance:", "",
                   "| tolerance (rel. L2) | winner | wall | its error |", "|---|---|---:|---:|"]
            for f in ws:
                md.append(f"| ≤ {f['tol']:g} | {SCHEME_LABEL.get(f['scheme'], f['scheme'])} · {f['solver']} · CFL {f['cfl']:g} | "
                          f"{f['wall_s']:.3g} s | {f['l2']:.2e} |")
            md.append("")
    md += ["## Piece H — hardware ladder (byte-identical 50-step problem)", ""]
    for k in h["tables"]:
        md += [h["tables"][k], ""]
    if h["speedups"]:
        md += ["### Speed-ups (R8-1: both baselines named; R7: same-node where the node has a CPU row)", "",
               "| nx | scheme/solver | GPU | baseline | vs best CPU config | vs serial |",
               "|---:|---|---|---|---:|---:|"]
        for sp in h["speedups"]:
            md.append(f"| {sp['nx']} | {sp['scheme']}/{sp['solver']} | {sp['gpu_device']} {sp['gpu_backend']} | "
                      f"{sp['baseline']} | {sp['speedup_vs_best_cpu']:.3g}× ({sp['best_cpu']}) | "
                      + (f"{sp['speedup_vs_serial']:.3g}× ({sp['serial_cpu']})" if sp["speedup_vs_serial"] else "—") + " |")
    if pc:
        md += ["", "## Physics cost — v0.6 (TKE closure + TVD UP3) over v0.5, same problem otherwise", "",
               "| device | backend | threads | scheme/solver | nx | ms/step v0.5 | ms/step v0.6 | ratio | solver iters v0.5 → v0.6 |",
               "|---|---|---:|---|---:|---:|---:|---:|---|"]
        for x in pc:
            md.append(f"| {DEVICE.get(x['host'], x['host'])} | {x['backend']} | {x['threads'] if x['backend'] in CPU_BACKENDS else '—'} | "
                      f"{x['scheme']}/{x['solver']} | {x['nx']} | {x['ms_v05']:.3g} | {x['ms_v06']:.3g} | **{x['ratio']:.2f}×** | {x['iters_v05']} → {x['iters_v06']} |")
    if mx:
        md += ["", "## Mixing-length axis — S11.2 `integral` (O(nz^2)) over `recursive` (O(nz))", "",
               "| device | backend | scheme/solver | nx | ms/step integral | ms/step recursive | integral / recursive | solver iters |",
               "|---|---|---|---:|---:|---:|---:|---|"]
        for x in mx:
            md.append(f"| {DEVICE.get(x['host'], x['host'])} | {x['backend']} | {x['scheme']}/{x['solver']} | {x['nx']} | "
                      f"{x['ms_integral']:.3g} | {x['ms_recursive']:.3g} | **{x['ratio']:.2f}×** | "
                      f"{x['iters_integral']} / {x['iters_recursive']} |")
    if d:
        md += ["", "## Derived time to solution (backends that did not run piece S)", "",
               "| case | scheme/solver | CFL | backend | L2 | derived wall | from |", "|---|---|---:|---|---:|---:|---|"]
        for x in d:
            md.append(f"| {x['case']} | {x['scheme']}/{x['solver']} | {x['cfl']:g} | {x['backend']}"
                      + (f" x{x['threads']}" if x['threads'] else "") + f" | {x['l2']:.2e} | {x['wall_s_derived']:.3g} s | {x['from']} |")
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows -> {args.out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
