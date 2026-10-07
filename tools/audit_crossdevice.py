#!/usr/bin/env python3
#########################################################################
#  Module: audit_crossdevice                                            #
#  Description: Mechanically checks that every cell of a cross-device    #
#               table in docs/ is (a) traceable to a preserved          #
#               measurement row, (b) drawn from the SAME problem as     #
#               every other cell of its table - same case, scheme,      #
#               solver, CFL, step count - and (c) not taken from a run   #
#               the pipeline flagged as diverged or from `ktcloud` CPU   #
#               time (R7-1). A table that mixes two CFLs or two solvers  #
#               across its columns looks like a device comparison and    #
#               is not one (docs/90 N39).                                #
#  Pipeline: tier2_collect -> rows.json -> audit_crossdevice -> docs/32  #
#########################################################################

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# The device column of a ladder table names hardware, not a host; this maps it
# back to the host label the CSVs carry.
HOST_OF = {"H100": "ktcloud", "RTX 5090": "gpgpu", "EPYC 9655": "geo85"}
# Column order of the ladder tables in docs/32.
COLUMNS = ["cuda", "openacc", "jax", "fortran_omp", "fortran_serial"]
CPU_BACKENDS = ("fortran_serial", "fortran_omp")

# Emphasis can wrap the whole cell or just the number (`**6.38** x16 (2.2 %)`),
# so the markers are stripped before matching rather than spelled out here.
CELL = re.compile(
    r"^(?P<ms>[0-9]+(?:\.[0-9]+)?)"                  # ms / step
    r"(?:\s*×(?P<threads>[0-9]+))?"                  # optional thread count
    r"\s*\((?P<mad>[0-9]+(?:\.[0-9]+)?)\s*%\)$")


def unbold(text: str) -> str:
    return text.replace("*", "").strip()


def device_of(text: str) -> str | None:
    for name, host in HOST_OF.items():
        if text.startswith(name):
            return host
    return None


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    for r in rows:
        r["ms_per_step"] = (1e3 * r["wall_s"] / r["n_steps"]) if r["n_steps"] else float("nan")
        r["mad_pct"] = 100.0 * r["wall_mad_s"] / r["wall_s"] if r["wall_s"] else float("nan")
        r["diverged"] = r.get("l2_max") == float("inf") or r.get("diverged", False)
    return rows


def canon_case(case: str) -> str:
    """The mixing-length sweep tags its case `.mxl`; with the default (integral)
    form that is the SAME problem as the ladder's `_v06` row, measured by a
    second sweep. Only the `_v06r` stem is different physics and keeps its tag."""
    if case.endswith(".mxl") and case[:-4].endswith("_v06"):
        return case[:-4]
    return case


def provenance(r: dict) -> tuple:
    return (canon_case(r["case"]), r["scheme"], r["solver"], r["cfl"], int(r["n_steps"]), int(r["nz"]))


def prov_text(p: tuple) -> str:
    return f"{p[0]} · {p[1]}/{p[2]} · CFL {p[3]:g} · {p[4]} steps · nz={p[5]}"


def parse_tables(md: str) -> tuple[list[dict], list[str]]:
    """Every markdown table whose first column is a grid size and whose second
    column names a device. Returns one dict per table with its parsed cells."""
    tables, cur = [], None
    skipped: list[str] = []
    heading = ""
    for line in md.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
        if line.startswith("|"):
            parts = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(parts) < 3:
                continue
            if parts[0].startswith("격자") or set(parts[0]) <= set("-: "):
                if parts[0].startswith("격자"):
                    # Only a LADDER table is this tool's business: first column a
                    # grid, second the device, then one column per backend. Other
                    # tables (ratios, summaries) have their own shapes, and
                    # pretending to audit them produces confident nonsense - so
                    # they are skipped and counted, not failed.
                    if [c.lower() for c in parts[2:]] == [b.replace("_", " ") for b in
                                                          ("cuda", "openacc", "jax",
                                                           "fortran openmp", "fortran 직렬")]:
                        cur = {"heading": heading, "cells": []}
                        tables.append(cur)
                    else:
                        cur = None
                        skipped.append(heading)
                continue
            if cur is None:
                continue
            m = re.match(r"^([0-9]+)²$", parts[0])
            host = device_of(parts[1])
            if not m or host is None:
                continue
            nx = int(m.group(1))
            for backend, text in zip(COLUMNS, parts[2:]):
                if text in ("—", "-", ""):
                    continue
                cur["cells"].append({"nx": nx, "host": host, "backend": backend,
                                     "text": text, "bold": text.startswith("**")})
        else:
            cur = cur if line.strip().startswith("|") else cur
    return [t for t in tables if t["cells"]], skipped


def decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def candidates(rows: list[dict], cell: dict, rel: float) -> list[dict]:
    """Rows whose ms/step ROUNDS to the printed figure.

    Comparing against a fixed relative tolerance rejected correct cells: a value
    printed as `0.64` is any measurement in [0.635, 0.645), which is 0.8 % wide,
    and a 0.2 % window called the true source row a miss.
    """
    out = []
    m = CELL.match(unbold(cell["text"]))
    if not m:
        return out
    ms = float(m.group("ms"))
    nd = decimals(m.group("ms"))
    thr = m.group("threads")
    for r in rows:
        if r["host"] != cell["host"] or int(r["nx"]) != cell["nx"]:
            continue
        # fp32 is a separate axis (R9) and never shares a ladder column.
        if r["backend"] != cell["backend"]:
            continue
        if thr is not None and int(r["threads"]) != int(thr):
            continue
        if r["ms_per_step"] != r["ms_per_step"] or ms <= 0:
            continue
        if not (round(r["ms_per_step"], nd) == ms or abs(r["ms_per_step"] - ms) / ms <= rel):
            continue
        # Second key: the printed MAD percentage must round the same way too.
        mad_txt = m.group("mad")
        if mad_txt is not None and r["wall_s"]:
            if round(r["mad_pct"], decimals(mad_txt)) != float(mad_txt):
                continue
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="audit a cross-device table against the measured rows")
    ap.add_argument("doc", type=Path)
    ap.add_argument("--rows", type=Path, required=True, help="rows.json from tier2_collect")
    ap.add_argument("--rel", type=float, default=0.0, help="extra relative slack beyond the printed rounding window")
    args = ap.parse_args()

    rows = load_rows(args.rows)
    tables, skipped = parse_tables(args.doc.read_text())
    if not tables:
        if skipped:
            # The file has tables, just none of this tool's shape. That is a
            # legitimate "nothing to audit here", not a parser regression.
            print(f"{args.doc}: no ladder-shaped table ({len(skipped)} other table(s) skipped) - nothing to audit")
            return 0
        print(f"FAIL: no cross-device table parsed out of {args.doc}")
        return 1

    failures = 0
    for t in tables:
        print(f"\n== {t['heading']}  ({len(t['cells'])} cells)")
        unresolved, percell = [], {}
        for c in t["cells"]:
            cand = candidates(rows, c, args.rel)
            key = (c["nx"], c["host"], c["backend"])
            percell[key] = cand
            if not cand:
                unresolved.append(f"{c['nx']}² {c['host']}/{c['backend']} = {c['text']}")
        # C1: every printed number must exist in the preserved data.
        if unresolved:
            failures += len(unresolved)
            print(f"  FAIL C1 traceability: {len(unresolved)} cell(s) match no preserved row")
            for u in unresolved:
                print(f"    {u}")
        # C2: one provenance must explain the WHOLE table. A cell that matched
        # nothing has already failed C1; it must not be dropped from the
        # intersection, or the audit prints "ok C2" next to a C1 failure for
        # the same table (external review 2026-09-15). Treat it as the empty
        # set, which makes the intersection empty.
        sets = [set(provenance(r) for r in cand) for cand in percell.values()]
        common = set.intersection(*sets) if sets and all(sets) else set()
        # C2b: value-matching proves a common provenance EXISTS, not that it was
        # the source. A printed cell that rounds onto rows of two different
        # problems is AMBIGUOUS and is reported as such - the reader must be
        # told which cells the audit could not pin to a single problem.
        ambiguous = [(k, sorted(set(provenance(r) for r in cand)))
                     for k, cand in percell.items()
                     if cand and len(set(provenance(r) for r in cand)) > 1]
        if ambiguous:
            failures += len(ambiguous)
            print(f"  FAIL C2b ambiguity: {len(ambiguous)} cell(s) round onto more than one problem - "
                  "the audit cannot tell which one the table used")
            for (nx, host, backend), provs in ambiguous[:8]:
                print(f"    {nx}² {host}/{backend} <- " + " | ".join(prov_text(p) for p in provs))
        if not common:
            failures += 1
            print("  FAIL C2 coherence: no single problem explains every cell - "
                  "this table is not a device comparison.")
            byprov = defaultdict(list)
            for (nx, host, backend), cand in sorted(percell.items()):
                if not cand:
                    continue
                for p in sorted(set(provenance(r) for r in cand)):
                    byprov[p].append(f"{nx}² {host}/{backend}")
            for p, who in sorted(byprov.items(), key=lambda kv: -len(kv[1])):
                print(f"    {len(who):3d} cell(s) <- {prov_text(p)}")
                if len(who) <= 6:
                    print(f"         {', '.join(who)}")
        else:
            print(f"  ok   C2 coherence: {prov_text(sorted(common)[0])}"
                  + (f"  (+{len(common)-1} equally consistent)" if len(common) > 1 else ""))
        # C3/C4/C5 are judged on the rows that survive the common provenance.
        for (nx, host, backend), cand in sorted(percell.items()):
            keep = [r for r in cand if provenance(r) in common] or cand
            if not keep:
                continue
            r = keep[0]
            if r["diverged"]:
                failures += 1
                print(f"  FAIL C3 diverged: {nx}² {host}/{backend} is a run that hit max_iter "
                      f"({int(r['solver_iters'])} iters / {int(r['n_steps'])} steps)")
            if host == "ktcloud" and backend in CPU_BACKENDS:
                failures += 1
                print(f"  FAIL C4 R7-1: {nx}² quotes ktcloud CPU time")
            c = next(x for x in t["cells"] if (x["nx"], x["host"], x["backend"]) == (nx, host, backend))
            if (r["mad_pct"] > 2.0) != c["bold"]:
                failures += 1
                print(f"  FAIL C5 MAD marking: {nx}² {host}/{backend} MAD {r['mad_pct']:.1f} % "
                      f"but {'not ' if not c['bold'] else ''}marked unstable")
    print(f"\naudit_crossdevice: {len(tables)} table(s), {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
