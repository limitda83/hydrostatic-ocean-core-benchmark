#!/usr/bin/env python3
#########################################################################
#  Module: audit_data                                                   #
#  Description: Validates every preserved measurement CSV under expr/    #
#               before it is used in the manuscript. Checks the things   #
#               the recording bugs of docs/90 N27 actually produced: a   #
#               field that swallowed a whole benchmark line, a truncated #
#               exponent, a shifted row, a missing dispersion, a repeat  #
#               count that violates R7-2, and a timing that is not a     #
#               finite positive number. A file that fails here must be   #
#               re-measured, not repaired.                              #
#  Pipeline: expr/E##/data/*.csv -> audit_data -> re-run list            #
#########################################################################

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

TIME_COLS = ("wall_s", "wall_mad_s", "wall_min_s")
INT_COLS = ("iterations", "solver_iters", "n_repeat", "threads", "nx", "nz", "n_steps")
# R7-2 sets a FLOOR of five repeats, not a fixed value: a contended shared node
# needs more before the median settles (docs/90 N28 raised the Helmholtz matrix
# to nine). So the rule is ">= 5", and a smaller count is the violation.
MIN_REPEATS = 5
# Fortran writes 1.0D-04; Python float() does not accept the D exponent, so the
# text is normalised before it is parsed. A regex that accepts what the parser
# then rejects would fail a perfectly good measurement.
NUMERIC = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?$")


def finite(x: str) -> bool:
    try:
        v = float(str(x).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def num(x: str) -> float:
    return float(str(x).replace("D", "E").replace("d", "e"))


def audit(path: Path, args_mad: float = 0.5) -> tuple[list[str], int, list[int]]:
    bad: list[str] = []
    absent = 0
    absent_rows: list[int] = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return ["empty file"], 0, []
        ncol = len(header)
        idx = {name: i for i, name in enumerate(header)}
        for n, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != ncol:
                bad.append(f"line {n}: {len(row)} fields, header has {ncol} (shifted row?)")
                continue
            # A run that did not produce a number is a RESULT when the whole
            # row says so (OOM at 2000^2 on a 32 GB card, or a diverged config
            # - docs/90 N24, N18): every timing column is nan together and the
            # collector already renders it as OOM/diverged. A row where only
            # SOME timing columns are nan is corruption.
            present = [c for c in TIME_COLS if c in idx]
            blanks = [c for c in present
                      if row[idx[c]].strip() in ("", "nan", "NA")]
            all_absent = bool(blanks) and len(blanks) == len(present)
            if all_absent:
                # An absent measurement is a RESULT only if it is visible as one.
                # Excusing it silently would let a corrupt row that happens to be
                # all-nan through, so it is counted, listed, and its metadata is
                # still validated below.
                absent += 1
                absent_rows.append(n)
            for col in TIME_COLS:
                if col not in idx or all_absent:
                    continue
                v = row[idx[col]].strip()
                if v in ("", "nan", "NA"):
                    bad.append(f"line {n}: {col} is {v!r} while other timing "
                               f"columns are numbers (partial row = corruption)")
                elif not NUMERIC.match(v):
                    # this is what a sed that failed to match used to write
                    bad.append(f"line {n}: {col} is not a number: {v[:60]!r}")
                elif not finite(v):
                    bad.append(f"line {n}: {col} is non-finite: {v!r}")
                elif col in ("wall_s", "wall_min_s") and num(v) <= 0.0:
                    bad.append(f"line {n}: {col} <= 0: {v!r}")
                elif col == "wall_mad_s" and num(v) < 0.0:
                    bad.append(f"line {n}: wall_mad_s < 0: {v!r}")
            for col in INT_COLS:
                if col in idx:
                    v = row[idx[col]].strip()
                    if v and not re.match(r"^-?\d+$", v):
                        bad.append(f"line {n}: {col} is not an integer: {v[:40]!r}")
            if "n_repeat" in idx:
                v = row[idx["n_repeat"]].strip()
                if v and re.match(r"^\d+$", v) and int(v) < MIN_REPEATS:
                    bad.append(f"line {n}: n_repeat={v} violates R7-2 (floor is {MIN_REPEATS})")
            # a MAD larger than the median is a contended or broken measurement
            if "wall_s" in idx and "wall_mad_s" in idx and not all_absent:
                w, m = row[idx["wall_s"]].strip(), row[idx["wall_mad_s"]].strip()
                if finite(w) and finite(m) and num(w) > 0 and num(m) / num(w) > args_mad:
                    bad.append(f"line {n}: MAD/median = {num(m)/num(w):.1%} "
                               f"exceeds {args_mad:.0%} - re-measure or disclose "
                               f"(contention R7-7, or the huge-page bimodality of N29)")
    return bad, absent, absent_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="validate preserved measurement CSVs")
    ap.add_argument("paths", nargs="*", type=Path, default=[Path("expr")])
    ap.add_argument("--max-report", type=int, default=6)
    ap.add_argument("--mad-max", type=float, default=0.5,
                    help="flag a row whose MAD/median exceeds this (default 50 %%; "
                         "pass 0.02 to apply the R7-7 selection gate)")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files.extend(sorted(p.rglob("*.csv")) if p.is_dir() else [p])
    # A file listed in data/SUPERSEDED is kept for comparison only and is not
    # a manuscript source; auditing it would report failures nobody will fix.
    superseded: set[str] = set()
    for marker in Path("expr").rglob("SUPERSEDED"):
        for line in marker.read_text().splitlines():
            name = line.split("#")[0].strip()
            if name:
                superseded.add(str(marker.parent / name))
    skipped = [f for f in files if str(f) in superseded]
    files = [f for f in files if str(f) not in superseded]
    n_bad = 0
    for f in files:
        issues, absent, arows = audit(f, args.mad_max)
        rows = sum(1 for _ in f.open()) - 1
        note = (f", {absent} absent (OOM/diverged) at line(s) "
                f"{','.join(map(str, arows[:8]))}{'...' if len(arows) > 8 else ''}"
                if absent else "")
        if issues:
            n_bad += 1
            print(f"FAIL {f}  ({rows} rows{note}, {len(issues)} issue(s))")
            for line in issues[: args.max_report]:
                print(f"       {line}")
            if len(issues) > args.max_report:
                print(f"       ... {len(issues) - args.max_report} more")
        else:
            print(f"ok   {f}  ({rows} rows{note})")
    for f in skipped:
        print(f"skip {f}  (SUPERSEDED - kept for comparison, not a manuscript source)")
    print(f"\n{len(files)} file(s), {n_bad} failing"
          + (f", {len(skipped)} superseded" if skipped else ""))
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
