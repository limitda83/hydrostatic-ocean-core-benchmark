#!/usr/bin/env python3
#########################################################################
#  Module: shared_node_min                                              #
#  Description: Resolves repeated independent sweeps of the SAME         #
#               configuration on a node shared with other tenants        #
#               (ktcloud, RULES.md R7-7). Contention can only make a    #
#               run slower, never faster, and it inflates the median      #
#               without inflating the MAD when the intruder runs through  #
#               all five repeats (docs/90 N26). So the least contaminated #
#               estimate is the MINIMUM of the per-sweep medians, and a   #
#               sweep whose own MAD is large is dropped outright.         #
#  Pipeline: tier2_sweep (x N) -> shared_node_min -> docs table           #
#########################################################################

from __future__ import annotations

import argparse
import csv
import sys
import math
from collections import defaultdict
from pathlib import Path


def to_float(x) -> float | None:
    """A finite float, or None for missing / empty / nan / inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# Every column that names a point of the experiment matrix. Collapsing any of
# them compares two different points and breaks one-factor-at-a-time (R5). The
# list is generous on purpose: a column absent from a CSV is simply skipped, so
# feeding this tool a different matrix (helm: topo, r_std, ...) still keys on it.
AXES = ("host", "backend", "threads", "scheme", "theta", "solver", "cfl",
        "nx", "nz", "n_steps", "topo", "r_std", "rx0", "h_ratio")


def axes_of(r: dict) -> tuple[str, ...]:
    """The axis columns this CSV actually has, in AXES order."""
    return tuple(a for a in AXES if a in r)


def canon(v: str) -> str:
    """Canonical text for an axis value. A numeric axis rendered differently by
    two writers ("0" vs "0.0" vs "0.00", "1e3" vs "1000") is the SAME point of
    the matrix; keying on the raw text would put them in different buckets and
    the pair would silently never form. Non-numeric values pass through."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "" if v is None else str(v)
    return f"{int(f)}" if f == int(f) else repr(f)


def key_of(r: dict, axes: tuple[str, ...]) -> tuple:
    # A matrix without a `case` column (the Helmholtz kernel matrix) has no
    # two forms to pair; it is still resolved to a minimum-of-medians per
    # configuration, which is what a shared node needs. Every axis absent from
    # THIS row contributes "" so that files with different column sets still
    # key consistently instead of raising or collapsing.
    return tuple(canon(r.get(a, "")) for a in axes) + (r.get("case", ""),)


def main() -> int:
    ap = argparse.ArgumentParser(description="minimum-of-medians over repeated sweeps")
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--mad-max", type=float, default=0.02,
                    help="drop a sweep whose MAD/median exceeds this (default 2 %%)")
    ap.add_argument("--allow-mixed-axes", action="store_true",
                    help="permit inputs whose column sets differ; keys then use "
                         "only the columns every input has (say so in the report)")
    ap.add_argument("--show-single", action="store_true",
                    help="also print configurations that have only one form "
                         "(the whole table, for a matrix with no `case` column)")
    args = ap.parse_args()
    # The axis set is the UNION over every file, not the first file's. Two CSVs
    # with different column sets would otherwise either raise (a column the
    # first file lacked) or collapse a point (a column a later file adds).
    per_file: dict[str, tuple[str, ...]] = {}
    for path in args.csv:
        with path.open() as fh:
            row = next(iter(csv.DictReader(fh)), None)
            if row is not None:
                per_file[path.name] = axes_of(row)
    sets = set(per_file.values())
    if len(sets) > 1 and not args.allow_mixed_axes:
        # Filling a missing axis with a placeholder would put the two files in
        # different buckets and every pair would silently fail to form; treating
        # it as a wildcard could pair two DIFFERENT points. Neither is safe to
        # do quietly, so refuse and name the disagreement.
        print("REFUSED: the inputs do not describe the same experiment matrix.",
              file=sys.stderr)
        for name, ax in sorted(per_file.items()):
            print(f"  {name}: {', '.join(ax)}", file=sys.stderr)
        common = set.intersection(*(set(a) for a in sets))
        for name, ax in sorted(per_file.items()):
            extra = [a for a in ax if a not in common]
            if extra:
                print(f"  {name} has extra axis column(s): {', '.join(extra)}",
                      file=sys.stderr)
        print("  Pass --allow-mixed-axes to key on the common columns only, "
              "and say so in the report.", file=sys.stderr)
        return 2
    axes = tuple(a for a in AXES if a in set.intersection(*(set(a) for a in sets))) \
        if sets else ()

    best: dict[tuple, dict] = {}
    dropped, unusable = [], []
    for path in args.csv:
        for r in csv.DictReader(path.open()):
            w = to_float(r.get("wall_s"))
            if w is None or w <= 0.0:                   # nan / inf / missing
                continue
            mad = to_float(r.get("wall_mad_s"))
            if mad is None or mad < 0.0:
                # A row without a usable dispersion cannot be judged against the
                # MAD gate. Reporting it as if it passed would reintroduce the
                # very bias R7-2 exists to remove, so refuse it and say so.
                unusable.append((key_of(r, axes), w, r.get("wall_mad_s")))
                continue
            if mad / w > args.mad_max:
                dropped.append((key_of(r, axes), w, mad / w))
                continue
            k = key_of(r, axes)
            if k not in best or w < best[k]["w"]:
                best[k] = {"w": w, "mad": mad, "iters": r.get("solver_iters", ""),
                           "src": path.name}
    # The pair key is EVERY axis plus the case with the form stripped. Dropping
    # even one axis silently collapses distinct points onto one row and reports
    # whichever survived (verified: an nz sweep 15/30/60 collapsed to one row,
    # hiding two of the three ratios).
    pair = defaultdict(dict)
    for k, v in best.items():
        case = k[-1]
        form = "recursive" if "_v06r" in case else "integral"
        pair[k[:-1] + (case.replace("_v06r", "_v06"),)][form] = v
    head = "  ".join(f"{a:>8s}" for a in axes)
    print(f"{head}  {'integral':>9s} {'recursive':>9s}   ratio   iters")
    unpaired = 0
    for k in sorted(pair, key=lambda t: [str(x) for x in t]):
        v = pair[k]
        if len(v) < 2:
            unpaired += 1
            if args.show_single:
                only = next(iter(v))
                row = "  ".join(f"{x:>8s}" for x in k[:-1])
                print(f"{row}  {v[only]['w']:9.3f} {'-':>9s}  {only:>7s}  {v[only]['iters']}")
            continue
        a, b = v["integral"], v["recursive"]
        row = "  ".join(f"{x:>8s}" for x in k[:-1])
        print(f"{row}  {a['w']:9.3f} {b['w']:9.3f}  {a['w'] / b['w']:6.3f}x  "
              f"{a['iters']}/{b['iters']}")
    if unpaired and not args.show_single:
        print(f"\n{unpaired} configuration(s) had only one of the two forms - "
              f"not paired (--show-single lists them).")
    if dropped:
        print(f"\ndropped {len(dropped)} sweep row(s) on MAD > {args.mad_max:.0%}:")
        for k, w, f in dropped:
            print(f"  {' '.join(str(x) for x in k)}  wall {w:.3f}  MAD {f:.1%}")
    if unusable:
        print(f"\nREFUSED {len(unusable)} row(s) with no usable MAD "
              f"(R7-2: a timing without its dispersion is not a measurement):")
        for k, w, raw in unusable:
            print(f"  {' '.join(str(x) for x in k)}  wall {w:.3f}  wall_mad_s={raw!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
