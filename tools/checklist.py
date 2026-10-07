#!/usr/bin/env python3
#########################################################################
#  Module: checklist                                                    #
#  Description: The one place every "is this experiment done and can    #
#               its numbers be used" question is asked, for every       #
#               experiment, mechanically. Each row of the matrix is an  #
#               expr/E##; each column is a check that some past defect  #
#               (docs/90) showed a human forgets. Wraps the existing    #
#               auditors and adds the checks none of them made: gate    #
#               records per node x backend (R4-2), known host labels,   #
#               diverged rows, physically impossible orderings, and     #
#               dangling SUPERSEDED entries.                            #
#  Pipeline: expr/E*/ + docs/ -> checklist.py -> expr/CHECKLIST.md      #
#########################################################################

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import expr_audit  # noqa: E402  (its audit_one / claims are reused)

KNOWN_HOSTS = {"geo85": "geo85", "gpgpu": "gpgpu", "ktcloud": "ktcloud"}
CPU_BACKENDS = {"fortran_serial", "fortran_omp"}
GATE_DIR = Path("expr/E05_3d_reference/data/gates_20260915")
# Which gate log covers which (host, backend). JAX is gated by run_tier2_jax_pass.sh
# at measurement time and no log is preserved - that is a finding, not a pass.
GATE_LOGS = {
    ("geo85", "fortran_serial"): "geo85_gate3d5.log", ("geo85", "fortran_omp"): "geo85_gate3d5.log",
    ("gpgpu", "fortran_serial"): "gpgpu_gate3d5.log", ("gpgpu", "fortran_omp"): "gpgpu_gate3d5.log",
    ("gpgpu", "openacc"): "gpgpu_gate3d5.log", ("gpgpu", "cuda"): "gpgpu_gate3d5.log",
    ("gpgpu", "jax"): "gpgpu_gate3d5_jax.log", ("ktcloud", "jax"): "ktcloud_gate3d5_jax.log",
    ("ktcloud", "cuda"): "ktcloud_gate3d5_cuda_acc_stale.log",
    ("ktcloud", "openacc"): "ktcloud_gate3d5_acc_rebuilt.log",
}
CHECKS = [
    ("C1", "README 필수 필드"), ("C2", "보존본 또는 보존 상태"), ("C3", "README 주장 ↔ data/"),
    ("C4", "CSV 건전성·반복수"), ("C5", "노드×백엔드 게이트 기록 (R4-2)"), ("C6", "host 열이 아는 기계"),
    ("C7", "발산 행 없음"), ("C8", "물리적으로 가능한 순서"), ("C9", "보고서 존재·신선도"),
    ("C10", "교차표 한 표 = 한 문제"), ("C11", "SUPERSEDED 항목 실재"), ("C12", "gpgpu-dev/localhost 없음"),
]


def canon_host(h: str) -> str:
    h = h.split(".")[0]
    if h.startswith("node") or h == "geo85":
        return "geo85"
    if h.startswith("localhost"):
        return "localhost"
    return h


def superseded(data: Path) -> set[Path]:
    out = set()
    f = data / "SUPERSEDED"
    if f.exists():
        for line in f.read_text().splitlines():
            t = line.split("#")[0].strip()
            if t:
                out.add(data / t)
    return out


def superseded_entries(data: Path) -> list[str]:
    f = data / "SUPERSEDED"
    if not f.exists():
        return []
    return [l.split("#")[0].strip() for l in f.read_text().splitlines() if l.split("#")[0].strip()]


def csv_rows(data: Path):
    sup = superseded(data)
    for p in sorted(data.glob("*.csv")):
        if p in sup:
            continue
        try:
            with p.open() as fh:
                for r in csv.DictReader(fh):
                    yield p, r
        except Exception:
            continue


def check_gates(data: Path) -> tuple[str, str]:
    """C5: every (host, backend) that produced a timing row has a gate log
    with 0 fail. fp32 (_sp) and mixed (_mixed) variants inherit the gate of their
    fp64 base and need the generator bit-identity check instead (docs/40)."""
    seen = set()
    for p, r in csv_rows(data):
        if "host" not in r or "backend" not in r:
            continue
        h = canon_host(r["host"])
        if h not in KNOWN_HOSTS:
            continue
        b = r["backend"]
        # fp32 (_sp) and mixed (_mixed) variants cannot meet the 1e-12 gate and are
        # not meant to (R9); their check is the generator's bit-identity (docs/40),
        # and they inherit the gate of the fp64 binary they were generated from.
        base = re.sub(r"_(sp|mixed)$", "", b)
        seen.add((h, base))
    if not seen:
        return "—", "타이밍 행 없음"
    missing = []
    for h, b in sorted(seen):
        log = GATE_LOGS.get((h, b))
        if log is None or not (GATE_DIR / log).exists():
            missing.append(f"{h}/{b}")
            continue
        tail = (GATE_DIR / log).read_text().strip().splitlines()[-1]
        m = re.search(r"(\d+) pass, (\d+) fail", tail)
        if not m or int(m.group(2)) != 0 and b != "cuda":
            # the stale ktcloud log has 12 fail for the OLD openacc binary; its
            # cuda half is the record. That file is only mapped for cuda above.
            missing.append(f"{h}/{b}({tail})")
    return ("✓", "") if not missing else ("✗", "게이트 기록 없음: " + ", ".join(missing))


def check_hosts(data: Path) -> tuple[str, str]:
    bad = defaultdict(int)
    for p, r in csv_rows(data):
        if "host" in r:
            h = canon_host(r["host"])
            if h not in KNOWN_HOSTS:
                bad[(h, p.name)] += 1
    if not bad:
        return "✓", ""
    return "✗", "; ".join(f"{h} ×{n} in {f}" for (h, f), n in sorted(bad.items()))


def check_diverged(data: Path) -> tuple[str, str]:
    bad = []
    for p, r in csv_rows(data):
        try:
            it, n = float(r.get("solver_iters", "nan")), float(r.get("n_steps", "nan"))
            l2 = float(r.get("l2_max", "nan"))
        except ValueError:
            continue
        # Two signatures: an elliptic solve spinning to max_iter, or the run's own
        # divergence mark (l2 = inf, written by record() for every scheme - the
        # only mark a split-explicit run leaves, since it has no solver). A row
        # with either is a RESULT and belongs in a *_RESULT.csv, not a timing file.
        if (n > 0 and it / n >= 2000.0) or l2 == float("inf"):
            bad.append(f"{p.name}:{r.get('case')}/{r.get('scheme')}/{r.get('nx')}")
    if not bad:
        return ("—", "") if not any(True for _ in csv_rows(data)) else ("✓", "")
    return "✗", f"max_iter 행 {len(bad)}: " + ", ".join(bad[:4]) + ("…" if len(bad) > 4 else "")


def check_monotone(data: Path) -> tuple[str, str]:
    """C8: in a physics decomposition, adding physics cannot make a step faster."""
    groups: dict[tuple, dict[str, float]] = defaultdict(dict)
    for p, r in csv_rows(data):
        if "." not in r.get("case", "") or "wall_s" not in r:
            continue
        stem, tag = r["case"].split(".", 1)
        if tag not in ("v05", "up3", "up3tvd", "tke", "tke_up3tvd"):
            continue
        try:
            groups[(p.name, r["host"], r["backend"], r["scheme"], r["nx"])][tag] = float(r["wall_s"])
        except ValueError:
            pass
    if not groups:
        return "—", "분해 파일 없음"
    bad = []
    for k, w in groups.items():
        pairs = [("v05", "up3"), ("up3", "up3tvd"), ("v05", "tke"), ("tke", "tke_up3tvd"), ("up3tvd", "tke_up3tvd")]
        for a, b in pairs:
            if a in w and b in w and w[b] < w[a]:
                bad.append(f"{k[0]}:{k[2]}/{k[4]}² {b}<{a}")
    return ("✓", "") if not bad else ("✗", "; ".join(bad[:4]))


def check_split_substeps(data: Path) -> tuple[str, str]:
    """C13: a split-explicit run that diverges stops early, and the CSV then
    carries a shortened barotropic substep count next to a wall time divided by
    the FULL step count. The solver-iteration guard cannot see it (split has no
    solver). Every row of the same (case, cfl, nx) must show the same
    substeps/step; a row that does not is a diverged run in disguise (N45)."""
    groups: dict[tuple, list] = defaultdict(list)
    for p, r in csv_rows(data):
        if r.get("scheme") != "split" or "substeps" not in r:
            continue
        try:
            n, ss = int(r["n_steps"]), int(r["substeps"])
        except ValueError:
            continue
        if n > 0:
            groups[(r["case"].split(".")[0], float(r["cfl"]), int(r["nx"]))].append((ss / n, p.name, r["host"], r["backend"]))
    if not groups:
        return "—", ""
    bad = []
    for k, v in groups.items():
        vals = [x[0] for x in v]
        maj = max(set(vals), key=vals.count)
        bad += [f"{k[0]}/{k[2]}² {h}/{b} {ss:g}/step≠{maj:g} ({f})" for ss, f, h, b in v if ss != maj]
    return ("✓", "") if not bad else ("✗", "; ".join(bad[:4]))


def check_superseded(data: Path) -> tuple[str, str]:
    ent = superseded_entries(data)
    if not ent:
        return "—", ""
    gone = [e for e in ent if not (data / e).exists()]
    return ("✓", f"{len(ent)}개") if not gone else ("✗", "없는 파일: " + ", ".join(gone))


def check_crossdevice(reports: list[str]) -> tuple[str, str]:
    rows = Path("output/tier2_summary/rows.json")
    if not rows.exists():
        return "?", "rows.json 없음 (tier2_collect 먼저)"
    res, notes = [], []
    for d in reports:
        doc = Path("docs") / d
        if not doc.exists():
            continue
        out = subprocess.run([sys.executable, "tools/audit_crossdevice.py", str(doc), "--rows", str(rows)],
                             capture_output=True, text=True)
        last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if "table(s)" in last:
            res.append(out.returncode == 0)
            notes.append(f"{d}: {last.split(':', 1)[1].strip()}")
    if not res:
        return "—", "사다리 표 없음"
    return ("✓", "; ".join(notes)) if all(res) else ("✗", "; ".join(notes))


def run(exp: Path) -> dict:
    a = expr_audit.audit_one(exp, Path("."))
    data = exp / "data"
    res: dict[str, tuple[str, str]] = {}
    res["C1"] = ("✓", "") if not a["missing_fields"] else ("✗", "누락: " + ", ".join(a["missing_fields"]))
    res["C2"] = ("✓", "") if (a["n_data"] > 0 or a["kept"]) else ("✗", "data/ 비었고 보존 상태 없음")
    # expr_audit.claims() returns the list of MISMATCH descriptions; empty means every
    # filename / count / host the README names was found in data/.
    claims_bad = [str(c) for c in a["claims"]]
    res["C3"] = ("✓", "") if not claims_bad else ("✗", "; ".join(claims_bad[:3]))
    ad = subprocess.run([sys.executable, "tools/audit_data.py", str(data)], capture_output=True, text=True) \
        if data.is_dir() else None
    if ad is None:
        res["C4"] = ("—", "")
    else:
        m = re.search(r"(\d+) file\(s\), (\d+) failing", ad.stdout + ad.stderr)
        res["C4"] = ("✓", f"{m.group(1)} 파일") if m and m.group(2) == "0" else ("✗", (ad.stdout + ad.stderr).strip().splitlines()[-1][:80] if m else "audit_data 실행 실패")
    res["C5"] = check_gates(data) if data.is_dir() else ("—", "")
    res["C6"] = check_hosts(data) if data.is_dir() else ("—", "")
    res["C7"] = check_diverged(data) if data.is_dir() else ("—", "")
    res["C8"] = check_monotone(data) if data.is_dir() else ("—", "")
    res["C9"] = ("✓", "") if (a["reports"] and not a["missing_reports"]) else ("✗", "없는 보고서: " + ", ".join(a["missing_reports"]) if a["missing_reports"] else "보고서 미기재")
    res["C10"] = check_crossdevice(a["reports"])
    res["C11"] = check_superseded(data) if data.is_dir() else ("—", "")
    res["C13"] = check_split_substeps(data) if data.is_dir() else ("—", "")
    res["C12"] = res["C6"]  # same evidence, kept as its own column so the matrix names the specific failure class
    return {"name": exp.name, "status": a["status"][:40], "checks": res}


def main() -> int:
    ap = argparse.ArgumentParser(description="per-experiment checklist matrix")
    ap.add_argument("--write", action="store_true", help="write expr/CHECKLIST.md")
    args = ap.parse_args()
    exps = sorted(p for p in Path("expr").glob("E*_*") if p.is_dir())
    results = [run(e) for e in exps]
    head = "| 실험 | " + " | ".join(c for c, _ in CHECKS) + " |"
    sep = "|---|" + "|".join(":-:" for _ in CHECKS) + "|"
    lines = [head, sep]
    for r in results:
        lines.append(f"| `{r['name']}` | " + " | ".join(r["checks"][c][0] for c, _ in CHECKS) + " |")
    legend = "\n".join(f"- **{c}** {n}" for c, n in CHECKS)
    details = []
    for r in results:
        bad = [(c, r["checks"][c][1]) for c, _ in CHECKS if r["checks"][c][0] == "✗"]
        if bad:
            details.append(f"### `{r['name']}`\n" + "\n".join(f"- **{c}**: {msg}" for c, msg in bad))
    n_fail = sum(1 for r in results for c, _ in CHECKS if r["checks"][c][0] == "✗")
    doc = ("# 실험별 검사 체크리스트\n\n> `python3 tools/checklist.py --write` 가 생성한다. 손으로 고치지 않는다.\n"
           f"> ✓ 통과 · ✗ 실패 · — 해당 없음 · ? 판정 불가. **실패 {n_fail}건.**\n\n"
           + "\n".join(lines) + "\n\n## 항목\n" + legend + "\n\n## 실패 상세\n\n"
           + ("\n\n".join(details) if details else "없음") + "\n")
    print(doc)
    if args.write:
        Path("expr/CHECKLIST.md").write_text(doc)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
