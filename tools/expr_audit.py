#!/usr/bin/env python3
#########################################################################
#  Module: expr_audit                                                   #
#  Description: Audits the experiment ledger of RULES.md R13. For every #
#               expr/E##_* folder it checks the required README fields,  #
#               whether a preserved copy of the data exists under        #
#               data/, and whether the report named by the README is     #
#               actually present in docs/. Writes the result back into   #
#               expr/README.md between two markers so the index cannot   #
#               silently drift away from the folders it describes.       #
#  Pipeline: expr/E##/ -> expr_audit -> expr/README.md provenance table  #
#########################################################################

from __future__ import annotations

import argparse
import csv
import fnmatch
import re
from pathlib import Path

FIELDS = ("질문/내용", "보고서", "노드", "상태", "재현 명령", "데이터", "코드 리비전")
# A finished experiment must either keep a copy of its data under data/, or say
# in one line where the data actually is. Silence is the failure mode that
# creates confusion later, not the absence of a copy.
EXEMPT = "보존 상태"
BEGIN = "<!-- BEGIN expr_audit -->"
END = "<!-- END expr_audit -->"


def field(text: str, name: str) -> str:
    """The one-line value of a **name:** field, or '' when absent."""
    m = re.search(rf"^\*\*{re.escape(name)}[^:]*:\*\*\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def audit_one(folder: Path, repo: Path) -> dict:
    readme = folder / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    missing = [f for f in FIELDS if f"**{f}" not in text]
    data_dir = folder / "data"
    files = sorted(p.name for p in data_dir.iterdir()) if data_dir.is_dir() else []
    reports = re.findall(r"docs/(\d\d_[\w]+\.md)", field(text, "보고서"))
    missing_reports = [r for r in reports if not (repo / "docs" / r).exists()]
    return {"name": folder.name, "status": field(text, "상태"), "nodes": field(text, "노드"),
            "claims": claims(text, folder) if readme.exists() else [],
            "reports": reports, "missing_reports": missing_reports,
            "n_data": len(files), "data": files, "missing_fields": missing,
            "kept": field(text, EXEMPT), "has_readme": readme.exists()}


def expand(token: str) -> list[str]:
    """A README filename template as a list of globs: `<...>` and `{a,b}` gone."""
    # `<stamp>` and a `{a..b}` RANGE both stand for "anything"; only a
    # `{a,b}` alternation enumerates. A token that starts with a separator
    # (`_c_metrics.json`) is a suffix the README wrote for brevity, not a
    # whole filename, so it is anchored with a leading wildcard.
    t = re.sub(r"<[^>]*>", "*", token)
    t = re.sub(r"\{[^{}]*\.\.[^{}]*\}", "*", t)
    if t[:1] in "_-.":
        t = "*" + t
    outs = [t]
    while any("{" in o for o in outs):
        nxt = []
        for o in outs:
            m = re.search(r"\{([^{}]*)\}", o)
            if not m:
                nxt.append(o); continue
            for alt in m.group(1).split(","):
                nxt.append(o[:m.start()] + alt.strip() + o[m.end():])
        outs = nxt
    return outs


def claims(text: str, folder: Path) -> list[str]:
    """Check the README's claims against what the folder actually holds.

    R13 asks for a record, but a record that drifts from its data is worse than
    none: it is a wrong answer delivered with confidence. So every filename the
    README names must exist, every count it states must match, and every node it
    names must appear in the data (and no node it does not name may appear).
    """
    out: list[str] = []
    data = folder / "data"
    have = {q.name for q in data.iterdir()} if data.is_dir() else set()
    line = field(text, "데이터")
    # Filenames the README names. A token carrying a template marker
    # (`<stamp>`, `*`, or a `{a,b}` alternation) describes a FAMILY of files,
    # so it is expanded to globs instead of being looked up verbatim. A token
    # containing a "/" is a path outside data/ and is checked as such.
    tokens = {t.strip() for t in re.findall(r"`([^`]+)`", line)}
    tokens |= {m.group(1) for m in
               re.finditer(r"(?:^|[\s(,])([A-Za-z][A-Za-z0-9_.\-]*\.(?:csv|json|txt|md))", line)}
    for tok in sorted(tokens):
        if not re.search(r"\.(csv|json|txt|md)$", tok):
            continue
        if "/" in tok:                      # a path outside data/: check it exists
            hits = list(Path().glob(tok)) if ("*" in tok or "?" in tok) else (
                [Path(tok)] if Path(tok).exists() else [])
            if not hits:
                out.append(f"README 가 가리키는 경로 `{tok}` 에 파일이 없음 "
                           f"(git 에 없는 곳이면 data/ 로 보존할 것)")
            continue
        for g in expand(tok):
            if not fnmatch.filter(have, g):
                out.append(f"README 의 `{tok}` 에 맞는 파일이 data/ 에 없음")
                break
    # "N행" / "N개" claims attached to a filename
    for name, n in re.findall(r"`([A-Za-z0-9_.\-]+\.csv)`\s*\((\d+)\s*행", line):
        q = data / name
        if q.is_file():
            rows = sum(1 for _ in q.open()) - 1
            if abs(rows - int(n)) > 0:
                out.append(f"`{name}`: README 는 {n}행, 실제 {rows}행")
    for n in re.findall(r"(\d+)\s*개", line):
        # a bare "N개" refers to the whole folder
        if have and abs(len(have) - int(n)) > 0 and not tokens:
            out.append(f"README 는 data/ 에 {n}개, 실제 {len(have)}개")
        break
    # A preserved verification result must be NEWER than the numerical core it
    # verifies. V5-1 failed for three days because N15 changed the core and the
    # stored PASS predated it (docs/90 N35).
    core = [Path("libs/core/model3d_v05.py"), Path("libs/core/model3d.py"),
            Path("libs/core/cases3d_v05.py"), Path("libs/core/closure.py")]
    newest_core = max((q.stat().st_mtime for q in core if q.exists()), default=0.0)
    superseded: set[str] = set()
    marker = data / "SUPERSEDED"
    if marker.is_file():
        for ln in marker.read_text(encoding="utf-8").splitlines():
            name = ln.split("#")[0].strip()
            if name:
                superseded.add(name)
    for q in sorted(have):
        if "verif" not in q.lower() or q in superseded:
            continue
        st = (data / q).stat().st_mtime
        if st < newest_core:
            import datetime as _dt
            f = _dt.datetime.fromtimestamp(st).strftime("%Y-%m-%d %H:%M")
            c = _dt.datetime.fromtimestamp(newest_core).strftime("%Y-%m-%d %H:%M")
            out.append(f"`{q}` 는 {f} 산출인데 수치 코어가 {c} 에 바뀌었다 "
                       f"— 검증을 다시 돌려야 한다 (N35)")

    # nodes named in the README vs hosts present in the CSVs
    nodes = field(text, "노드")
    hosts: set[str] = set()
    for q in sorted(have):
        if not q.endswith(".csv"):
            continue
        try:
            with (data / q).open(newline="") as fh:
                rd = csv.DictReader(fh)
                for r in rd:
                    h = (r.get("host") or "").strip()
                    if h:
                        hosts.add(h.split(".")[0])
        except (OSError, csv.Error):
            continue
    alias = {"node01": "geo85", "node02": "geo85", "node03": "geo85",
             "node04": "geo85", "node05": "geo85", "main1": "ktcloud",
             "localhost": "gpgpu", "gpgpu-dev": "gpgpu"}
    for h in sorted(hosts):
        canon_h = alias.get(h, h)
        if canon_h not in nodes and h not in nodes:
            out.append(f"data/ 에 host={h} 이 있는데 `노드` 줄에 없음: {nodes[:60]!r}")
    return out


def verdict(a: dict) -> str:
    if not a["has_readme"]:
        return "README 없음"
    if a["missing_fields"]:
        return "필드 누락: " + ", ".join(a["missing_fields"])
    if a["missing_reports"]:
        return "보고서 없음: " + ", ".join(a["missing_reports"])
    if a["claims"]:
        return "**기록 불일치**: " + " · ".join(a["claims"][:2])
    if a["n_data"] == 0 and not a["kept"]:
        return "**출처 불명** (data/ 비어 있고 `보존 상태` 도 없음)"
    if a["n_data"] == 0:
        return "보존 상태 명시"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser(description="audit the R13 experiment ledger")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--write", action="store_true", help="update expr/README.md in place")
    args = ap.parse_args()
    expr = args.repo / "expr"
    rows = [audit_one(d, args.repo) for d in sorted(expr.glob("E*_*")) if d.is_dir()]
    lines = ["> 상태의 단일 출처는 각 `expr/E##/README.md` 의 `**상태:**` 필드다. 이 표는 그것을 읽어 쓴다.",
             "", "| 실험 | 상태 | 보고서 | `data/` | 감사 |", "|---|---|---|---:|---|"]
    for a in rows:
        rep = ", ".join(f"docs/{r}" for r in a["reports"]) or "—"
        kept = f"{a['n_data']} 파일" if a["n_data"] else (a["kept"][:44] or "—")
        lines.append(f"| `{a['name']}` | {a['status'][:52] or '—'} | {rep} | {kept} | {verdict(a)} |")
    table = "\n".join(lines)
    ok = ("OK", "보존 상태 명시")          # an explicit provenance line is acceptable
    bad = [a["name"] for a in rows if verdict(a) not in ok]
    if args.write:
        p = expr / "README.md"
        s = p.read_text(encoding="utf-8")
        block = (f"{BEGIN}\n\n## 대장 감사 (`python3 tools/expr_audit.py --write` 가 생성)\n\n"
                 f"{table}\n\n{END}")
        if BEGIN in s:
            s = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block, s, flags=re.S)
        else:
            s = s.rstrip() + "\n\n" + block + "\n"
        p.write_text(s, encoding="utf-8")
        print(f"expr/README.md updated ({len(rows)} experiments)")
    else:
        print(table)
    if bad:
        print("\n감사 실패: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
