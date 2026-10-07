#!/usr/bin/env python3
#########################################################################
#  Module: tier2_report                                                 #
#  Description: Renders the tier-2 summary (tools/tier2_collect.py     #
#               output) as one self-contained HTML page: the Pareto     #
#               plane of piece S (error vs simulated seconds per wall   #
#               second, per case), the hardware ladder of piece H       #
#               (ms per step vs cells, per backend and device), the     #
#               R8-1 speed-up table and the findings log. Charts are    #
#               inline SVG drawn here - no library, so the page holds   #
#               under the artifact CSP. Re-run after every collect.     #
#  Pipeline: tier2_collect -> tier2_report -> output/tier2_report.html  #
#########################################################################

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEVICE = {"gpgpu": "RTX 5090", "ktcloud": "H100", "geo85": "EPYC 9655"}
BACKEND_LABEL = {"cuda": "CUDA", "openacc": "OpenACC", "jax": "JAX", "fortran_omp": "Fortran OpenMP",
                 "fortran_serial": "Fortran serial"}
BACKEND_COLOR = {"cuda": "var(--c-cuda)", "openacc": "var(--c-acc)", "jax": "var(--c-jax)",
                 "fortran_omp": "var(--c-omp)", "fortran_serial": "var(--c-ser)"}
SCHEME_COLOR = {"fb": "var(--c-fb)", "theta0.5": "var(--c-t05)", "theta0.6": "var(--c-t06)",
                "theta1.0": "var(--c-t10)", "split": "var(--c-split)"}
SCHEME_LABEL = {"fb": "forward–backward", "theta0.5": "θ = 0.5", "theta0.6": "θ = 0.6",
                "theta1.0": "θ = 1.0 (implicit)", "split": "split-explicit"}
CASE_LABEL = {"baroclinic_igw": "① internal wave over a seamount",
              "basin_seiche": "② barotropic seiche in a closed basin",
              "lock_exchange": "③ lock exchange (T/S, TEOS-10, advection)"}
CPU = ("fortran_omp", "fortran_serial")


def esc(x) -> str:
    return html.escape(str(x))


def fmt(x: float, d: int = 3) -> str:
    return "—" if x != x else f"{x:.{d}g}"


# ------------------------------------------------------------------ charts
def log_ticks(lo: float, hi: float) -> list[float]:
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10.0 ** e for e in range(a, b + 1)]


def scatter_svg(points: list[dict], xlab: str, ylab: str, width=760, height=440) -> str:
    """Log–log scatter. points: {x, y, color, label, marker, title, front}."""
    if not points:
        return "<p class='muted'>no data yet</p>"
    L, R, T, B = 64, 20, 16, 52
    xs = [p["x"] for p in points if p["x"] > 0]
    ys = [p["y"] for p in points if p["y"] > 0]
    xlo, xhi = min(xs) / 1.5, max(xs) * 1.5
    ylo, yhi = min(ys) / 1.5, max(ys) * 1.5
    def X(v): return L + (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) * (width - L - R)
    def Y(v): return T + (1 - (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))) * (height - T - B)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(ylab)} against {esc(xlab)}">']
    for t in log_ticks(xlo, xhi):
        if xlo <= t <= xhi:
            out.append(f'<line x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{height-B}" class="grid"/>'
                       f'<text x="{X(t):.1f}" y="{height-B+18}" class="tick" text-anchor="middle">{t:g}</text>')
    for t in log_ticks(ylo, yhi):
        if ylo <= t <= yhi:
            out.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{width-R}" y2="{Y(t):.1f}" class="grid"/>'
                       f'<text x="{L-8}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    out.append(f'<text x="{(L+width-R)/2:.0f}" y="{height-8}" class="axis" text-anchor="middle">{esc(xlab)}</text>')
    out.append(f'<text transform="translate(14,{(T+height-B)/2:.0f}) rotate(-90)" class="axis" text-anchor="middle">{esc(ylab)}</text>')
    front = [p for p in points if p.get("front")]
    if len(front) > 1:
        front.sort(key=lambda p: p["x"])
        out.append('<polyline class="front" points="' + " ".join(f"{X(p['x']):.1f},{Y(p['y']):.1f}" for p in front) + '"/>')
    for p in points:
        if p["x"] <= 0 or p["y"] <= 0:
            continue
        x, y = X(p["x"]), Y(p["y"])
        m = p.get("marker", "o")
        title = f"<title>{esc(p['title'])}</title>"
        if m == "s":
            out.append(f'<rect x="{x-4.5:.1f}" y="{y-4.5:.1f}" width="9" height="9" fill="{p["color"]}" class="mk">{title}</rect>')
        elif m == "d":
            out.append(f'<polygon points="{x:.1f},{y-6:.1f} {x+6:.1f},{y:.1f} {x:.1f},{y+6:.1f} {x-6:.1f},{y:.1f}" fill="{p["color"]}" class="mk">{title}</polygon>')
        elif m == "t":
            out.append(f'<polygon points="{x:.1f},{y-6:.1f} {x+5.5:.1f},{y+4:.1f} {x-5.5:.1f},{y+4:.1f}" fill="{p["color"]}" class="mk">{title}</polygon>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{p["color"]}" class="mk">{title}</circle>')
        if p.get("label"):
            out.append(f'<text x="{x+7:.1f}" y="{y-6:.1f}" class="pt">{esc(p["label"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


def lines_svg(series: list[dict], xlab: str, ylab: str, width=760, height=440) -> str:
    """Log–log line chart. series: {name, color, dash, pts:[(x,y)], marker}."""
    pts = [p for s in series for p in s["pts"]]
    if not pts:
        return "<p class='muted'>no data yet</p>"
    L, R, T, B = 64, 20, 16, 52
    xlo, xhi = min(p[0] for p in pts) / 1.4, max(p[0] for p in pts) * 1.4
    ylo, yhi = min(p[1] for p in pts) / 1.5, max(p[1] for p in pts) * 1.5
    def X(v): return L + (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) * (width - L - R)
    def Y(v): return T + (1 - (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))) * (height - T - B)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(ylab)} against {esc(xlab)}">']
    for t in log_ticks(xlo, xhi):
        if xlo <= t <= xhi:
            out.append(f'<line x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{height-B}" class="grid"/>'
                       f'<text x="{X(t):.1f}" y="{height-B+18}" class="tick" text-anchor="middle">{t:g}</text>')
    for t in log_ticks(ylo, yhi):
        if ylo <= t <= yhi:
            out.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{width-R}" y2="{Y(t):.1f}" class="grid"/>'
                       f'<text x="{L-8}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    out.append(f'<text x="{(L+width-R)/2:.0f}" y="{height-8}" class="axis" text-anchor="middle">{esc(xlab)}</text>')
    out.append(f'<text transform="translate(14,{(T+height-B)/2:.0f}) rotate(-90)" class="axis" text-anchor="middle">{esc(ylab)}</text>')
    for s in series:
        p = sorted(s["pts"])
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        out.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2"{dash} points="'
                   + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in p) + '"/>')
        for x, y in p:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{s["color"]}"><title>{esc(s["name"])}: {y:.3g} at {x:.3g}</title></circle>')
        x, y = p[-1]
        out.append(f'<text x="{X(x)+8:.1f}" y="{Y(y)+4:.1f}" class="pt" fill="{s["color"]}">{esc(s["name"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


# -------------------------------------------------------------------- page
def build(summary: Path, out: Path) -> None:
    rows = json.load(open(summary / "rows.json"))
    pareto = json.load(open(summary / "pareto.json"))
    ladder = json.load(open(summary / "ladder.json"))
    derived = json.load(open(summary / "derived.json")) if (summary / "derived.json").exists() else []
    s_rows = [r for r in rows if r["l2_max"] == r["l2_max"] and r["l2_max"] != float("inf")]
    h_rows = [r for r in rows if r["l2_max"] != r["l2_max"]]
    hosts = sorted({r["host"] for r in rows})

    # ---- piece S per case
    s_sections = []
    by_case = defaultdict(list)
    for r in s_rows:
        by_case[r["case"]].append(r)
    for case in ("baroclinic_igw", "basin_seiche", "lock_exchange"):
        rs = by_case.get(case, [])
        key = next((k for k in pareto["front"] if k.startswith(case + "|")), None)
        front = {(f["scheme"], f["solver"], f["cfl"]) for f in pareto["front"].get(key, [])} if key else set()
        fast = (pareto["fastest_at_tol"].get(key) or []) if key else []
        pts = []
        for r in rs:
            pts.append({"x": r["sim_s_per_wall_s"], "y": r["l2_max"],
                        "color": SCHEME_COLOR.get(r["scheme"], "var(--ink)"),
                        "marker": {"none": "o", "multigrid": "s", "pcg_jacobi": "d"}.get(r["solver"], "o"),
                        "front": (r["scheme"], r["solver"], r["cfl"]) in front,
                        "label": f"CFL {r['cfl']:g}",
                        "title": f"{SCHEME_LABEL.get(r['scheme'], r['scheme'])} · {r['solver']} · CFL {r['cfl']:g}: L2 {r['l2_max']:.2e}, {r['wall_s']:.3g} s, {int(r['n_steps'])} steps"})
        table = ["<table><thead><tr><th>scheme</th><th>solver</th><th class='num'>CFL</th><th class='num'>steps</th><th class='num'>solver iters / step</th><th class='num'>wall [s]</th><th class='num'>±MAD</th><th class='num'>rel. L2 error</th><th class='num'>sim s / wall s</th></tr></thead><tbody>"]
        for r in sorted(rs, key=lambda r: (r["scheme"], r["solver"], r["cfl"])):
            cls = " class='front'" if (r["scheme"], r["solver"], r["cfl"]) in front else ""
            ips = r["solver_iters"] / r["n_steps"] if r["n_steps"] else float("nan")
            table.append(f"<tr{cls}><td><span class='sw' style='background:{SCHEME_COLOR.get(r['scheme'],'var(--ink)')}'></span>{esc(SCHEME_LABEL.get(r['scheme'], r['scheme']))}</td><td>{esc(r['solver'])}</td><td class='num'>{r['cfl']:g}</td><td class='num'>{int(r['n_steps'])}</td><td class='num'>{fmt(ips)}</td><td class='num'>{r['wall_s']:.4g}</td><td class='num'>{r['wall_mad_s']:.2g}</td><td class='num'>{r['l2_max']:.2e}</td><td class='num'>{r['sim_s_per_wall_s']:.4g}</td></tr>")
        table.append("</tbody></table>")
        note = ""
        if fast:
            rowsx = "".join(f"<tr><td>≤ {f['tol']:g}</td><td>{esc(SCHEME_LABEL.get(f['scheme'], f['scheme']))} · {esc(f['solver'])} · CFL {f['cfl']:g}</td>"
                            f"<td class='num'>{f['wall_s']:.3g} s</td><td class='num'>{f['l2']:.2e}</td></tr>" for f in fast)
            note = ("<div class='callout'><strong>Winner at each tolerance</strong> — the ranking changes with the "
                    "error you accept, which is the point of the plane."
                    f"<table><thead><tr><th>tolerance (rel. L2)</th><th>fastest configuration</th><th class='num'>wall</th><th class='num'>its error</th></tr></thead>"
                    f"<tbody>{rowsx}</tbody></table></div>")
        elif rs:
            note = "<p class='callout'>No configuration reaches relative L2 ≤ 10⁻¹ at this resolution.</p>"
        status = "" if len(rs) >= 31 else f"<span class='pill'>{len(rs)} / 31 configurations measured</span>"
        s_sections.append(f"<h3>{esc(CASE_LABEL.get(case, case))} {status}</h3>"
                          + (scatter_svg(pts, "simulated seconds per wall-clock second (RTX 5090, CUDA)", "relative L2 error against the converged reference") if pts else "<p class='muted'>Not started yet — the sweep runs case by case.</p>")
                          + note + ("".join(table) if rs else ""))

    # ---- piece H ladder chart per problem
    h_sections = []
    by_prob = defaultdict(list)
    for r in h_rows:
        by_prob[(r["scheme"], r["solver"])].append(r)
    for (scheme, solver), rs in sorted(by_prob.items()):
        series = {}
        for r in rs:
            cells = r["nx"] ** 2 * r["nz"]
            ms = 1e3 * r["wall_s"] / r["n_steps"]
            if r["backend"] in CPU:
                # best thread count per grid on the CPU node; serial as its own line
                if r["backend"] == "fortran_serial":
                    name = f"{DEVICE.get(r['host'], r['host'])} · serial"
                else:
                    name = f"{DEVICE.get(r['host'], r['host'])} · OpenMP (best threads)"
                key = (name, r["host"])
                cur = series.setdefault(key, {"name": name, "color": BACKEND_COLOR[r["backend"]],
                                              "dash": r["host"] == "gpgpu", "pts": {}})
                cur["pts"][cells] = min(cur["pts"].get(cells, 1e99), ms)
            else:
                name = f"{DEVICE.get(r['host'], r['host'])} · {BACKEND_LABEL[r['backend']]}"
                cur = series.setdefault((name, r["host"]), {"name": name, "color": BACKEND_COLOR[r["backend"]],
                                                            "dash": r["host"] == "gpgpu", "pts": {}})
                cur["pts"][cells] = ms
        ser = [{"name": s["name"], "color": s["color"], "dash": s["dash"], "pts": list(s["pts"].items())}
               for s in series.values()]
        iters = sorted({(int(r["nx"]), int(r["solver_iters"])) for r in rs})
        it_note = ""
        if solver != "none":
            it_note = "<p class='muted'>Solver iterations per 50 steps, identical on every backend and node: " + ", ".join(f"{nx}²: {it}" for nx, it in iters) + ".</p>"
        h_sections.append(f"<h3>{esc(SCHEME_LABEL.get(scheme, scheme))}" + (f" · {esc(solver)}" if solver != "none" else "") + "</h3>"
                          + lines_svg(ser, "cells (nx² × 30)", "milliseconds per time step") + it_note)

    # ---- speed-ups (R8-1)
    sp_rows = ["<table><thead><tr><th class='num'>nx</th><th>scheme · solver</th><th>device · backend</th><th class='num'>ms / step</th><th>vs best CPU configuration (EPYC 9655)</th><th class='num'>vs serial</th></tr></thead><tbody>"]
    for sp in sorted(ladder, key=lambda s: (s["scheme"], s["solver"], s["nx"], s["gpu_host"], s["gpu_backend"])):
        sp_rows.append(f"<tr><td class='num'>{sp['nx']}</td><td>{esc(SCHEME_LABEL.get(sp['scheme'], sp['scheme']))} · {esc(sp['solver'])}</td><td>{esc(DEVICE.get(sp['gpu_host'], sp['gpu_host']))} · {esc(BACKEND_LABEL.get(sp['gpu_backend'], sp['gpu_backend']))}</td><td class='num'>{sp['gpu_ms_per_step']:.3g}</td><td><strong>{sp['speedup_vs_best_cpu']:.3g}×</strong> <span class='muted'>({esc(sp['best_cpu'])})</span></td><td class='num'>{fmt(sp['speedup_vs_serial']) if sp['speedup_vs_serial'] else '—'}</td></tr>")
    sp_rows.append("</tbody></table>")

    # ---- CPU scaling (geo85) for fb at each grid
    cpu_series = {}
    for r in h_rows:
        if r["host"] == "geo85" and r["scheme"] == "fb":
            nx = int(r["nx"])
            name = f"{nx}² × 30"
            cur = cpu_series.setdefault(nx, {"name": name, "color": f"var(--g{[100,200,400,1000,2000].index(nx) if nx in (100,200,400,1000,2000) else 0})", "pts": {}})
            t = 1 if r["backend"] == "fortran_serial" else int(r["threads"])
            if r["backend"] == "fortran_serial":
                cur["serial"] = r["wall_s"]
            else:
                cur["pts"][t] = r["wall_s"]
    cpu_ser = []
    for nx, s in sorted(cpu_series.items()):
        base = s.get("serial") or (s["pts"].get(1))
        if not base:
            continue
        cpu_ser.append({"name": s["name"], "color": s["color"], "pts": [(t, base / w) for t, w in s["pts"].items()]})
    cpu_chart = lines_svg(cpu_ser, "OpenMP threads (EPYC 9655, one exclusive node)", "speed-up over the serial build") if cpu_ser else ""

    counts = {"S": len(s_rows), "H": len(h_rows)}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<title>Ocean Time-Stepping Benchmark</title>
<meta name="description" content="Scheme × backend × hardware benchmark of a controlled ocean model core">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap">
<style>
:root {{
  --ground:#f3f5f7; --paper:#ffffff; --ink:#151a20; --ink-2:#4b5563; --ink-3:#8b95a3; --rule:#d9dee5;
  --accent:#0b5d73; --accent-soft:#dfeef2;
  --c-cuda:#0b5d73; --c-acc:#b5541c; --c-jax:#6b3fa0; --c-omp:#2d6a4f; --c-ser:#7c8794;
  --c-fb:#2d6a4f; --c-t05:#0b5d73; --c-t06:#3b7dd8; --c-t10:#6b3fa0; --c-split:#b5541c;
  --g0:#9fb8c8; --g1:#6f98b1; --g2:#3f7897; --g3:#1d5a7c; --g4:#0b3d5a;
  --front:#b5541c;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --ground:#10151b; --paper:#171d25; --ink:#e7ecf1; --ink-2:#b4bec9; --ink-3:#7b8794; --rule:#2b3542;
  --accent:#5fb6cc; --accent-soft:#193540;
  --c-cuda:#5fb6cc; --c-acc:#e8925a; --c-jax:#b48ee6; --c-omp:#6fc59a; --c-ser:#9aa5b1;
  --c-fb:#6fc59a; --c-t05:#5fb6cc; --c-t06:#7fb0f0; --c-t10:#b48ee6; --c-split:#e8925a;
  --g0:#3f5566; --g1:#4e7590; --g2:#6a97b4; --g3:#8fb9d3; --g4:#bcd9ea; --front:#e8925a;
}} }}
:root[data-theme="dark"] {{
  --ground:#10151b; --paper:#171d25; --ink:#e7ecf1; --ink-2:#b4bec9; --ink-3:#7b8794; --rule:#2b3542;
  --accent:#5fb6cc; --accent-soft:#193540;
  --c-cuda:#5fb6cc; --c-acc:#e8925a; --c-jax:#b48ee6; --c-omp:#6fc59a; --c-ser:#9aa5b1;
  --c-fb:#6fc59a; --c-t05:#5fb6cc; --c-t06:#7fb0f0; --c-t10:#b48ee6; --c-split:#e8925a;
  --g0:#3f5566; --g1:#4e7590; --g2:#6a97b4; --g3:#8fb9d3; --g4:#bcd9ea; --front:#e8925a;
}}
body {{ background:var(--ground); color:var(--ink); font-family:"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; font-size:15px; line-height:1.55; margin:0; }}
main {{ max-width:1060px; margin:0 auto; padding:40px 24px 80px; }}
h1 {{ font-family:Fraunces, Georgia, serif; font-weight:600; font-size:2.4rem; line-height:1.1; margin:0 0 6px; text-wrap:balance; }}
h2 {{ font-family:Fraunces, Georgia, serif; font-weight:500; font-size:1.55rem; margin:56px 0 12px; padding-top:18px; border-top:1px solid var(--rule); text-wrap:balance; }}
h3 {{ font-size:1.05rem; font-weight:600; margin:32px 0 8px; }}
p {{ max-width:70ch; }}
.lede {{ font-size:1.1rem; color:var(--ink-2); max-width:72ch; }}
.eyebrow {{ font-family:"IBM Plex Mono", monospace; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); }}
.grid {{ stroke:var(--rule); stroke-width:1; }}
.tick {{ font-family:"IBM Plex Mono", monospace; font-size:11px; fill:var(--ink-3); }}
.axis {{ font-size:12px; fill:var(--ink-2); }}
.pt {{ font-family:"IBM Plex Mono", monospace; font-size:10.5px; fill:var(--ink-2); }}
.mk {{ stroke:var(--paper); stroke-width:1; }}
.front {{ fill:none; stroke:var(--front); stroke-width:1.5; stroke-dasharray:3 3; }}
svg {{ width:100%; height:auto; background:var(--paper); border:1px solid var(--rule); border-radius:4px; display:block; }}
.table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem; margin:10px 0 4px; background:var(--paper); }}
th, td {{ padding:6px 10px; border-bottom:1px solid var(--rule); text-align:left; vertical-align:top; }}
th {{ font-weight:600; color:var(--ink-2); font-size:.8rem; letter-spacing:.03em; text-transform:uppercase; }}
td.num, th.num {{ text-align:right; font-family:"IBM Plex Mono", monospace; font-variant-numeric:tabular-nums; }}
tr.front td {{ background:var(--accent-soft); }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.pill {{ font-family:"IBM Plex Mono", monospace; font-size:.72rem; background:var(--accent-soft); color:var(--accent); padding:2px 8px; border-radius:999px; margin-left:8px; vertical-align:middle; }}
.callout {{ border-left:3px solid var(--accent); padding:6px 14px; background:var(--paper); }}
.muted {{ color:var(--ink-3); font-size:.9rem; }}
.facts {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:14px; margin:28px 0 8px; }}
.fact {{ background:var(--paper); border:1px solid var(--rule); border-radius:6px; padding:14px 16px; }}
.fact .k {{ font-family:"IBM Plex Mono", monospace; font-size:.74rem; color:var(--ink-3); letter-spacing:.06em; text-transform:uppercase; }}
.fact .v {{ font-family:Fraunces, Georgia, serif; font-size:1.7rem; line-height:1.15; margin-top:4px; }}
.fact .s {{ color:var(--ink-2); font-size:.86rem; margin-top:4px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:.86rem; color:var(--ink-2); margin:8px 0 0; }}
ol.findings li {{ margin-bottom:10px; max-width:78ch; }}
code {{ font-family:"IBM Plex Mono", monospace; font-size:.86em; }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; }} }}
</style>
<main>
<div class="eyebrow">cfd_exp · tier 2 · docs/04 design v2 · generated {stamp}</div>
<h1>Ocean Time-Stepping Benchmark</h1>
<p class="lede">One hydrostatic ocean core — z-level partial cells over real bathymetry, closed walls, flux-form advection, T/S with TEOS-10 — implemented five times (Fortran serial · Fortran OpenMP · OpenACC · CUDA · JAX) and stepped five ways (forward–backward, θ = 0.5 / 0.6 / 1.0, split-explicit). Every implementation agrees with the NumPy oracle to 10⁻¹² per step. What differs is only speed, and what the paper asks is: <em>at a fixed error, which combination of scheme, implementation and device reaches the answer first?</em></p>

<div class="facts">
  <div class="fact"><div class="k">Piece S · Pareto plane</div><div class="v">{counts['S']} / 93</div><div class="s">scheme × solver × CFL × case at 400²×30, RTX 5090 CUDA, full physical horizon, error against a converged reference</div></div>
  <div class="fact"><div class="k">Piece H · hardware ladder</div><div class="v">{counts['H']}</div><div class="s">byte-identical 50-step problems, 100²–2000²×30, on {", ".join(DEVICE.get(h, h) for h in hosts)}</div></div>
  <div class="fact"><div class="k">Protocol</div><div class="v">5 × median</div><div class="s">warm-up discarded, five repeats, median ± MAD, exclusive nodes (PBS <code>place=excl</code>, idle GPU asserted)</div></div>
  <div class="fact"><div class="k">Verification</div><div class="v">35 / 35</div><div class="s">R2 gate: five backends × seven configurations on the RTX 5090 node; 21/21 on the H100 node; 14/14 on the EPYC node</div></div>
</div>

<h2>1. Piece S — error against time to solution</h2>
<p>Each point is one configuration integrated to the case's physical horizon (one internal-wave period, two seiche periods, six hours of lock exchange). The error is the relative L2 distance of the final state from a converged forward–backward run at CFL 0.1 on the same grid, so it measures the time discretisation alone. Marker shape is the free-surface solver (circle: none, square: multigrid, diamond: PCG-Jacobi); colour is the scheme; the dashed line joins the Pareto front — configurations no other configuration beats on both axes. Highlighted rows are on the front.</p>
<div class="legend">{" ".join(f"<span><span class='sw' style='background:{c}'></span>{esc(SCHEME_LABEL[k])}</span>" for k, c in SCHEME_COLOR.items())}</div>
{"".join(s_sections)}

<h2>2. Piece H — cost per step across hardware</h2>
<p>The same binary problem (same domain file, same initial state, same dt) is stepped 50 times on every backend and node. For the θ scheme the solver iteration counts agree exactly across nodes, so the ratios below compare devices, not algorithms. CPU lines show the best OpenMP thread count of the exclusive EPYC 9655 node at each grid; the serial build is a separate line because a 1-thread OpenMP run is not a serial baseline (it is 2× slower).</p>
<div class="legend">{" ".join(f"<span><span class='sw' style='background:{c}'></span>{esc(BACKEND_LABEL[k])}</span>" for k, c in BACKEND_COLOR.items())} <span>solid: H100 / EPYC 9655 · dashed: RTX 5090</span></div>
{"".join(h_sections)}

<h3>OpenMP scaling on one EPYC 9655 node, forward–backward</h3>
{cpu_chart}
<p class="muted">96 cores, 192 threads, <code>OMP_PROC_BIND=close OMP_PLACES=cores</code>. The optimum moves with the grid: 32–64 threads at 100²–200², 128 at 400²–1000², 192 only at 2000²×30.</p>

<h2>3. Speed-ups, with both baselines named</h2>
<p>"N× faster" is only meaningful with the baseline stated (rule R8-1). The primary baseline is the best CPU configuration of the exclusive EPYC 9655 node; the serial column is the single-core Fortran build, which is what the loose reading of the literature usually means.</p>
<div class="table-wrap">{"".join(sp_rows)}</div>

<h2>4. What the sweep found so far</h2>
<ol class="findings">
<li><strong>The verification gate cannot see a wrong reference.</strong> Piece S's first case showed the θ family 130 % away from the forward–backward reference at every CFL. The 3D forward–backward step had been applying only (1−θ) of the barotropic pressure gradient because it read the configured θ = 0.5; the NumPy oracle carried the defect and every backend reproduced it to 10⁻¹², so the gate passed. Fixed in all five implementations; piece S was restarted from scratch. (docs/90 N14)</li>
<li><strong>Split-explicit needs its two modes to use the same Coriolis operator.</strong> Over a seamount the depth integral of the 3D masked Coriolis average and the 2D transport average differ by O(1); subtracting one and adding the other left a structural error that does not vanish as dt → 0 (u error 158 % in the closed basin, 8.6 % after the fix, against 4.3 % for forward–backward). The spec now defines the variable-depth form. (N15)</li>
<li><strong>At 2000²×30 on a 32 GB consumer card, managed memory is the only GPU path that runs — and it loses to the CPU.</strong> OpenACC (<code>-gpu=mem:managed</code>) takes 169 s per 50 steps against 149 s for the same node's 16-core CPU, while CUDA and JAX cannot allocate at all. The H100 solves the same problem in 6.9 s. At this size "the GPU is faster" is a statement about card memory, not about arithmetic. (N24)</li>
<li><strong>The JAX build of the full v0.6 core does not fit the largest grid.</strong> At 2000²×30 (1.2×10⁸ cells) with the TKE closure it asks XLA for a single 28.0 GiB buffer on top of its working set and fails on an 80 GB H100; the same algorithm in CUDA runs in 45.9 GB (383 bytes per cell), and OpenACC and Fortran run too. Removing the unrolled mixing-length march (a 60-fold graph expansion) cut the request by only 0.7 GiB: the rest is XLA's own intermediate buffering of a whole time step compiled as one function. (N21)</li>
<li><strong>Split-explicit without a barotropic time filter is unstable over topography once it actually splits.</strong> With n_split = 1 it coincides with forward–backward (2.6 % at CFL 0.5); at CFL 2, 8 and 32 (n_split = 3, 10, 39) it diverges exponentially, with or without rotation. The simple substep average left as a configuration axis by the spec makes it diverge faster: a real filter (Shchepetkin & McWilliams 2005) averages transports and η together past t + dt. Its Pareto contribution therefore collapses onto the forward–backward point — the property of filter-free mode splitting, not a bug. (N16)</li>
<li><strong>Multigrid on rough bathymetry is not "10 V-cycles".</strong> On the smooth seamount the θ solve needs 18–20 V-cycles per step at CFL 4 (4 at CFL 0.5); on the rough lock-exchange bathymetry it needs ~60. The coefficient contrast of the free-surface Helmholtz operator, not the grid size, sets the count.</li>
<li><strong>JAX ties or beats hand-written CUDA on the PCG solve.</strong> On the H100 at 400²×30 the JAX PCG-Jacobi step costs 10.9 ms against 11.7 ms for CUDA: the JAX <code>while_loop</code> keeps the convergence test on the device, whereas the CUDA driver copies the residual back every iteration. For everything without a host-synchronised loop CUDA leads JAX by 1.4–9×, the gap closing with grid size.</li>
<li><strong>OpenACC is the slowest GPU path at small grids and loses to the CPU node on multigrid below 400².</strong> Managed memory and per-loop kernel launches cost 7–35 ms per step at 100²×30 on the H100 where CUDA needs 0.5–2.7 ms; at 2000²×30 the gap is 1.4–1.8×.</li>
<li><strong>Cross-node comparisons need a controlled CPU threshold.</strong> The OpenMP build disables its parallel regions below 65 536 horizontal points, so the first 100² and 200² ladders ran effectively serial; they were re-measured with the threshold off (the two are separate rows above).</li>
</ol>

<h2>5. Method</h2>
<p>Design: <code>docs/04_design_v2.md</code>. Discretisation: <code>docs/03</code> §10 (v0.5). Timing rule R7: monotonic clocks, device synchronisation before every stop, five repeats after one discarded warm-up, median ± MAD, exclusive placement (<code>PBS -l place=excl</code> on the EPYC cluster; the RTX 5090 node's idle GPU asserted and pinned with <code>CUDA_VISIBLE_DEVICES</code>). The H100 host CPU is shared with other tenants, so no CPU time from that node appears anywhere. Cross-node device comparisons are allowed only for byte-identical problems with matching iteration counts, and always name both devices.</p>
<p class="muted">Derived times to solution for the backends that did not run piece S (piece-H cost ratio × CUDA time; validated to 1.1 % in docs/26): {len(derived)} rows in <code>output/tier2_summary/derived.json</code>.</p>
</main>
"""
    out.write_text(page)
    print(f"{out}: S={counts['S']} H={counts['H']} rows")


def main() -> int:
    ap = argparse.ArgumentParser(description="render the tier-2 summary as HTML")
    ap.add_argument("--summary", type=Path, default=Path("output/tier2_summary"))
    ap.add_argument("--out", type=Path, default=Path("output/tier2_report.html"))
    args = ap.parse_args()
    build(args.summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
