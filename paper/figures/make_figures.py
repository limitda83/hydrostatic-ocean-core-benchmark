#########################################################################
#  Module: make_figures                                                  #
#  Description: Build every manuscript figure from paper/cases/ data    #
#               only (no expr/ access), so the figures are reproducible #
#               from the promoted case folders alone.                    #
#  Pipeline: paper/cases/*  ->  paper/cageo/manuscript/figs/*.pdf        #
#########################################################################
"""Figures for the Computers & Geosciences manuscript.

Every number drawn here comes from paper/cases/.  The hardware ladder is
re-collected with tools/tier2_collect.py from the promoted CSVs so that the
figure and the audited tables share one provenance.
"""
from __future__ import annotations
import csv, json, logging, subprocess, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker, matplotlib.lines, matplotlib.patches

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "paper" / "cases"
FIGS = ROOT / "paper" / "figures"
SUMMARY = ROOT / "output" / "tier2_summary_cases"
DEVICE = {("ktcloud", "cuda"): "H100 CUDA", ("ktcloud", "openacc"): "H100 OpenACC", ("ktcloud", "jax"): "H100 JAX",
          ("gpgpu", "cuda"): "RTX 5090 CUDA", ("gpgpu", "openacc"): "RTX 5090 OpenACC", ("gpgpu", "jax"): "RTX 5090 JAX",
          ("geo85", "fortran_omp"): "EPYC 9655 OpenMP (best threads)", ("geo85", "fortran_serial"): "EPYC 9655 serial"}
STYLE = {"H100 CUDA": ("#c0392b", "o", "-"), "H100 OpenACC": ("#c0392b", "s", "--"), "H100 JAX": ("#c0392b", "^", "-."),
         "RTX 5090 CUDA": ("#1f5fa8", "o", "-"), "RTX 5090 OpenACC": ("#1f5fa8", "s", "--"), "RTX 5090 JAX": ("#1f5fa8", "^", "-."),
         "EPYC 9655 OpenMP (best threads)": ("k", "D", "-"), "EPYC 9655 serial": ("k", "x", ":")}
def savefig(fig, name):
    """Write the PDF for LaTeX and a 300-dpi PNG for the DOCX build."""
    fig.savefig(FIGS / f"{name}.pdf"); fig.savefig(FIGS / f"{name}.png", dpi=300)


plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 150,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "semibold", "axes.titlepad": 6, "axes.edgecolor": "0.3",
                     "xtick.color": "0.3", "ytick.color": "0.3", "axes.labelcolor": "0.15",
                     "grid.color": "0.85", "grid.linewidth": 0.5, "legend.handlelength": 2.2})
RED, BLUE, GREY = "#c0392b", "#1f5fa8", "0.55"          # H100, RTX 5090, CPU — fixed across every figure
# scheme palette (Okabe–Ito, colour-blind safe) for the accuracy plane
SCHEME = {"fb": ("#0072B2", "o", "forward–backward"), "split": ("#009E73", "s", "split-explicit"),
          "theta0.5": ("#D55E00", "^", r"$\theta=0.5$"), "theta0.6": ("#E69F00", "v", r"$\theta=0.6$"),
          "theta1.0": ("#CC79A7", "D", r"$\theta=1.0$")}


def pareto_front(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left envelope of (time, error) points: the non-dominated set, sorted by time."""
    best, front = float("inf"), []
    for w, e in sorted(pts):
        if e < best:
            front.append((w, e)); best = e
    return front


def collect() -> list[dict]:
    """Re-run the collector on the promoted ladder case so rows.json is case-only."""
    files = sorted((CASES / "hardware_ladder" / "data").rglob("*.csv"))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "tier2_collect.py"), *map(str, files), "--out", str(SUMMARY)],
                   check=True, capture_output=True)
    return json.load(open(SUMMARY / "rows.json"))


def canon_host(h: str) -> str:
    return "geo85" if h.startswith("node") else h


def ladder(rows: list[dict], case: str, scheme: str, solver: str, cfl: float) -> dict[str, dict[int, float]]:
    """ms/step per device label per nx: fastest thread count, fastest equal-provenance duplicate."""
    best: dict[tuple, float] = {}
    for r in rows:
        if (r["case"], r["scheme"], r["solver"], float(r["cfl"]), int(r["n_steps"])) != (case, scheme, solver, cfl, 50):
            continue
        if not (r["wall_s"] < float("inf")) or r["wall_s"] != r["wall_s"]:
            continue
        key = (canon_host(r["host"]), r["backend"])
        if key not in DEVICE:
            continue
        k = (DEVICE[key], int(r["nx"]))
        ms = 1e3 * r["wall_s"] / r["n_steps"]
        best[k] = min(best.get(k, ms), ms)
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for (lab, nx), ms in best.items():
        out[lab][nx] = ms
    return out


def multi_gpu_series() -> dict[str, dict[str, dict[int, float]]]:
    """Multi-card ms/step from paper/cases/multi_gpu (E19): {'RTX 5090 x2 (one node)': {case: {nx: ms}},
    'RTX 5090 x4 (one node)': ..., 'H100 x2 (two nodes)': ...}. Only rows whose state matched the single-card
    state bit for bit are used; for the H100 the CUDA-aware MPI exchange (the faster of the two measured paths)
    is taken."""
    out: dict[str, dict[str, dict[int, float]]] = {"RTX 5090 ×2 (one node)": {}, "RTX 5090 ×4 (one node)": {},
                                                    "H100 ×2 (two nodes)": {}}
    for f in (CASES / "multi_gpu" / "data").rglob("e19_*.csv"):
        for r in csv.DictReader(open(f)):
            nd = int(r["n_devices"])
            if r["status"] != "ok" or nd < 2 or r["max_abs_diff_vs_1dev"] != "0.000e+00":
                continue
            if r["devices"].startswith("mpi") and not r["devices"].endswith(":device"):
                continue
            lab = f"RTX 5090 ×{nd} (one node)" if r["host"] == "gpgpu" else "H100 ×2 (two nodes)"
            if lab not in out:
                continue
            ms = 1e3 * float(r["wall_s"]) / float(r["n_steps"])
            case = "lock_exchange" if r["physics"] == "DC" else "lock_exchange_v06"
            out[lab].setdefault(case, {})[int(r["nx"])] = min(ms, out[lab].get(case, {}).get(int(r["nx"]), ms))
    return out


def fig_ladder(rows):
    """Two panels, shared legend below; device = colour, implementation = marker/line style."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.5), sharey=True)
    order = ["H100 CUDA", "H100 OpenACC", "H100 JAX", "RTX 5090 CUDA", "RTX 5090 OpenACC", "RTX 5090 JAX",
             "EPYC 9655 OpenMP (best threads)", "EPYC 9655 serial"]
    pretty = {"EPYC 9655 OpenMP (best threads)": "EPYC 9655, OpenMP (best thread count)", "EPYC 9655 serial": "EPYC 9655, serial (1 core)"}
    handles = {}
    two = multi_gpu_series()
    for ax, (case, title) in zip(axes, [("lock_exchange", "(a) dynamical core (DC)"),
                                        ("lock_exchange_v06", "(b) with TKE closure + 3rd-order TVD (DC+TKE)")]):
        lad = ladder(rows, case, "fb", "none", 0.5)
        for lab, col, mk in (("RTX 5090 ×2 (one node)", BLUE, "P"), ("RTX 5090 ×4 (one node)", BLUE, "X"),
                             ("H100 ×2 (two nodes)", RED, "P")):   # multi-card series (E19)
            ser = two.get(lab, {}).get(case, {})
            if ser:
                xs = sorted(ser); ys = [ser[x] for x in xs]
                h, = ax.plot(xs, ys, color=col, marker=mk, ls=":", ms=7, lw=1.4, mfc=col, mec="white", mew=0.6, label=lab, zorder=4, alpha=0.9)
                handles.setdefault(lab, h)
        for lab in order:
            if lab not in lad:
                continue
            c, m, ls = STYLE[lab]
            xs = sorted(lad[lab]); ys = [lad[lab][x] for x in xs]
            h, = ax.plot(xs, ys, color=c, marker=m, ls=ls, ms=6, lw=1.6, mew=0.8, mfc="white" if "OpenACC" in lab else c,
                         label=pretty.get(lab, lab), zorder=3 if "CUDA" in lab else 2)
            handles.setdefault(pretty.get(lab, lab), h)
        # the 32 GB limit of the consumer card: a shaded band at 2000² instead of arrows across the data
        ax.axvspan(1450, 2700, color="0.94", zorder=0)
        ax.text(2000, 0.5, "RTX 5090, 32 GB:\nCUDA / JAX out of memory;\nOpenACC managed memory,\npaging over PCIe (□)",
                ha="center", va="bottom", fontsize=6.0, color=BLUE, linespacing=1.25)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([100, 200, 400, 1000, 2000]); ax.set_xticklabels(["100²", "200²", "400²", "1000²", "2000²"])
        ax.set_xlim(85, 2700); ax.set_xlabel("horizontal grid (×30 levels)")
        ax.set_title(title, fontsize=9); ax.grid(True, which="major", lw=0.5, alpha=0.6); ax.grid(True, which="minor", lw=0.3, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("wall time per step [ms] (median of 5)")
    axes[0].set_ylim(0.4, 2e4)
    order_lab = ["H100 CUDA", "H100 OpenACC", "H100 JAX", "H100 ×2 (two nodes)", "RTX 5090 CUDA", "RTX 5090 OpenACC", "RTX 5090 JAX",
                 "RTX 5090 ×2 (one node)", "RTX 5090 ×4 (one node)", "EPYC 9655, OpenMP (best thread count)", "EPYC 9655, serial (1 core)"]
    hs = [handles[k] for k in order_lab if k in handles]; ls_ = [k for k in order_lab if k in handles]
    fig.legend(hs, ls_, loc="lower center", ncol=4, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.15, 1, 1)); savefig(fig, "fig_ladder"); plt.close(fig)


def fig_pareto(rows):
    """Error-vs-wall-time plane: gpgpu CUDA fp64 400²×30 runs to the physical horizon (n_steps > 50)."""
    pts = [r for r in rows if canon_host(r["host"]) == "gpgpu" and r["backend"] == "cuda" and int(r["nx"]) == 400
           and int(r["n_steps"]) > 50 and r["case"] in ("baroclinic_igw", "basin_seiche", "lock_exchange")]
    cases = [("baroclinic_igw", "(a) internal wave over a seamount"), ("basin_seiche", "(b) closed-basin seiche"),
             ("lock_exchange", "(c) lock exchange, rough bottom")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    handles = {}
    for ax, (case, title) in zip(axes, cases):
        cloud = []
        for r in pts:
            if r["case"] != case:
                continue
            err = float(r["l2_max"]); w = float(r["wall_s"])
            if not (err == err) or err == float("inf") or err > 10:
                continue
            c, m, lab = SCHEME[r["scheme"]]
            filled = r["solver"] != "pcg_jacobi"
            h = ax.scatter(w, err, c=c if filled else "white", edgecolors=c, marker=m, s=22, lw=0.9, zorder=3)
            handles.setdefault(lab, h); cloud.append((w, err))
        # the Pareto front (lower-left envelope) is the object the text reads off this plane
        front = pareto_front(cloud)
        if len(front) > 1:
            ax.plot([w for w, _ in front], [e for _, e in front], color="0.45", lw=0.8, ls="-", drawstyle="steps-post",
                    zorder=1, label="_front")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title, fontsize=8.5)
        for tol, txt in ((1e-1, "10⁻¹"), (1e-2, "10⁻²"), (1e-3, "10⁻³")):
            ax.axhline(tol, color="0.7", lw=0.5, ls="--", zorder=0)
        ax.grid(True, which="major", axis="x", lw=0.3, alpha=0.5)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(1.2e-4, 5)
    axes[0].set_ylabel("relative $L_2$ error vs converged reference")
    axes[1].set_xlabel("wall-clock time to the physical horizon [s]  (RTX 5090, CUDA, fp64, $400^2\\times30$)")
    axes[2].text(0.98, 0.97, "dashed: tolerances $10^{-1}$, $10^{-2}$, $10^{-3}$\ngrey step: Pareto front",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=6, color="0.35")
    # solver encoding as two extra legend entries (marker fill), then the five schemes
    solver_h = [matplotlib.lines.Line2D([], [], marker="o", color="0.4", mfc="0.4", ls="", ms=5, label="multigrid (filled)"),
                matplotlib.lines.Line2D([], [], marker="o", color="0.4", mfc="white", ls="", ms=5, label="PCG-Jacobi (open)")]
    fig.legend(list(handles.values()) + solver_h, list(handles.keys()) + [h.get_label() for h in solver_h],
               loc="lower center", ncol=7, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); savefig(fig, "fig_pareto"); plt.close(fig)


def fig_physics(rows):
    """Added physics per cell (ns) at 1000² fb, and closure share from the decomposition."""
    lad5 = ladder(rows, "lock_exchange", "fb", "none", 0.5); lad6 = ladder(rows, "lock_exchange_v06", "fb", "none", 0.5)
    labs = ["H100 CUDA", "H100 OpenACC", "H100 JAX", "RTX 5090 CUDA", "RTX 5090 OpenACC", "RTX 5090 JAX",
            "EPYC 9655 OpenMP (best threads)"]
    nx = 1000; cells = nx * nx * 30
    core = [1e6 * lad5[l][nx] / cells for l in labs]; added = [1e6 * (lad6[l][nx] - lad5[l][nx]) / cells for l in labs]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    y = list(range(len(labs)))
    ax.barh(y, core, color="0.78", label="dynamical core (DC)", height=0.72)
    ax.barh(y, added, left=core, color=RED, label="added physics (DC+TKE − DC)", height=0.72)
    for yi, c0, a0 in zip(y, core, added):
        ax.text(c0 + a0 + 0.15, yi, f"{c0 + a0:.1f}  (×{(c0 + a0) / c0:.1f})", va="center", fontsize=6, color="0.25")
    ax.set_yticks(y); ax.set_yticklabels([l.replace(" (best threads)", "\n(best threads)") for l in labs])
    ax.invert_yaxis(); ax.set_xlabel("ns per cell per step, $1000^2\\times30$, forward–backward   (label: total, × DC)")
    ax.set_xlim(0, 16.5); ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    ax.set_title("(a) where the physics penalty lands")
    # decomposition (gpgpu fb, both precisions) from the promoted decompose CSV:
    # variants v05 / tke / up3 / up3tvd / tke_up3tvd (docs/31 §3c definitions)
    dec = {}
    for r in csv.DictReader(open(CASES / "physics_cost" / "data" / "E14_v06_representative" / "decompose_gpgpu_fb_20260915.csv")):
        if r["backend"] not in ("cuda", "cuda_sp") or r["scheme"] != "fb":
            continue
        p = "fp32" if r["backend"] == "cuda_sp" else "fp64"
        key = (p, int(r["nx"]), r["case"].split(".")[1]); ms = 1e3 * float(r["wall_s"]) / float(r["n_steps"])
        dec[key] = min(dec.get(key, ms), ms)
    grids = [100, 400, 1000]; comp = {}
    for p in ("fp64", "fp32"):
        for g in grids:
            v = {k: dec.get((p, g, k)) for k in ("v05", "tke", "up3", "up3tvd", "tke_up3tvd")}
            if None in v.values():
                continue
            add = v["tke_up3tvd"] - v["v05"]
            closure = (v["tke"] - v["v05"]) / add; up3 = (v["up3"] - v["v05"]) / add; tvd = (v["up3tvd"] - v["up3"]) / add
            comp[(p, g)] = [closure, up3, tvd]
    x = 0; ticks = []; labels = []
    for p in ("fp64", "fp32"):
        for g in grids:
            if (p, g) not in comp:
                continue
            c = comp[(p, g)]; inter = 1 - sum(c)
            bottom = 0
            for val, col, nm in zip(c + [inter], [RED, "#E69F00", "#009E73", "0.8"], ["TKE closure", "3rd-order advection", "TVD limiter", "interaction"]):
                ax2.bar(x, 100 * val, bottom=100 * bottom, color=col, width=0.72, label=nm if x == 0 else None, edgecolor="white", lw=0.5); bottom += val
            ax2.text(x, 100 * c[0] / 2, f"{100 * c[0]:.0f}", ha="center", va="center", fontsize=6, color="white")
            ticks.append(x); labels.append(f"{g}²"); x += 1
        x += 0.6
    ax2.set_xticks(ticks); ax2.set_xticklabels(labels, fontsize=6.5); ax2.set_ylabel("% of added physics cost")
    ax2.set_ylim(0, 108)
    ax2.text(1.0, 104, "fp64", ha="center", va="bottom", fontsize=7, color="0.3"); ax2.text(4.6, 104, "fp32", ha="center", va="bottom", fontsize=7, color="0.3")
    ax2.set_title("(b) decomposition, RTX 5090, fb"); ax2.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(); savefig(fig, "fig_physics"); plt.close(fig)


def fig_precision():
    """fp64→fp32 speed-up of the whole step (E15) and mixed-precision step ratio (E18)."""
    def load(fn):
        return list(csv.DictReader(open(fn)))
    e15 = {"RTX 5090": load(CASES / "precision_axis" / "data" / "E15_precision" / "gpgpu_rtol6.csv"),
           "H100": load(CASES / "precision_axis" / "data" / "E15_precision" / "ktcloud_rtol6.csv")}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.45, 1]})
    combos = [("lock_exchange.rtol6", "fb", 400, "DC fb 400²"), ("lock_exchange_v06.rtol6", "fb", 400, "DC+TKE fb 400²"),
              ("lock_exchange.rtol6", "fb", 1000, "DC fb 1000²"), ("lock_exchange_v06.rtol6", "fb", 1000, "DC+TKE fb 1000²"),
              ("lock_exchange.rtol6", "theta0.5", 1000, "DC θ·mg 1000²"), ("lock_exchange_v06.rtol6", "theta0.5", 1000, "DC+TKE θ·mg 1000²")]
    w = 0.38
    for i, (dev, col) in enumerate((("RTX 5090", BLUE), ("H100", RED))):
        vals = []
        for case, sch, nx, _ in combos:
            t = {}
            for r in e15[dev]:
                if r["case"] == case and r["scheme"] == sch and int(r["nx"]) == nx and r["backend"] in ("cuda", "cuda_sp"):
                    t[r["backend"]] = min(t.get(r["backend"], 1e9), float(r["wall_s"]))
            vals.append(t["cuda"] / t["cuda_sp"] if "cuda" in t and "cuda_sp" in t else float("nan"))
        ax.bar([j + (i - 0.5) * w for j in range(len(combos))], vals, width=w, color=col, label=f"{dev}")
        for j, v in enumerate(vals):
            if v == v:
                ax.text(j + (i - 0.5) * w, v + 0.12, f"{v:.1f}", ha="center", fontsize=6, color="0.2", bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=4)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([c[3].replace(" fb ", "\nfb\n").replace(" θ·mg ", "\nθ·mg\n") for c in combos], fontsize=6.2)
    ax.set_ylabel("fp64 / fp32 wall time (whole-fp32 build)"); ax.axhline(2, color="0.5", lw=0.6, ls="--", zorder=1)
    ax.text(len(combos) - 0.55, 8.6, "dashed: 2× = half the bytes moved", ha="right", va="top", fontsize=6, color="0.4")
    ax.set_ylim(0, 8.9); ax.set_title("(a) whole-step fp32 speed-up, CUDA")
    # mixed precision from E18 ladders
    e18 = {"RTX 5090": load(CASES / "precision_axis" / "data" / "E18_mixed_precision" / "ladder_gpgpu_mixed_20260915.csv"),
           "H100": load(CASES / "precision_axis" / "data" / "E18_mixed_precision" / "ladder_ktcloud_mixed_20260915.csv")}
    mcombos = [("fb", "none", 400), ("fb", "none", 1000), ("theta0.5", "multigrid", 400), ("theta0.5", "multigrid", 1000)]
    hs = []
    for i, (dev, col) in enumerate((("RTX 5090", BLUE), ("H100", RED))):
        vals = []
        for sch, sol, nx in mcombos:
            t = {}
            for r in e18[dev]:
                if r["scheme"] == sch and r["solver"] == sol and int(r["nx"]) == nx:
                    t[r["backend"]] = min(t.get(r["backend"], 1e9), float(r["wall_s"]))
            f64 = t.get("cuda"); mix = t.get("cuda_mixed")
            vals.append(mix / f64 if f64 and mix else float("nan"))
        hs.append(ax2.bar([j + (i - 0.5) * w for j in range(len(mcombos))], vals, width=w, color=col, label=dev))
        for j, v in enumerate(vals):
            if v == v:
                ax2.text(j + (i - 0.5) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, color="0.2")
    ax2.set_xticks(range(len(mcombos))); ax2.set_xticklabels(["fb\n400²", "fb\n1000²", "θ·mg\n400²", "θ·mg\n1000²"], fontsize=6.5)
    ax2.set_ylabel("mixed / fp64 wall time"); ax2.set_ylim(0, 1.12); ax2.axhline(1, color="0.5", lw=0.6, ls="--")
    ax2.text(len(mcombos) - 0.55, 1.015, "1 = fp64 step", ha="right", va="bottom", fontsize=6, color="0.4")
    ax2.set_title("(b) fp64 core + fp32 closure (DC+TKE)")
    fig.legend(hs, ["RTX 5090", "H100"], loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); savefig(fig, "fig_precision"); plt.close(fig)


def fig_mxl():
    """Gain of the O(nz) mixing length over the O(nz²) one, per configuration (from mxl_cost.json of the case collect)."""
    d = json.load(open(SUMMARY / "mxl_cost.json"))
    items = []
    for r in d:
        if r["scheme"] != "fb" or r["nz"] != 30:
            continue
        host = canon_host(r["host"]); be = r["backend"]; th = r["threads"]; nx = r["nx"]
        if host == "gpgpu" and be.startswith("fortran"):
            continue  # the RTX node's own CPU is not a baseline in the manuscript
        dev = {"ktcloud": "H100", "gpgpu": "RTX 5090" if be in ("cuda", "cuda_sp", "openacc", "jax") else "TR 9955WX", "geo85": "EPYC 9655"}[host]
        prec = "fp32" if be == "cuda_sp" else "fp64"
        be_lab = {"cuda": "CUDA", "cuda_sp": "CUDA", "openacc": "OpenACC", "jax": "JAX", "fortran_omp": f"OpenMP ×{th}", "fortran_serial": "serial"}[be]
        items.append((f"{dev} · {be_lab} · {prec} · {nx}²", r["ratio"]))
    # keep one entry per label (fastest-of-duplicates already applied by the collector)
    seen = {}
    for lab, v in items:
        seen[lab] = v
    items = sorted(seen.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.4, 0.19 * len(items) + 1.4))
    cols = [BLUE if "5090" in l and "fp64" in l else RED if "H100" in l else GREY for l, _ in items]
    ax.barh(range(len(items)), [v for _, v in items], color=cols, height=0.74)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=5.8, color="0.25")
    ax.set_yticks(range(len(items))); ax.set_yticklabels([l for l, _ in items], fontsize=6)
    ax.axvline(1, color="k", lw=0.7); ax.set_xlim(0, 2.7)
    ax.set_xlabel("step time ratio, integral / recursive (>1: recursive form faster)")
    ax.text(1.03, len(items) - 0.35, "1 = no gain", ha="left", va="bottom", fontsize=6, color="0.3")
    ax.set_title("value of the $O(n_z)$ mixing length, by configuration", fontsize=8)
    patches = [matplotlib.patches.Patch(color=BLUE, label="RTX 5090 fp64"),
               matplotlib.patches.Patch(color=RED, label="H100 (fp64 and fp32)"),
               matplotlib.patches.Patch(color=GREY, label="EPYC 9655 CPU; RTX 5090 fp32")]
    fig.legend(handles=patches, frameon=False, fontsize=6.5, loc="lower center", ncol=3, columnspacing=1.6, handlelength=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); savefig(fig, "fig_mxl"); plt.close(fig)


def fig_boundary():
    """Host round trip of the closure footprint vs in-place closure cost (E17 + E14/E18 decomposition)."""
    def closure_ms(dev, prec, nx):
        fn = {"gpgpu": "decompose_gpgpu_fb_20260915.csv", "ktcloud": "decompose_ktcloud_fb_20260915.csv"}[dev]
        be = "cuda_sp" if prec == "fp32" else "cuda"; t = {}
        for r in csv.DictReader(open(CASES / "physics_cost" / "data" / "E14_v06_representative" / fn)):
            if r["backend"] == be and int(r["nx"]) == nx and r["scheme"] == "fb":
                t[r["case"].split(".")[1]] = 1e3 * float(r["wall_s"]) / float(r["n_steps"])
        return t["tke"] - t["v05"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for dev, host, col in (("RTX 5090", "gpgpu", BLUE), ("H100", "ktcloud", RED)):
        rows = list(csv.DictReader(open(CASES / "module_boundary" / "data" / "E17_module_boundary" / f"{host}_boundary.csv")))
        for prec, item, ls, m in (("fp32", 4, "-", "o"), ("fp64", 8, "--", "s")):
            xs, ys = [], []
            for nx in (100, 400, 1000):
                rt = [float(r["median_s"]) for r in rows if int(r["itemsize"]) == item and int(r["nx"]) == nx
                      and r["case"].startswith("C_") and r["case"].endswith("pinned")]
                if not rt:
                    continue
                xs.append(nx); ys.append(1e3 * rt[0] / closure_ms(host, prec, nx))
            ax.plot(xs, ys, color=col, ls=ls, marker=m, ms=4.5, lw=1.2, label=f"{dev} {prec}", mfc=col if prec == "fp32" else "white", mew=1.2)
    ax.axhline(1, color="k", lw=0.7); ax.set_xscale("log"); ax.set_xticks([100, 400, 1000]); ax.set_xticklabels(["100²", "400²", "1000²"])
    ax.text(1000, 1.04, "round trip = closure cost", ha="right", va="bottom", fontsize=6, color="0.3")
    ax.axhspan(1, 3, color="0.95", zorder=0); ax.text(1000, 1.72, "above the line: moving the data costs\nmore than computing the closure", ha="right", va="top", fontsize=6, color="0.4")
    ax.set_ylim(0, 3); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("host round trip / in-place closure"); ax.set_xlabel("grid (30 levels)"); ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.set_title("cost of reaching a host-side module, pinned PCIe 5.0", fontsize=8)
    fig.tight_layout(); savefig(fig, "fig_boundary"); plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = collect()
    fig_ladder(rows); fig_pareto(rows); fig_physics(rows); fig_precision(); fig_mxl(); fig_boundary()
    for p in sorted(FIGS.glob("*.pdf")):
        logging.info(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
