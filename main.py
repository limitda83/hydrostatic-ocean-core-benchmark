#!/usr/bin/env python3
#########################################################################
#  Module: main                                                         #
#  Description: Experiment driver for cfd_exp. Subcommands:             #
#                 run     - one simulation, writes NetCDF + manifest    #
#                 verify  - convergence study (space or time)           #
#                 bench   - error-vs-wallclock frontier over a dt sweep #
#  Pipeline: config -> grid/case/scheme -> integrate -> diagnostics/IO  #
#########################################################################

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from libs.core.cases import build_case
from libs.core.diagnostics import (summarize, total_energy, total_mass,
                                   wave_errors)
from libs.core.driver3d import simulate3d
from libs.core.grid import build_grid
from libs.core.schemes import (Stepper, explicit_dt_max, physics_from_config,
                               scheme_params_from_config)
from libs.io.manifest import write_manifest, write_metrics
from libs.io.netcdf_writer import write_state
from libs.utils.config import Config, load_config
from libs.utils.logging_setup import setup_logging
from libs.utils.timing import is_bench_node, timed_repeat

LOGGER = logging.getLogger("cfd_exp")


# --------------------------------------------------------------------- helpers
def _run_dir(config: Config, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(config.get("paths.output_dir")) / f"{tag}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_dt(config: Config, grid, physics, cfl: float | None) -> tuple[float, float]:
    """Return (dt, dt_max_explicit). dt = cfl * dt_max unless config pins dt."""
    dt_max = explicit_dt_max(grid, physics)
    pinned = float(config.get("time.dt", 0.0))
    if pinned > 0.0:
        return pinned, dt_max
    factor = cfl if cfl is not None else float(config.get("time.cfl_factor"))
    return factor * dt_max, dt_max


def simulate(config: Config, nx: int, cfl: float | None, case_name: str,
             t_final_override: float | None = None,
             n_steps_override: int | None = None) -> dict[str, Any]:
    """Integrate one configuration and return state, exact solution and metadata."""
    grid = build_grid(config, nx=nx, ny=nx)
    physics = physics_from_config(config)
    params = scheme_params_from_config(config)
    case, spec = build_case(case_name, grid, physics, config)

    t_final = spec.t_final if t_final_override is None else t_final_override
    dt_req, dt_max = _resolve_dt(config, grid, physics, cfl)
    if n_steps_override is not None:
        # Pinned step count: dt comes straight from the CFL and t_final follows.
        n_steps = n_steps_override
        dt = dt_req
        t_final = dt * n_steps
    else:
        n_steps = max(1, int(round(t_final / dt_req)))
        dt = t_final / n_steps                  # land exactly on t_final

    stepper = Stepper(grid, physics, params, dt)
    initial = case.initial()
    m0, e0 = total_mass(initial, grid), total_energy(initial, grid, physics)

    final, timing = timed_repeat(lambda: stepper.integrate(initial, n_steps),
                                 n_repeat=1, n_warmup=0)
    exact = case.exact(t_final)

    metrics = summarize(final, exact, grid, physics, m0, e0)
    if case_name == "igw":
        # Modal amplitude/phase diagnostics need a single (k, l); the broadband
        # case reports L2 and solver iteration counts instead.
        metrics.update(wave_errors(final, exact, grid, case.k, case.l))
    metrics.update({
        "nx": grid.nx, "ny": grid.ny, "dx": grid.dx,
        "dt": dt, "dt_max_explicit": dt_max, "cfl": dt / dt_max,
        "n_steps": n_steps, "t_final": t_final,
        "scheme": params.name, "theta": params.theta,
        "n_picard": params.n_picard, "solver": params.solver,
        "solver_iterations": stepper.solver_iterations,
        "solver_failures": stepper.solver_failures,
        "wall_s": timing.median_s, "timing_reportable": timing.reportable,
    })
    return {"grid": grid, "physics": physics, "params": params, "case": case,
            "spec": spec, "state": final, "exact": exact, "metrics": metrics}


def _order(errors: list[float], resolution: list[float]) -> list[float]:
    """Observed order between successive refinements.

    ``resolution`` is the quantity that doubles under refinement: nx for the
    spatial study, 1/dt for the temporal one. Using it directly (instead of an
    integer axis label) keeps the ratio exact for non-power-of-two sweeps.
    """
    orders: list[float] = [float("nan")]
    for prev, curr, r_prev, r_curr in zip(errors[:-1], errors[1:],
                                          resolution[:-1], resolution[1:]):
        if prev > 0.0 and curr > 0.0 and r_curr > 0.0 and r_prev > 0.0 \
                and r_curr != r_prev:
            orders.append(float(np.log(prev / curr) / np.log(r_curr / r_prev)))
        else:
            orders.append(float("nan"))
    return orders


# ---------------------------------------------------------------- subcommands
def cmd_run(config: Config, args: argparse.Namespace) -> int:
    case_name = args.case or config.get("case.name")
    result = simulate(config, args.nx or config.get("grid.nx"), args.cfl, case_name,
                      n_steps_override=args.steps)
    run_dir = _run_dir(config, f"run_{case_name}_{result['params'].name}")
    setup_logging(config.get("logging.level"), run_dir / config.get("logging.filename"))

    LOGGER.info(f"case={case_name} nx={result['metrics']['nx']} "
                f"scheme={result['params'].name} theta={result['params'].theta} "
                f"dt={result['metrics']['dt']:.2f}s "
                f"steps={result['metrics']['n_steps']}")
    LOGGER.info(f"L2(eta)={result['metrics']['l2_rel_eta']:.4e}  "
                f"mass drift={result['metrics']['mass_drift']:+.3e}  "
                f"energy drift={result['metrics']['energy_drift']:+.3e}")

    if config.get("netcdf.write_state"):
        write_state(run_dir / "state.nc", result["grid"], result["state"],
                    result["exact"], config.get("netcdf.format"),
                    int(config.get("netcdf.compression_level")))
    write_metrics(run_dir, result["metrics"])
    write_manifest(run_dir, config, {"mode": "run", "case": case_name})
    return 0


def cmd_verify(config: Config, args: argparse.Namespace) -> int:
    case_name = args.case or config.get("case.name")
    run_dir = _run_dir(config, f"verify_{args.study}_{case_name}")
    setup_logging(config.get("logging.level"), run_dir / config.get("logging.filename"))

    if args.study == "space":
        sizes = list(config.get("grid.sweep.nx_list"))
        rows = [simulate(config, nx, args.cfl, case_name)["metrics"] for nx in sizes]
        errors = [r["l2_rel_eta"] for r in rows]
        label = "nx"
        resolution = [float(r["nx"]) for r in rows]
        axis_values = [float(r["nx"]) for r in rows]
    else:
        nx = args.nx or int(config.get("grid.nx"))
        cfls = sorted(config.get("time.sweep.cfl_list"), reverse=True)
        # Reference: same grid, 8x smaller dt than the finest sweep point,
        # so the spatial error cancels and only the temporal order shows.
        LOGGER.info("building same-grid reference solution (dt/8 of finest sweep point)")
        reference = simulate(config, nx, min(cfls) / 8.0, case_name)["state"]
        rows, errors = [], []
        for cfl in cfls:
            res = simulate(config, nx, cfl, case_name)
            denom = float(np.sqrt(np.sum(reference.eta ** 2)))
            err = float(np.sqrt(np.sum((res["state"].eta - reference.eta) ** 2))) / denom
            res["metrics"]["l2_rel_eta_vs_reference"] = err
            rows.append(res["metrics"])
            errors.append(err)
        label = "1/cfl"
        resolution = [1.0 / r["dt"] for r in rows]
        axis_values = [1.0 / r["cfl"] for r in rows]

    orders = _order(errors, resolution)
    for row, err, order in zip(rows, errors, orders):
        row["observed_order"] = order

    header = (f"{label:>8} {'dx [km]':>10} {'dt [s]':>10} {'cfl':>7} "
              f"{'L2(eta)':>12} {'order':>7} {'amp_ratio':>10} {'phase[rad]':>11}")
    LOGGER.info(f"--- convergence study: {args.study}, case={case_name}, "
                f"scheme={rows[0]['scheme']}, theta={rows[0]['theta']}, "
                f"n_picard={rows[0]['n_picard']} ---")
    LOGGER.info(header)
    for row, err, order, axis in zip(rows, errors, orders, axis_values):
        LOGGER.info(
            f"{axis:>8.3g} "
            f"{row['dx'] / 1000:>10.2f} {row['dt']:>10.2f} {row['cfl']:>7.3f} "
            f"{err:>12.4e} {order:>7.2f} "
            f"{row.get('amp_ratio', float('nan')):>10.6f} "
            f"{row.get('phase_err_rad', float('nan')):>11.3e}")

    write_metrics(run_dir, {"study": args.study, "case": case_name, "rows": rows})
    write_manifest(run_dir, config, {"mode": f"verify:{args.study}", "case": case_name})
    return 0


def cmd_verify3d(config: Config, args: argparse.Namespace) -> int:
    """Convergence study for the 3D model (spec S7.6)."""
    case_name = args.case or "barotropic3d"
    run_dir = _run_dir(config, f"verify3d_{case_name}")
    setup_logging(config.get("logging.level"), run_dir / config.get("logging.filename"))

    sizes = [int(n) for n in (args.sweep or "16,32,64").split(",")]
    # --dim nz refines the vertical instead, which is what a boundary-layer
    # case like the Ekman spiral needs: its error lives in the vertical.
    if args.dim == "nz":
        pairs = [(args.nx or int(config.get("grid.nx")), n) for n in sizes]
    elif args.scale_nz:
        pairs = [(n, max(1, n // 2)) for n in sizes]
    else:
        pairs = [(n, args.nz) for n in sizes]
    rows, errors = [], []
    for n, (nx_i, nz_i) in zip(sizes, pairs):
        res = simulate3d(config, nx_i, nz_i, args.cfl, case_name,
                         n_steps_override=(args.steps or 4 * n))
        m = res["metrics"]
        # Each case puts its error in a different field: the wave cases in
        # buoyancy or velocity, the advection case in the tracer it carries.
        field = {"baroclinic_igw": "l2_rel_b",
                 "tracer_advect": "l2_rel_T"}.get(case_name, "l2_rel_u")
        rows.append(m)
        errors.append(m[field])

    orders = _order(errors, [float(n) for n in sizes])
    LOGGER.info(f"--- 3D convergence: case={case_name}, scheme={rows[0]['scheme']}, "
                f"theta={rows[0]['theta']}, theta_v={rows[0]['theta_v']} ---")
    LOGGER.info(f"{'nx':>6}{'nz':>5}{'cells':>10}{'dt [s]':>10}"
                f"{'L2':>13}{'order':>7}{'pcg/solve':>11}")
    for m, e, o in zip(rows, errors, orders):
        LOGGER.info(f"{m['nx']:>6}{m['nz']:>5}{m['cells']:>10}{m['dt']:>10.2f}"
                    f"{e:>13.4e}{o:>7.2f}{m['pcg_per_solve']:>11.1f}")

    write_metrics(run_dir, {"study": "3d", "case": case_name, "rows": rows})
    write_manifest(run_dir, config, {"mode": "verify3d", "case": case_name})
    return 0


def cmd_bench3d(config: Config, args: argparse.Namespace) -> int:
    """Per-cell cost of the 3D model across grid sizes."""
    case_name = args.case or "baroclinic_igw"
    run_dir = _run_dir(config, f"bench3d_{case_name}")
    setup_logging(config.get("logging.level"), run_dir / config.get("logging.filename"))
    if not is_bench_node():
        LOGGER.warning("NOT on a designated bench node - timings are DEV-ONLY (R7-1)")

    sizes = [int(n) for n in (args.sweep or "64,100,128").split(",")]
    rows = []
    LOGGER.info(f"{'nx':>6}{'nz':>5}{'cells':>10}{'steps':>7}{'wall [s]':>11}"
                f"{'ms/step':>10}{'ns/cell/step':>14}{'pcg/solve':>11}")
    for n in sizes:
        m = simulate3d(config, n, args.nz, args.cfl, case_name,
                       n_steps_override=(args.steps or 20),
                       n_repeat=args.repeat)["metrics"]
        rows.append(m)
        LOGGER.info(f"{m['nx']:>6}{m['nz']:>5}{m['cells']:>10}{m['n_steps']:>7}"
                    f"{m['wall_s']:>11.4f}{m['us_per_step'] / 1000:>10.2f}"
                    f"{m['ns_per_cell_step']:>14.2f}{m['pcg_per_solve']:>11.1f}")

    write_metrics(run_dir, {"mode": "bench3d", "case": case_name, "rows": rows})
    write_manifest(run_dir, config, {"mode": "bench3d", "case": case_name})
    return 0


def cmd_bench(config: Config, args: argparse.Namespace) -> int:
    """Error-vs-wallclock frontier over the dt sweep (RULES.md R8)."""
    case_name = args.case or config.get("case.name")
    run_dir = _run_dir(config, f"bench_{case_name}")
    setup_logging(config.get("logging.level"), run_dir / config.get("logging.filename"))
    if not is_bench_node():
        LOGGER.warning("NOT on the designated bench node - these timings are "
                       "DEV-ONLY and must not enter a report (RULES.md R7-1)")

    nx = args.nx or int(config.get("grid.nx"))
    rows: list[dict[str, Any]] = []
    LOGGER.info(f"{'cfl':>7} {'dt [s]':>10} {'steps':>7} {'L2(eta)':>12} "
                f"{'wall [s]':>10} {'us/step':>9}")
    for cfl in config.get("time.sweep.cfl_list"):
        metrics = simulate(config, nx, cfl, case_name,
                           n_steps_override=args.steps)["metrics"]
        metrics["us_per_step"] = 1e6 * metrics["wall_s"] / metrics["n_steps"]
        rows.append(metrics)
        LOGGER.info(f"{cfl:>7.3f} {metrics['dt']:>10.2f} {metrics['n_steps']:>7d} "
                    f"{metrics['l2_rel_eta']:>12.4e} {metrics['wall_s']:>10.4f} "
                    f"{metrics['us_per_step']:>9.1f}")

    write_metrics(run_dir, {"mode": "bench", "case": case_name, "nx": nx, "rows": rows})
    write_manifest(run_dir, config, {"mode": "bench", "case": case_name,
                                     "timings_reportable": is_bench_node()})
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfd_exp", description="Ocean-model numerics experiment driver")
    parser.add_argument("--config", default="config", type=Path,
                        help="configuration directory (default: config)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a dotted config key, e.g. --set scheme.theta=0.6")
    sub = parser.add_subparsers(dest="command", required=True)

    common = dict(case=dict(help="verification case (default: config case.name)"),
                  nx=dict(type=int, help="horizontal resolution override"),
                  cfl=dict(type=float, help="dt as a fraction of the explicit limit"))

    p_run = sub.add_parser("run", help="single simulation")
    p_verify = sub.add_parser("verify", help="convergence study")
    p_verify.add_argument("--study", choices=("space", "time"), default="space")
    p_bench = sub.add_parser("bench", help="error-vs-wallclock frontier")
    p_v3d = sub.add_parser("verify3d", help="3D convergence study (spec v0.2)")
    p_v3d.add_argument("--sweep", default=None, help="comma-separated nx list")
    p_v3d.add_argument("--scale-nz", action="store_true",
                       help="scale nz with nx (nz = nx/2) for a full 3D refinement")
    p_v3d.add_argument("--dim", choices=("nx", "nz"), default="nx",
                       help="which dimension the sweep refines (default nx)")
    p_b3d = sub.add_parser("bench3d", help="3D cost across grid sizes")
    p_b3d.add_argument("--sweep", default=None, help="comma-separated nx list")
    p_b3d.add_argument("--repeat", type=int, default=1)

    for sp in (p_v3d, p_b3d):
        sp.add_argument("--nz", type=int, default=30)

    for sp in (p_run, p_verify, p_bench, p_v3d, p_b3d):
        sp.add_argument("--case", **common["case"])
        sp.add_argument("--nx", **common["nx"])
        sp.add_argument("--cfl", **common["cfl"])
        sp.add_argument("--steps", type=int, default=None,
                        help="pin the step count (per-step cost benchmarks); "
                             "without it the run integrates to the case's t_final")
    return parser


def _apply_overrides(config: Config, overrides: list[str]) -> Config:
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            value: Any = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = {"true": True, "false": False}.get(raw.lower(), raw)
        parsed[key] = value
    return config.with_overrides(parsed) if parsed else config


def main() -> int:
    args = build_parser().parse_args()
    config = _apply_overrides(load_config(args.config), args.set)
    setup_logging(config.get("logging.level"))

    handlers = {"run": cmd_run, "verify": cmd_verify, "bench": cmd_bench,
                "verify3d": cmd_verify3d, "bench3d": cmd_bench3d}
    return handlers[args.command](config, args)


if __name__ == "__main__":
    raise SystemExit(main())
