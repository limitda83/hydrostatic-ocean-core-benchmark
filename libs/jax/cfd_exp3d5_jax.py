#!/usr/bin/env python3
#########################################################################
#  Module: cfd_exp3d5_jax                                               #
#  Description: Driver of the JAX v0.5 backend. Reads the namelist and  #
#               binaries written by toml2nml --v05 (the same files the   #
#               Fortran/OpenACC/CUDA drivers read), integrates n_steps   #
#               with n_warmup discarded runs (the first carries the JIT  #
#               compile, reported separately per R7-5) and n_repeat      #
#               timed runs (median + MAD), then writes                   #
#               <prefix>_state3d5.bin and <prefix>_metrics.json in the   #
#               layout of cfd_exp3d5.f90 so gate3d5 / tier2_sweep treat  #
#               it as one more binary.                                   #
#  Pipeline: toml2nml --v05 -> cfd_exp3d5_jax -> gate3d5 / tier2_sweep  #
#########################################################################

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read_namelist(path: Path) -> dict:
    """Flat key -> value map of every group (config.hpp semantics)."""
    kv: dict = {}
    for line in path.read_text().splitlines():
        line = line.split("!", 1)[0].strip()
        if not line or line[0] in "&/":
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if val.startswith("'"):
            kv[key.strip()] = val.strip("'")
        elif val in (".true.", ".false."):
            kv[key.strip()] = val == ".true."
        else:
            kv[key.strip()] = float(re.sub(r"[dD]", "e", val))
    return kv


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: cfd_exp3d5_jax.py <namelist>", file=sys.stderr)
        return 1
    cfg = read_namelist(Path(sys.argv[1]))
    import jax
    import jax.numpy as jnp
    from libs.jax.model3d_v05_jax import StepperJAX

    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    n2, n3 = nx * ny, nx * ny * nz
    ts = cfg["tracers"] == "TS"
    dom = np.fromfile(cfg["domain_file"], dtype=np.float64)
    h, mask = dom[:n2].reshape(ny, nx), dom[n2:2 * n2].reshape(ny, nx)
    raw = np.fromfile(cfg["init_file"], dtype=np.float64)
    eta0 = raw[:n2].reshape(ny, nx)
    fields = [raw[n2 + i * n3:n2 + (i + 1) * n3].reshape(nz, ny, nx) for i in range(5 if ts else 3)]
    u0, v0, b0 = fields[:3]
    t0, s0 = (fields[3], fields[4]) if ts else (np.zeros_like(u0), np.zeros_like(u0))

    dev = jax.devices()[0]
    st = StepperJAX(cfg, h, mask)
    n_steps, n_rep, n_warm = int(cfg["n_steps"]), int(cfg["n_repeat"]), int(cfg["n_warmup"])
    print(f"jax3d5: case={cfg['case_name']} nx={nx} nz={nz} scheme={cfg['scheme_name']} "
          f"solver={cfg['solver_kind']} eos={cfg['eos']} dt={cfg['dt']:.5E} steps={n_steps} "
          f"device={dev.device_kind} x64={jax.config.jax_enable_x64}")

    e_min = float(cfg.get("e_min", 1.0e-6))

    def reset():
        base = tuple(jnp.asarray(a) for a in (u0, v0, b0, eta0, t0, s0)) + (jnp.int64(0), jnp.int64(0))
        # v0.6 closure state: TKE and the interface eddy coefficients (constant when off)
        return base + (jnp.full((nz + 1, ny, nx), e_min), jnp.full((nz + 1, ny, nx), float(cfg["nu"])),
                       jnp.full((nz + 1, ny, nx), float(cfg["kappa"])))

    diverged = [False]

    def run(state):
        """Time ONLY the step loop. `reset()` allocates and copies nine device
        arrays (1.4 GB of H2D at 1000^2 x 30); CUDA and Fortran both do that
        outside their timer, so leaving it inside made every JAX number in the
        ladder carry an initialisation cost the others did not (docs/90 N33)."""
        for i in range(n_steps):
            state = st.step(state)
            if i % 10 == 9 and not bool(jnp.max(jnp.abs(state[3])) < 1.0e6):   # NaN or > 1e6 m
                diverged[0] = True
                break
        jax.block_until_ready(state)
        return state

    compile_s = None
    reset_s = []
    for i in range(max(1, n_warm)):
        tic = time.perf_counter()
        state = jax.block_until_ready(reset())
        state = run(state)
        if i == 0:
            compile_s = time.perf_counter() - tic
    samples = []
    for _ in range(n_rep):
        t_reset = time.perf_counter()
        state = jax.block_until_ready(reset())      # R7-5: initialisation is a
        tic = time.perf_counter()                   # separate column, not part
        reset_s.append(tic - t_reset)               # of the kernel time
        state = run(state)
        samples.append(time.perf_counter() - tic)
        if diverged[0]:
            print("  DIVERGED (|eta| > 1e6 or NaN) - run stopped early")
            break
    t_reset_med = float(np.median(reset_s)) if reset_s else float("nan")
    t_med = float(np.median(samples))
    t_mad = float(np.median(np.abs(np.array(samples) - t_med)))
    u, v, b, eta, t, s, iters, fails = (np.asarray(a) for a in state[:8])
    print(f"  wall median={t_med:12.5E} s  mad={t_mad:12.5E} s  compile+first={compile_s:.3f} s  "
          f"reset={t_reset_med:.4f} s  solver_iters/run={int(iters)}")

    prefix = cfg["out_prefix"]
    with open(f"{prefix}_state3d5.bin", "wb") as f:
        for a in ([eta, u, v, b] + ([t, s] if ts else [])):
            np.ascontiguousarray(a, dtype=np.float64).tofile(f)
    metrics = {
        "backend": "jax3d5", "device": dev.device_kind, "jax_version": jax.__version__,
        "nx": nx, "ny": ny, "nz": nz, "cells": n3, "n_steps": n_steps,
        "scheme": cfg["scheme_name"], "solver": cfg["solver_kind"], "eos": cfg["eos"],
        "advection": cfg["advection"], "tracers": cfg["tracers"], "theta": cfg["theta"],
        "dt": cfg["dt"], "solver_iterations": int(iters), "solver_failures": int(fails),
        "n_split": st.n_split, "barotropic_substeps": st.n_split * n_steps if st.is_split else 0,
        "wall_s": t_med, "wall_mad_s": t_mad, "n_repeat": n_rep, "diverged": diverged[0],
        "compile_and_first_run_s": compile_s, "reset_s": t_reset_med,
        "omp_num_threads": 0,
    }
    Path(f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
