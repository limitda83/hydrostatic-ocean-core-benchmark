#!/usr/bin/env python3
#########################################################################
#  Module: gpu_microbench                                               #
#  Description: Measure the roofline constants this project's           #
#               conclusions depend on - achievable memory bandwidth and #
#               the FP64:FP32 compute ratio - so that RQ4 and threat T2 #
#               rest on measurement rather than on vendor claims.       #
#  Pipeline: standalone -> docs/11 roofline constants                   #
#########################################################################

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Callable

try:
    import cupy as cp
except ImportError as exc:                       # pragma: no cover
    raise SystemExit(f"cupy is required for the GPU microbenchmark: {exc}")


def timed(op: Callable[[], None], n_iter: int, n_warmup: int = 5) -> float:
    """Seconds per iteration, timed with CUDA events after a warm-up."""
    for _ in range(n_warmup):
        op()
    cp.cuda.Stream.null.synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(n_iter):
        op()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / n_iter / 1e3


def bandwidth(size_bytes: int, dtype, n_iter: int) -> dict[str, float]:
    """STREAM-style copy and add. Reports achieved bandwidth, not peak."""
    n = size_bytes // cp.dtype(dtype).itemsize
    a = cp.ones(n, dtype=dtype)
    b = cp.ones(n, dtype=dtype)
    out = cp.empty_like(a)
    nbytes = a.nbytes

    t_copy = timed(lambda: cp.copyto(out, a), n_iter)
    t_add = timed(lambda: cp.add(a, b, out=out), n_iter)
    return {
        "copy_s": t_copy,
        "copy_gbs": 2 * nbytes / t_copy / 1e9,      # 1 read + 1 write
        "add_s": t_add,
        "add_gbs": 3 * nbytes / t_add / 1e9,        # 2 reads + 1 write
    }


def compute_ratio(n: int, n_fma: int, dtype, n_iter: int) -> dict[str, float]:
    """Compute-bound FMA chain: isolates arithmetic throughput from bandwidth."""
    x = cp.ones(n, dtype=dtype)
    name = cp.dtype(dtype).name
    kernel = cp.ElementwiseKernel(
        "T x", "T z",
        f"T t = x; for (int i = 0; i < {n_fma}; ++i) {{ t = t * t + x; }} z = t;",
        f"fma_chain_{name}",
    )
    t = timed(lambda: kernel(x), n_iter)
    return {"s": t, "tflops": 2.0 * n_fma * n / t / 1e12}


def main() -> int:
    parser = argparse.ArgumentParser(description="RTX-class GPU roofline constants")
    parser.add_argument("--size-mib", type=int, default=1024,
                        help="array size per operand in MiB (default 1024)")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("output/gpu_microbench.json"))
    args = parser.parse_args()

    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    result: dict[str, Any] = {
        "host": platform.node(),
        "gpu": props["name"].decode(),
        "compute_capability": device.compute_capability,
        "cupy": cp.__version__,
        "size_mib": args.size_mib,
        "bandwidth": {},
        "compute": {},
    }

    size_bytes = args.size_mib << 20
    print(f"{result['gpu']} (cc {result['compute_capability']}), "
          f"cupy {cp.__version__}, {args.size_mib} MiB operands\n")
    print(f"{'kernel':<10}{'dtype':>10}{'ms':>10}{'GB/s':>10}")
    for dtype in (cp.float32, cp.float64):
        name = cp.dtype(dtype).name
        res = bandwidth(size_bytes, dtype, args.n_iter)
        result["bandwidth"][name] = res
        print(f"{'copy':<10}{name:>10}{res['copy_s'] * 1e3:>10.2f}{res['copy_gbs']:>10.0f}")
        print(f"{'add':<10}{name:>10}{res['add_s'] * 1e3:>10.2f}{res['add_gbs']:>10.0f}")

    print(f"\n{'kernel':<10}{'dtype':>10}{'ms':>10}{'TFLOP/s':>12}")
    for dtype in (cp.float32, cp.float64):
        name = cp.dtype(dtype).name
        res = compute_ratio(1 << 22, 256, dtype, args.n_iter)
        result["compute"][name] = res
        print(f"{'fma chain':<10}{name:>10}{res['s'] * 1e3:>10.2f}{res['tflops']:>12.2f}")

    f32 = result["compute"]["float32"]["tflops"]
    f64 = result["compute"]["float64"]["tflops"]
    result["fp32_over_fp64"] = f32 / f64 if f64 > 0 else float("nan")
    bw32 = result["bandwidth"]["float32"]["add_gbs"]
    bw64 = result["bandwidth"]["float64"]["add_gbs"]
    result["bw_fp32_over_fp64"] = bw32 / bw64 if bw64 > 0 else float("nan")

    print(f"\nFP32:FP64 compute ratio   = {result['fp32_over_fp64']:.1f} : 1")
    print(f"FP32:FP64 bandwidth ratio = {result['bw_fp32_over_fp64']:.2f} : 1")
    print("\nInterpretation: a bandwidth-bound ocean kernel pays the BANDWIDTH "
          "ratio, not the compute ratio, when moving fp64 -> fp32.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
