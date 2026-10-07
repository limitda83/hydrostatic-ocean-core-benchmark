#########################################################################
#  Module: timing                                                       #
#  Description: Timing protocol enforced by RULES.md R7 - warm-up      #
#               discarded, >=5 repeats, median + MAD, monotonic clock.  #
#  Pipeline: schemes / driver -> timing -> metrics.json                 #
#########################################################################

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

# Reported timings are only valid on this host (RULES.md R7-1).
BENCH_HOST_MARKERS = ("localhost.localdomain",)


@dataclass(frozen=True)
class TimingStats:
    """Median and median-absolute-deviation over repeated measurements."""

    median_s: float
    mad_s: float
    min_s: float
    max_s: float
    n_repeat: int
    n_warmup: int
    samples_s: list[float]
    reportable: bool          # False when measured off the designated bench node
    host: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def is_bench_node() -> bool:
    """True only on the node whose timings may enter a report (R7-1)."""
    node = platform.node()
    return any(marker in node for marker in BENCH_HOST_MARKERS)


def timed_repeat(
    func: Callable[[], Any],
    n_repeat: int = 5,
    n_warmup: int = 1,
    sync: Callable[[], None] | None = None,
) -> tuple[Any, TimingStats]:
    """Run ``func`` n_warmup+n_repeat times; return its last result and stats.

    ``sync`` is the device-synchronisation hook required before stopping the
    clock on GPU backends (R7-4). CPU backends pass None.
    """
    if n_repeat < 5:
        LOGGER.warning(f"n_repeat={n_repeat} violates RULES.md R7-2 (minimum 5)")

    for _ in range(n_warmup):
        func()
        if sync is not None:
            sync()

    samples: list[float] = []
    result: Any = None
    for _ in range(n_repeat):
        start = time.perf_counter()
        result = func()
        if sync is not None:
            sync()
        samples.append(time.perf_counter() - start)

    med = _median(samples)
    stats = TimingStats(
        median_s=med,
        mad_s=_median([abs(s - med) for s in samples]),
        min_s=min(samples),
        max_s=max(samples),
        n_repeat=n_repeat,
        n_warmup=n_warmup,
        samples_s=samples,
        reportable=is_bench_node(),
        host=platform.node(),
    )
    if not stats.reportable:
        LOGGER.warning(
            f"timings measured on '{stats.host}' are DEV-ONLY and must not be "
            f"reported (RULES.md R7-1; bench node = gpgpu)"
        )
    return result, stats
