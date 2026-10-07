#########################################################################
#  Module: manifest                                                     #
#  Description: Reproducibility record for every run (RULES.md R6):    #
#               git provenance, full config + hash, host and toolchain  #
#               identification, timing block.                           #
#  Pipeline: driver -> manifest -> output/<run_id>/manifest.json        #
#########################################################################

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from libs.utils.config import Config

LOGGER = logging.getLogger(__name__)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), capture_output=True, text=True,
                             timeout=10, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug(f"git {' '.join(args)} failed: {exc}")
        return None


def git_provenance() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def host_info() -> dict[str, Any]:
    return {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def write_manifest(run_dir: Path, config: Config, extra: dict[str, Any]) -> Path:
    """Write manifest.json. A run without one is invalid (RULES.md R6)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec_version": config.get("project.spec_version"),
        "git": git_provenance(),
        "host": host_info(),
        "config_sha256": config.sha256(),
        "config": config.as_dict(),
        **extra,
    }
    path = run_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False, default=str)
    LOGGER.info(f"manifest written: {path}")
    return path


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    LOGGER.info(f"metrics written: {path}")
    return path
