#########################################################################
#  Module: config                                                       #
#  Description: Load and merge the TOML configuration tree, expose it   #
#               through dotted-key lookup, and hash it for the run      #
#               manifest.                                               #
#  Pipeline: entry point -> config -> grid / schemes / cases / io       #
#########################################################################

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:                                   # Python >= 3.11
    import tomllib
except ModuleNotFoundError:            # Python 3.9/3.10 on the RHEL cluster nodes
    import tomli as tomllib            # type: ignore[no-redef]

# Files merged, in order, into a single configuration mapping.
CONFIG_FILES = ("settings.toml", "grid.toml", "schemes.toml", "cases.toml", "backends.toml")


class ConfigError(KeyError):
    """Raised when a required configuration key is missing or invalid."""


class Config:
    """Immutable view over the merged TOML configuration."""

    def __init__(self, data: dict[str, Any], source_dir: Path) -> None:
        self._data = data
        self.source_dir = source_dir

    # ---------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = ...) -> Any:
        """Look up ``a.b.c``. Raises ConfigError when missing and no default."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise ConfigError(f"missing config key '{dotted}' (at '{part}')")
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> dict[str, Any]:
        node = self.get(dotted)
        if not isinstance(node, dict):
            raise ConfigError(f"config key '{dotted}' is not a table")
        return dict(node)

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))  # deep copy via round-trip

    # ------------------------------------------------------------ provenance
    def sha256(self) -> str:
        """Stable hash of the whole configuration, recorded in the manifest."""
        blob = json.dumps(self._data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------- override
    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides applied (axis sweeps use this)."""
        data = self.as_dict()
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data, self.source_dir)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_dir: Path | str = "config") -> Config:
    """Read and merge every TOML file in CONFIG_FILES from ``config_dir``."""
    directory = Path(config_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"config directory not found: {directory}")

    merged: dict[str, Any] = {}
    for name in CONFIG_FILES:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"required config file missing: {path}")
        with path.open("rb") as handle:
            _deep_merge(merged, tomllib.load(handle))
    return Config(merged, directory)
