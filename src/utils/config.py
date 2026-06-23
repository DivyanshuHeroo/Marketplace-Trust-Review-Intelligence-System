"""Configuration + path helpers shared across the OlistTrust project.

Everything resolves relative to the project root (the folder containing config.yaml),
so scripts work regardless of the current working directory.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache config.yaml."""
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def db_path() -> Path:
    """Absolute path to the SQLite database file."""
    return resolve_path(load_config()["paths"]["database"])


def raw_dir() -> Path:
    return resolve_path(load_config()["paths"]["raw_dir"])


def processed_dir() -> Path:
    d = resolve_path(load_config()["paths"]["processed_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = resolve_path(load_config()["paths"]["models_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir() -> Path:
    d = resolve_path(load_config()["paths"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d
