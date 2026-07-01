"""
utils.py
========
Small, reusable helper functions used across the project: logging,
random-seed control, safe percentage parsing, and JSON/CSV IO helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src import config


def get_logger(name: str = "BioVentureSim") -> logging.Logger:
    """Return a configured logger (idempotent: avoids duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = config.RANDOM_SEED) -> np.random.Generator:
    """
    Seed global NumPy state AND return a modern Generator.

    We seed the legacy global state so that any third-party library relying
    on np.random is reproducible, and we return a Generator for our own
    sampling (preferred modern API).
    """
    np.random.seed(seed)
    return np.random.default_rng(seed)


def as_fraction(value: float) -> float:
    """
    Normalise a possibly-percentage value into a 0-1 fraction.

    Accepts 27 (meaning 27%) or 0.27 and returns 0.27 in both cases. Values
    already <= 1 are assumed to be fractions and returned unchanged.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v / 100.0 if v > 1.0 else v


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Constrain a value to the [low, high] interval."""
    return max(low, min(high, value))


def save_json(obj: Any, path: Path) -> None:
    """Write an object to disk as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Read a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def human_money(value: float) -> str:
    """Format a USD amount compactly, e.g. 1.8e9 -> '$1.8B'."""
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"

