"""Load the benchmark configuration and resolve project paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "benchmark.yaml"

PATHS = {
    "private": ROOT / "data" / "private",          # reference labels, case map, annotated plots
    "model_inputs": ROOT / "data" / "model_inputs",  # clean images only
    "results": ROOT / "results",
    "docs": ROOT / "docs",
}


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable hash of the frozen protocol settings.

    Excluded: budget (a cap change is not a protocol change) and the model roster and
    prices, so models added later are evaluated under the same frozen protocol. Each
    attempt record logs its exact model settings.
    """
    clean = {k: v for k, v in cfg.items()
             if not k.startswith("_") and k not in ("budget", "models", "pricing_usd_per_mtok")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:16]
