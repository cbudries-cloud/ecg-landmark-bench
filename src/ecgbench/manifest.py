"""File manifest with SHA-256 checksums, cross-checked against PhysioNet's SHA256SUMS.txt."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import ROOT


def build_manifest(cfg: dict) -> pd.DataFrame:
    raw = ROOT / cfg["dataset"]["raw_dir"]
    official = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        official[name.strip()] = digest
    rows = []
    for name, expected in sorted(official.items()):
        p = raw / name
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        rows.append({"file": name, "bytes": p.stat().st_size if p.exists() else None,
                     "sha256": actual, "matches_official": actual == expected})
    return pd.DataFrame(rows)
