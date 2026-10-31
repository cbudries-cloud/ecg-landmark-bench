"""Reproducible development / held-out partition at the subject level.

LUDB has exactly one recording per subject (200 records, 200 subjects), so a
record-level split is also a subject-level split. Any later derived asset
(extra leads, crops) inherits its source record's partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PATHS
from .data import all_records

SPLIT_FILE = PATHS["results"].parent / "config" / "split.csv"


def make_split(cfg: dict) -> pd.DataFrame:
    s = cfg["split"]
    records = sorted(all_records(cfg))
    if len(records) != s["n_dev"] + s["n_test"]:
        raise ValueError(f"{len(records)} records but split asks for {s['n_dev']}+{s['n_test']}")
    rng = np.random.default_rng(s["seed"])
    order = rng.permutation(records)
    dev, test = sorted(order[: s["n_dev"]]), sorted(order[s["n_dev"]:])
    pilot = set(rng.choice(dev, size=s["n_pilot"], replace=False).tolist())
    rows = [(int(r), "dev", int(r) in pilot) for r in dev] + [(int(r), "test", False) for r in test]
    df = pd.DataFrame(rows, columns=["record", "partition", "pilot"]).sort_values("record")
    df["subject"] = df["record"]  # one subject per record in LUDB 1.0.1
    return df.reset_index(drop=True)


def load_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_FILE)
