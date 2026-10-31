"""Read LUDB records and expert QRS reference peaks with WFDB."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from .config import ROOT


@dataclass
class LeadSignal:
    record: int
    lead: str
    fs: float
    units: str
    signal: np.ndarray       # physical units (mV), untransformed

    @property
    def times_s(self) -> np.ndarray:
        return np.arange(len(self.signal)) / self.fs


def record_path(cfg: dict, record: int) -> str:
    return str(ROOT / cfg["dataset"]["raw_dir"] / "data" / str(record))


def load_lead(cfg: dict, record: int) -> LeadSignal:
    lead = cfg["task"]["lead"]
    rec = wfdb.rdrecord(record_path(cfg, record), channel_names=[lead], physical=True)
    if rec.sig_name != [lead]:
        raise ValueError(f"record {record}: lead {lead!r} not found (got {rec.sig_name})")
    if rec.fs != cfg["dataset"]["expected_fs"] or rec.sig_len != cfg["dataset"]["expected_samples"]:
        raise ValueError(f"record {record}: unexpected fs/length {rec.fs}/{rec.sig_len}")
    return LeadSignal(record, lead, float(rec.fs), rec.units[0], rec.p_signal[:, 0].astype(float))


def load_annotations(cfg: dict, record: int) -> wfdb.Annotation:
    # LUDB stores one annotation file per lead, with the lead name as the extension.
    return wfdb.rdann(record_path(cfg, record), cfg["task"]["lead"])


def reference_peaks_s(cfg: dict, record: int) -> np.ndarray:
    """Expert QRS reference peaks in seconds from the start of the original recording."""
    ann = load_annotations(cfg, record)
    sym = np.asarray(ann.symbol)
    samples = ann.sample[sym == cfg["task"]["qrs_peak_symbol"]]
    return np.sort(samples / float(ann.fs or cfg["dataset"]["expected_fs"]))


def annotation_span_s(cfg: dict, record: int) -> tuple[float, float]:
    """First and last annotation of any kind (wave boundary or peak) on the lead, in seconds."""
    ann = load_annotations(cfg, record)
    fs = float(ann.fs or cfg["dataset"]["expected_fs"])
    return ann.sample.min() / fs, ann.sample.max() / fs


def case_id(cfg: dict, record: int) -> str:
    c = cfg["case_ids"]
    digest = hmac.new(c["salt"].encode(), str(record).encode(), hashlib.sha256).hexdigest()
    return "c_" + digest[: c["length"]]


def all_records(cfg: dict) -> list[int]:
    path = ROOT / cfg["dataset"]["raw_dir"] / "RECORDS"
    return [int(Path(line.strip()).name) for line in path.read_text().splitlines() if line.strip()]
