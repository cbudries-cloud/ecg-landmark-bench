"""Conventional raw-signal baseline: WFDB XQRS on the numerical Lead II signal.

This method sees the digital waveform, not an image, so it is reported
separately and never described as an equal-input comparison.
"""
from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
from wfdb import processing

from .data import LeadSignal


def xqrs_peaks_s(sig: LeadSignal, cfg: dict, corrected: bool = False) -> np.ndarray:
    """XQRS detections in seconds. `corrected` snaps each detection to the local
    extremum with wfdb.processing.correct_peaks (a documented variant)."""
    logging.getLogger("wfdb").setLevel(logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()):
        xq = processing.XQRS(sig=sig.signal, fs=sig.fs)
        xq.detect(verbose=False)
    idx = np.asarray(xq.qrs_inds, dtype=int)
    if corrected and len(idx):
        b = cfg["baseline"]
        idx = processing.correct_peaks(
            sig.signal, idx,
            search_radius=int(b["correct_search_radius_s"] * sig.fs),
            smooth_window_size=int(b["correct_smooth_window_s"] * sig.fs),
            peak_dir="compare",
        )
        idx = idx[idx >= 0]
    return np.sort(idx / sig.fs)
