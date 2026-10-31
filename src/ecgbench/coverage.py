"""Annotation coverage and eligibility rules (brief Step 5).

Rule set v1 (fixed before any model or baseline result was computed):
  E1 coverage: the first and last Lead II annotation (any symbol) must bracket
     [window_start - tol, window_end + tol], so every reference a scored
     prediction could match is known.
  E2 completeness: a labeled RR gap > 1.6 x the record's median RR that overlaps
     [window_start - tol, window_end + tol] needs a manual verdict in
     config/coverage_review.yaml; missing verdict or "incomplete" -> excluded.
  E3 at least one reference peak inside the scoring window.
Exclusions keep the record in its original partition and are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from .config import ROOT
from .data import annotation_span_s, reference_peaks_s

GAP_FACTOR = 1.6
REVIEW_FILE = ROOT / "config" / "coverage_review.yaml"


def eligibility_table(cfg: dict, records: list[int]) -> pd.DataFrame:
    a, b = cfg["task"]["scoring_window_s"]
    tol = cfg["task"]["tolerance_s"]
    lo, hi = a - tol, b + tol
    review = (yaml.safe_load(REVIEW_FILE.read_text()) or {}).get("records", {}) if REVIEW_FILE.exists() else {}
    rows = []
    for r in records:
        first, last = annotation_span_s(cfg, r)
        refs = reference_peaks_s(cfg, r)
        n_win = int(((refs >= a) & (refs <= b)).sum())
        reasons = []
        if not (first <= lo and last >= hi):
            reasons.append(f"E1_coverage: annotations span {first:.3f}-{last:.3f} s")
        rr = np.diff(refs)
        flagged = []
        if len(rr) >= 2:
            med = np.median(rr)
            for k in np.where(rr > GAP_FACTOR * med)[0]:
                if refs[k] < hi and refs[k + 1] > lo:
                    flagged.append(f"{refs[k]:.2f}-{refs[k + 1]:.2f}")
        if flagged:
            verdict = review.get(r, {}).get("verdict")
            if verdict != "complete":
                reasons.append(f"E2_gap_unreviewed_or_incomplete: {','.join(flagged)} ({verdict})")
        if n_win == 0:
            reasons.append("E3_no_reference_in_window")
        rows.append({
            "record": r, "ann_first_s": round(first, 3), "ann_last_s": round(last, 3),
            "n_refs_total": len(refs), "n_refs_in_window": n_win,
            "gap_flags": ";".join(flagged), "eligible": not reasons, "exclusion_reason": "; ".join(reasons),
        })
    return pd.DataFrame(rows)
