"""Final-report statistics: case-level bootstrap CIs, paired zoom effects, subgroups.

The unit of resampling is the case (one record = one subject). Beats and attempts
are never resampled on their own. Subgroups come from the private record headers
and are used for error analysis only; they never reach a model.
"""
from __future__ import annotations

import re

import numpy as np
import wfdb

from .data import record_path
from .scoring import score_case

SUBGROUP_RULES = {
    # Header diagnosis line -> subgroup. "Wandering atrial pacemaker" and "PAC" are not pacing.
    "paced": re.compile(r"pacing\.", re.I),
    "atrial_fibrillation": re.compile(r"^Rhythm: Atrial fibrillation", re.I),
}


def subgroups(cfg: dict, record: int) -> list[str]:
    lines = [c.strip() for c in wfdb.rdheader(record_path(cfg, record)).comments]
    return [name for name, rx in SUBGROUP_RULES.items() if any(rx.search(line) for line in lines)]


def case_table(cases: list[tuple[list[float], list[float], bool]], window_s, tol_s) -> list[dict]:
    """Per-case counts and absolute errors (ms) at one tolerance."""
    rows = []
    for preds, refs, valid in cases:
        s = score_case(preds, refs, window_s, tol_s, valid=valid)
        rows.append({"tp": s.tp, "fp": s.fp, "fn": s.fn, "passed": s.passed,
                     "abs_err_ms": np.abs(np.asarray(s.errors_s)) * 1e3})
    return rows


def metrics(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    errs = np.concatenate([r["abs_err_ms"] for r in rows]) if rows else np.array([])
    return {"f1": f1, "case_pass_rate": float(np.mean([r["passed"] for r in rows])) if rows else float("nan"),
            "p95_abs_err_ms": float(np.percentile(errs, 95)) if len(errs) else float("nan")}


def bootstrap(arms: dict[str, list[dict]], n_boot: int, seed: int, pairs: list[tuple[str, str]] = ()) -> dict:
    """Point estimate and 95% percentile CI per arm, and for paired differences b - a.

    All arms must list the same cases in the same order, so every replicate draws
    the same cases for every arm (a paired design).
    """
    n = {len(rows) for rows in arms.values()}
    assert len(n) == 1, "arms must cover the same cases"
    n = n.pop()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = {a: [metrics([rows[i] for i in rep]) for rep in idx] for a, rows in arms.items()}

    def summarize(point: dict, reps: list[dict]) -> dict:
        out = {}
        for k, v in point.items():
            vals = np.array([r[k] for r in reps], float)
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            out |= {k: v, f"{k}_lo": float(lo), f"{k}_hi": float(hi)}
        return out

    result = {a: summarize(metrics(rows), draws[a]) for a, rows in arms.items()}
    for a, b in pairs:
        pa, pb = metrics(arms[a]), metrics(arms[b])
        diff = {k: pb[k] - pa[k] for k in pa}
        reps = [{k: rb[k] - ra[k] for k in ra} for ra, rb in zip(draws[a], draws[b])]
        result[f"{b} minus {a}"] = summarize(diff, reps)
    return result
