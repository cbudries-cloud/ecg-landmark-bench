"""One-to-one event matching between predicted and reference QRS peaks.

Matching rule
-------------
1. A prediction and a reference may match only if |error| <= tolerance.
2. Choose the matching with the most pairs.
3. Among those, the smallest total absolute error.
4. Ties are broken deterministically: inputs are sorted, errors are rounded to
   whole microseconds, and the remaining choice is made by
   scipy.optimize.linear_sum_assignment, which is deterministic for fixed input.

Scoring window [a, b]
---------------------
* References with a <= t <= b are scored (TP or FN).
* All valid predictions are matched against *all* references of the record,
  so a prediction just inside the window can pair with a reference just outside
  it (and vice versa) without being penalized.
* Unmatched predictions inside [a, b] are false positives. Predictions outside
  [a, b] never count as false positives; they are counted as `n_outside_window`.
* The eligibility rule guarantees expert coverage of [a - tol, b + tol], so every
  reference a scored prediction could match is known.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

_US = 1_000_000  # microsecond integer arithmetic avoids float edge effects at the tolerance


@dataclass
class CaseScore:
    valid: bool
    tp: int
    fp: int
    fn: int
    n_refs: int
    n_preds_in_window: int
    n_outside_window: int
    errors_s: list[float] = field(default_factory=list)  # signed (pred - ref) for scored TPs
    matched: list[tuple[float, float]] = field(default_factory=list)  # (pred, ref) scored TPs
    false_positives: list[float] = field(default_factory=list)
    false_negatives: list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.valid and self.fn == 0 and self.fp == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def match_events(preds: np.ndarray, refs: np.ndarray, tol_s: float) -> list[tuple[int, int]]:
    """Return (pred_idx, ref_idx) pairs of a maximum-cardinality, minimum-total-error matching.

    Indices refer to the arrays as given.
    """
    preds, refs = np.asarray(preds, float), np.asarray(refs, float)
    if len(preds) == 0 or len(refs) == 0:
        return []
    p_order, r_order = np.argsort(preds, kind="stable"), np.argsort(refs, kind="stable")
    p_us = np.round(preds[p_order] * _US).astype(np.int64)
    r_us = np.round(refs[r_order] * _US).astype(np.int64)
    tol_us = int(round(tol_s * _US))
    err = np.abs(p_us[:, None] - r_us[None, :])
    allowed = err <= tol_us
    if not allowed.any():
        return []
    # Every allowed pair gets a large bonus so that one extra match always beats
    # any saving in total error: bonus > (max possible pairs) * tol.
    bonus = (min(len(p_us), len(r_us)) + 1) * (tol_us + 1)
    cost = np.where(allowed, err - bonus, 0).astype(np.int64)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(p_order[i]), int(r_order[j])) for i, j in zip(rows, cols) if allowed[i, j]]
    return sorted(pairs, key=lambda ij: (refs[ij[1]], preds[ij[0]]))


def score_case(preds_s, refs_s, window_s: tuple[float, float], tol_s: float, valid: bool = True) -> CaseScore:
    """Score one case. An invalid response is scored as an empty prediction list."""
    a, b = window_s
    refs = np.sort(np.asarray(refs_s, float))
    preds = np.sort(np.asarray(preds_s if valid else [], float))
    in_win = lambda t: a <= t <= b  # noqa: E731 - closed interval on both ends
    pairs = match_events(preds, refs, tol_s)
    matched_p = {i for i, _ in pairs}
    matched_r = {j for _, j in pairs}

    scored_refs = [j for j in range(len(refs)) if in_win(refs[j])]
    tp_pairs = [(i, j) for i, j in pairs if in_win(refs[j])]
    fn = [float(refs[j]) for j in scored_refs if j not in matched_r]
    fp = [float(preds[i]) for i in range(len(preds)) if i not in matched_p and in_win(preds[i])]
    return CaseScore(
        valid=valid,
        tp=len(tp_pairs),
        fp=len(fp),
        fn=len(fn),
        n_refs=len(scored_refs),
        n_preds_in_window=int(sum(in_win(p) for p in preds)),
        n_outside_window=int(sum(not in_win(p) for p in preds)),
        errors_s=[float(preds[i] - refs[j]) for i, j in tp_pairs],
        matched=[(float(preds[i]), float(refs[j])) for i, j in tp_pairs],
        false_positives=fp,
        false_negatives=fn,
    )


def summarize(scores: list[CaseScore], n_execution_failures: int = 0) -> dict:
    """Pool counts across cases (micro-averaged).

    Execution failures must already be present in `scores` as invalid cases
    (all references missed); `n_execution_failures` only labels how many of the
    invalid cases were infrastructure failures rather than bad model output.

    Empty denominators: precision is NaN when there are no predictions in any
    window; recall is NaN when there are no references; F1 is NaN if either is NaN
    or both are 0 -> 0.0.
    """
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errs = np.array([e for s in scores for e in s.errors_s])
    n = len(scores)
    return {
        "n_cases": n,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "case_pass_rate": sum(s.passed for s in scores) / n if n else float("nan"),
        "invalid_rate": sum(not s.valid for s in scores) / n if n else float("nan"),
        "execution_failure_rate": n_execution_failures / n if n else float("nan"),
        "timing_n": int(len(errs)),
        "timing_mean_signed_ms": float(errs.mean() * 1e3) if len(errs) else float("nan"),
        "timing_median_abs_ms": float(np.median(np.abs(errs)) * 1e3) if len(errs) else float("nan"),
        "timing_p95_abs_ms": float(np.percentile(np.abs(errs), 95) * 1e3) if len(errs) else float("nan"),
    }


def secondary_metrics(cases: list[tuple[list[float], list[float], bool]], window_s, tolerances_s) -> dict:
    """Re-score (preds, refs, valid) cases at each secondary tolerance; same matching rule."""
    out = {}
    for tol in tolerances_s:
        tag = f"@{round(tol * 1000)}ms"
        s = summarize([score_case(p, r, window_s, tol, valid=v) for p, r, v in cases])
        out |= {f"precision{tag}": s["precision"], f"recall{tag}": s["recall"],
                f"f1{tag}": s["f1"], f"case_pass_rate{tag}": s["case_pass_rate"]}
    return out
