"""Scorer correctness tests (brief Step 7). Times are in seconds."""
import math

import pytest

from ecgbench.responses import parse_response
from ecgbench.scoring import match_events, score_case, secondary_metrics, summarize

W, TOL = (3.0, 7.0), 0.150
REFS = [3.2, 4.1, 5.0, 5.9, 6.8]


def s(preds, refs=REFS, valid=True):
    return score_case(preds, refs, W, TOL, valid=valid)


def test_perfect_predictions():
    r = s(REFS)
    assert (r.tp, r.fp, r.fn) == (5, 0, 0) and r.passed
    assert all(e == 0 for e in r.errors_s)


def test_missing_prediction():
    r = s([3.2, 4.1, 5.9, 6.8])
    assert (r.tp, r.fp, r.fn) == (4, 0, 1) and r.false_negatives == [5.0] and not r.passed


def test_extra_prediction():
    r = s(REFS + [4.55])
    assert (r.tp, r.fp, r.fn) == (5, 1, 0) and r.false_positives == [4.55]


def test_duplicate_predictions_only_one_matches():
    r = s(REFS + [5.0])
    assert (r.tp, r.fp, r.fn) == (5, 1, 0)


def test_reordered_predictions_score_identically():
    assert s(list(reversed(REFS))).to_dict() == s(REFS).to_dict()


def test_prediction_exactly_at_tolerance_matches():
    # 4.0 + 0.15 is 4.1500000000000004 in floating point; microsecond rounding makes it exact.
    r = score_case([4.15], [4.0], W, TOL)
    assert (r.tp, r.fp, r.fn) == (1, 0, 0)
    assert math.isclose(r.errors_s[0], 0.15, abs_tol=1e-9)


def test_prediction_just_outside_tolerance_does_not_match():
    r = score_case([4.151], [4.0], W, TOL)
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)


def test_competing_matches_maximize_count():
    # Greedy nearest-neighbour would pair 4.10 with ref 4.05 (err .05), leaving
    # ref 4.20 unmatched. Optimal: 3.95->4.05 (.10) and 4.10->4.20 (.10): two matches.
    refs = [4.05, 4.20]
    preds = [3.95, 4.10]
    r = score_case(preds, refs, W, TOL)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)


def test_competing_matches_minimize_error_among_max():
    # Both assignments match both; the optimal one has total error .02 not .18.
    pairs = match_events([5.00, 5.10], [5.01, 5.09], TOL)
    assert sorted(pairs) == [(0, 0), (1, 1)]


def test_empty_output_is_valid_but_misses_everything():
    r = s([])
    assert r.valid and (r.tp, r.fp, r.fn) == (0, 0, 5) and not r.passed


def test_invalid_response_scored_as_empty_and_flagged():
    r = s(REFS, valid=False)
    assert not r.valid and (r.tp, r.fp, r.fn) == (0, 0, 5) and not r.passed


def test_window_boundaries():
    refs = [2.95, 3.0, 7.0, 7.05]
    # 3.0 and 7.0 are scored (closed interval); 2.95 and 7.05 are not.
    r = score_case([], refs, W, TOL)
    assert r.n_refs == 2 and r.false_negatives == [3.0, 7.0]
    # A prediction inside the window that matches an out-of-window reference is neither TP nor FP.
    r = score_case([3.05], [2.95], W, TOL)
    assert (r.tp, r.fp, r.fn) == (0, 0, 0)
    # A prediction outside the window that matches an in-window reference counts as a TP.
    r = score_case([2.9], [3.0], W, TOL)
    assert (r.tp, r.fp, r.fn, r.n_outside_window) == (1, 0, 0, 1)
    # Unmatched predictions outside the window are not false positives.
    r = score_case([1.0, 9.0], [5.0], W, TOL)
    assert (r.fp, r.n_outside_window) == (0, 2)


def test_summary_pools_counts_and_handles_empty_denominators():
    a, b = s(REFS), s([])
    out = summarize([a, b])
    assert (out["tp"], out["fp"], out["fn"]) == (5, 0, 5)
    assert out["precision"] == 1.0 and out["recall"] == 0.5
    assert out["case_pass_rate"] == 0.5
    none = summarize([s([]), s([])])
    assert math.isnan(none["precision"]) and none["recall"] == 0.0 and math.isnan(none["f1"])


# ---- response validation -------------------------------------------------

DUR = 10.0


def test_parse_valid():
    p = parse_response('{"qrs_peaks_s": [4.1, 3.2]}', DUR)
    assert p.valid and p.peaks_s == [3.2, 4.1]


def test_parse_code_fence_accepted_and_noted():
    p = parse_response('```json\n{"qrs_peaks_s": [3.2]}\n```', DUR)
    assert p.valid and "code_fence_stripped" in p.notes


def test_parse_empty_list_valid():
    p = parse_response('{"qrs_peaks_s": []}', DUR)
    assert p.valid and p.peaks_s == []


def test_parse_duplicates_kept_and_noted():
    p = parse_response('{"qrs_peaks_s": [3.2, 3.2]}', DUR)
    assert p.valid and p.peaks_s == [3.2, 3.2] and "duplicate_timestamps" in p.notes


@pytest.mark.parametrize("text,reason", [
    ("", "empty_response"),
    (None, "empty_response"),
    ("The peaks are at 3.2 and 4.1 s", "malformed_json"),
    ('{"qrs_peaks_s": [3.2, 4.1]', "malformed_json"),
    ('Here you go: {"qrs_peaks_s": [3.2]}', "malformed_json"),
    ('{"peaks": [3.2]}', "wrong_schema"),
    ('{"qrs_peaks_s": [3.2], "confidence": 0.9}', "wrong_schema"),
    ('[3.2, 4.1]', "wrong_schema"),
    ('{"qrs_peaks_s": 3.2}', "not_a_list"),
    ('{"qrs_peaks_s": ["3.2"]}', "non_numeric_value"),
    ('{"qrs_peaks_s": [true]}', "non_numeric_value"),
    ('{"qrs_peaks_s": [null]}', "non_numeric_value"),
    ('{"qrs_peaks_s": [NaN]}', "malformed_json"),
    ('{"qrs_peaks_s": [Infinity]}', "malformed_json"),
    ('{"qrs_peaks_s": [1e400]}', "non_finite_value"),
    # Milliseconds instead of seconds:
    ('{"qrs_peaks_s": [3200, 4100]}', "outside_recording_likely_wrong_unit"),
    ('{"qrs_peaks_s": [-0.5]}', "outside_recording_likely_wrong_unit"),
])
def test_parse_invalid(text, reason):
    p = parse_response(text, DUR)
    assert not p.valid and p.reason == reason


def test_wrong_unit_that_fits_inside_recording_is_scored_not_repaired():
    # Minutes-scale mistake (values / 60) stays inside [0, 10] and is valid but wrong:
    p = parse_response('{"qrs_peaks_s": [0.053, 0.068]}', DUR)
    assert p.valid
    r = score_case(p.peaks_s, REFS, W, TOL)
    assert r.tp == 0 and r.fn == 5 and r.fp == 0  # outside window: not FP, but misses everything


def test_secondary_tolerances_are_stricter():
    cases = [([3.24, 4.13, 5.0], [3.2, 4.1, 5.0], True)]  # errors 40, 30, 0 ms
    m = secondary_metrics(cases, W, [0.050, 0.025])
    assert m["recall@50ms"] == 1.0 and m["case_pass_rate@50ms"] == 1.0
    assert abs(m["recall@25ms"] - 1 / 3) < 1e-9 and m["case_pass_rate@25ms"] == 0.0
