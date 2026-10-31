"""Pipeline tests with the scripted MOCK adapter (no network, no spending).

Requires `python -m ecgbench.cli prepare` to have been run (real LUDB data).
"""
import json

import pytest
from PIL import Image

from ecgbench.config import PATHS, load_config
from ecgbench.models import MockAdapter, ZoomTool, build_prompt
from ecgbench.runner import BudgetExceeded, case_input, load_attempts, load_references, run_attempts

pytestmark = pytest.mark.skipif(not (PATHS["private"] / "references.csv").exists(), reason="run `prepare` first")
CFG = load_config()


@pytest.fixture
def dev_case():
    refs = load_references()
    row = refs[refs.partition == "dev"].iloc[0]
    return row.case_id, row.peaks_s


def test_clean_image_has_no_answer_leaks(dev_case):
    cid, _ = dev_case
    img = Image.open(PATHS["model_inputs"] / f"{cid}.png")
    assert set(img.info) <= {"dpi"}
    assert cid.startswith("c_") and not any(ch.isdigit() for ch in cid[:2])
    geom = json.loads((PATHS["model_inputs"] / f"{cid}_zoom_master.json").read_text())
    assert set(geom) == {"duration_s", "master_width_px", "master_height_px",
                         "axes_left_px", "axes_right_px", "axes_bottom_px"}


def test_prompt_has_no_peak_count_and_states_window():
    for cond in ("image_only", "image_zoom"):
        system, user = build_prompt(CFG, cond)
        assert "3.0 s to 7.0 s" in user and "qrs_peaks_s" in user and "Lead II" in user
        assert ("zoom" in user) == (cond == "image_zoom")


def test_zoom_limit_and_validation(dev_case):
    tool = ZoomTool(case_input(dev_case[0]), CFG)
    assert tool({"start_s": 3, "end_s": 3.1})[0] is False  # too narrow
    for _ in range(CFG["zoom"]["max_calls"] - 1):
        assert tool({"start_s": 3, "end_s": 4})[0]
    ok, msg, png = tool({"start_s": 3, "end_s": 4})
    assert not ok and "limit" in msg and png is None


def test_mock_run_zoom_retry_resume(tmp_path, dev_case):
    cid, refs = dev_case
    in_win = [t for t in refs if 3.0 <= t <= 7.0]
    calls = {"n": 0}

    def script(case_id, condition):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"infra_error": True}]  # first try fails, must be retried and logged
        return [{"zoom": [3.0, 4.0]}, {"final": json.dumps({"qrs_peaks_s": in_win})}]

    adapter = MockAdapter(script, CFG)
    path = run_attempts(CFG, adapter, [cid], "image_zoom", tmp_path, sleep=lambda s: None, log=lambda m: None)
    df = load_attempts(path)
    assert len(df) == 1
    r = df.iloc[0]
    assert r.status == "ok" and len(r.retries) == 1 and len(r.tool_calls) == 1
    assert r.score["passed"] and r.cost_usd is None
    # Resume: nothing is repeated.
    run_attempts(CFG, adapter, [cid], "image_zoom", tmp_path, sleep=lambda s: None, log=lambda m: None)
    assert len(load_attempts(path)) == 1 and calls["n"] == 2


def test_infra_exhaustion_is_recorded_as_execution_failure(tmp_path, dev_case):
    adapter = MockAdapter(lambda c, k: [{"infra_error": True}], CFG)
    path = run_attempts(CFG, adapter, [dev_case[0]], "image_only", tmp_path, sleep=lambda s: None, log=lambda m: None)
    r = load_attempts(path).iloc[0]
    assert r.status == "execution_failure" and len(r.retries) == CFG["execution"]["infra_retries"] + 1
    assert not r.score["valid"] and r.score["fn"] == r.score["n_refs"] > 0


def test_malformed_model_output_is_invalid_not_repaired(tmp_path, dev_case):
    adapter = MockAdapter(lambda c, k: [{"final": "peaks at 3.2 and 4.1"}], CFG)
    path = run_attempts(CFG, adapter, [dev_case[0]], "image_only", tmp_path, sleep=lambda s: None, log=lambda m: None)
    r = load_attempts(path).iloc[0]
    assert r.status == "ok" and not r.parsed["valid"] and r.parsed["reason"] == "malformed_json"


def test_budget_cap_blocks_paid_adapter(tmp_path, dev_case):
    class FakePaid(MockAdapter):
        provider = "fake-paid"
    adapter = FakePaid(lambda c, k: [{"final": '{"qrs_peaks_s": []}'}], CFG)
    cfg = {**CFG, "budget": {"max_spend_usd": 0.0, "first_attempt_estimate_usd": 0.25}}
    with pytest.raises(BudgetExceeded):
        run_attempts(cfg, adapter, [dev_case[0]], "image_only", tmp_path, sleep=lambda s: None, log=lambda m: None)
    assert not (tmp_path / "attempts.jsonl").exists()


def test_setup_errors_abort_instead_of_logging(tmp_path, dev_case):
    class AuthenticationError(Exception):
        pass

    class Broken(MockAdapter):
        def run(self, case, condition):
            raise AuthenticationError("invalid x-api-key")

    with pytest.raises(AuthenticationError):
        run_attempts(CFG, Broken(lambda c, k: [], CFG), [dev_case[0]], "image_only", tmp_path,
                     sleep=lambda s: None, log=lambda m: None)
    assert not (tmp_path / "attempts.jsonl").exists()
