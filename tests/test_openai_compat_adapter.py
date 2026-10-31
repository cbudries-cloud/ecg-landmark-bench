"""OpenAICompatAdapter against a fake HTTP server (no network, no credentials)."""
import copy
import json

import httpx
import numpy as np
import pytest

from ecgbench.analysis import bootstrap, case_table, subgroups
from ecgbench.config import PATHS, config_hash, load_config
from ecgbench.models import InfrastructureError, OpenAICompatAdapter, SetupError
from ecgbench.runner import BudgetExceeded, case_input, load_references, run_attempts

pytestmark = pytest.mark.skipif(not (PATHS["private"] / "references.csv").exists(), reason="run `prepare` first")
CFG = load_config()
MC = {"provider": "openai_compat", "model": "fake-vlm", "base_url": "http://fake/v1", "api_key_env": None,
      "max_tokens": 1000, "structured_output": True}


def reply(content=None, tool_calls=None, finish="stop", refusal=None):
    msg = {"role": "assistant", "content": content, "refusal": refusal}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"model": "fake-vlm", "choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50}}


def adapter_with(responses, mc=MC):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        r = responses.pop(0)
        return r if isinstance(r, httpx.Response) else httpx.Response(200, json=r)

    return OpenAICompatAdapter(mc, CFG, transport=httpx.MockTransport(handler)), sent


@pytest.fixture
def case():
    return case_input(load_references().query("partition == 'dev'").iloc[0].case_id)


def zoom_call(i, args='{"start_s": 3.0, "end_s": 4.0}'):
    return {"id": f"call_{i}", "type": "function", "function": {"name": "zoom", "arguments": args}}


def test_zoom_then_answer(case):
    a, sent = adapter_with([reply(tool_calls=[zoom_call(1)], finish="tool_calls"),
                            reply('{"qrs_peaks_s": [3.5]}')])
    out = a.run(case, "image_zoom")
    assert out.final_text == '{"qrs_peaks_s": [3.5]}' and out.turns == 2 and out.failure is None
    assert out.usage["input_tokens"] == 2000 and out.tool_calls[0]["ok"]
    msgs = sent[1]["messages"]
    assert msgs[-2]["role"] == "tool" and msgs[-2]["tool_call_id"] == "call_1"
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent[0]["tools"][0]["function"]["name"] == "zoom"
    assert sent[0]["response_format"]["json_schema"]["schema"]["required"] == ["qrs_peaks_s"]


def test_image_only_sends_image_and_no_tools(case):
    a, sent = adapter_with([reply('{"qrs_peaks_s": []}')])
    out = a.run(case, "image_only")
    assert "tools" not in sent[0] and out.turns == 1
    assert sent[0]["messages"][0]["role"] == "system"
    assert sent[0]["messages"][1]["content"][0]["type"] == "image_url"


def test_malformed_tool_arguments_count_toward_limit(case):
    a, _ = adapter_with([reply(tool_calls=[zoom_call(1, "{not json")], finish="tool_calls"),
                         reply('{"qrs_peaks_s": []}')])
    out = a.run(case, "image_zoom")
    assert not out.tool_calls[0]["ok"] and out.final_text is not None


def test_refusal_length_and_errors(case):
    a, _ = adapter_with([reply(refusal="no")])
    assert a.run(case, "image_only").failure == "refusal"
    a, _ = adapter_with([reply("", finish="length")])
    assert a.run(case, "image_only").failure == "max_tokens"
    a, _ = adapter_with([httpx.Response(503, text="overloaded")])
    with pytest.raises(InfrastructureError):
        a.run(case, "image_only")
    a, _ = adapter_with([httpx.Response(401, text="bad key")])
    with pytest.raises(SetupError):
        a.run(case, "image_only")


def test_missing_key_is_a_setup_error(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(SetupError):
        OpenAICompatAdapter(MC | {"api_key_env": "NO_SUCH_KEY"}, CFG)


def test_base_url_can_come_from_env(monkeypatch):
    mc = {k: v for k, v in MC.items() if k != "base_url"} | {"base_url_env": "FAKE_ROUTER_URL"}
    monkeypatch.delenv("FAKE_ROUTER_URL", raising=False)
    with pytest.raises(SetupError):
        OpenAICompatAdapter(mc, CFG)
    monkeypatch.setenv("FAKE_ROUTER_URL", "http://fake/v1")
    assert OpenAICompatAdapter(mc, CFG).client.base_url.host == "fake"


def test_unpriced_model_is_refused(case, tmp_path):
    a, _ = adapter_with([reply('{"qrs_peaks_s": []}')])
    with pytest.raises(BudgetExceeded):
        run_attempts(CFG, a, [case.case_id], "image_only", tmp_path)


def test_model_roster_not_in_protocol_hash():
    cfg = copy.deepcopy(CFG)
    cfg["models"].append(MC)
    cfg["pricing_usd_per_mtok"]["fake-vlm"] = {"input": 0, "output": 0}
    assert config_hash(cfg) == config_hash(CFG)
    cfg["task"]["tolerance_s"] = 0.1
    assert config_hash(cfg) != config_hash(CFG)


def test_bootstrap_paired_and_ci_brackets_point():
    refs = [[3.5, 4.5, 5.5]] * 30
    good = case_table([(r, r, True) for r in refs], (3.0, 7.0), 0.15)
    worse = case_table([([3.5, 4.5], r, True) for r in refs[:10]] + [(r, r, True) for r in refs[10:]], (3.0, 7.0), 0.15)
    res = bootstrap({"a": worse, "b": good}, n_boot=500, seed=0, pairs=[("a", "b")])
    assert res["b"]["f1"] == 1.0 and res["b"]["f1_lo"] == 1.0
    assert res["a"]["f1_lo"] <= res["a"]["f1"] <= res["a"]["f1_hi"]
    assert res["b minus a"]["f1"] > 0 and res["b minus a"]["f1_lo"] > 0
    assert np.isclose(res["a"]["case_pass_rate"], 20 / 30)


def test_subgroups_from_headers():
    from ecgbench.data import all_records

    groups = [subgroups(CFG, r) for r in all_records(CFG)]
    assert sum("paced" in g for g in groups) == 9  # 9 records carry a "pacing." diagnosis (record 45: atrial + ventricular)
    assert sum("atrial_fibrillation" in g for g in groups) == 15
    assert "paced" in subgroups(CFG, 34)
