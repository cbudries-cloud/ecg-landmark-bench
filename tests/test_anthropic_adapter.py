"""AnthropicAdapter loop logic against a fake client (no network, no credentials)."""
from types import SimpleNamespace as NS

import pytest

from ecgbench.config import PATHS, load_config
from ecgbench.models import AnthropicAdapter, InfrastructureError
from ecgbench.runner import case_input, load_references

pytestmark = pytest.mark.skipif(not (PATHS["private"] / "references.csv").exists(), reason="run `prepare` first")
CFG = load_config()


def usage():
    return NS(input_tokens=1000, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def resp(stop, content):
    return NS(stop_reason=stop, content=content, usage=usage(), model="fake-model")


def adapter_with(responses):
    a = AnthropicAdapter.__new__(AnthropicAdapter)
    a.mc, a.cfg = {"model": "claude-opus-5-5", "effort": "high", "max_tokens": 1000}, CFG
    sent = []

    def create(**kw):
        sent.append(kw)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    a._create = create
    return a, sent


@pytest.fixture
def case():
    return case_input(load_references().query("partition == 'dev'").iloc[0].case_id)


def test_zoom_then_answer(case):
    a, sent = adapter_with([
        resp("tool_use", [NS(type="tool_use", id="t1", name="zoom", input={"start_s": 3.0, "end_s": 4.0})]),
        resp("end_turn", [NS(type="text", text='{"qrs_peaks_s": [3.5]}')]),
    ])
    out = a.run(case, "image_zoom")
    assert out.final_text == '{"qrs_peaks_s": [3.5]}' and out.turns == 2 and out.failure is None
    assert out.usage["input_tokens"] == 2000 and len(out.tool_calls) == 1 and out.tool_calls[0]["ok"]
    tool_result = sent[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and any(c["type"] == "image" for c in tool_result["content"])
    assert sent[0]["tools"][0]["name"] == "zoom"


def test_image_only_has_no_tools_and_one_turn(case):
    a, sent = adapter_with([resp("end_turn", [NS(type="text", text='{"qrs_peaks_s": []}')])])
    out = a.run(case, "image_only")
    assert "tools" not in sent[0] and out.turns == 1
    content = sent[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[1]["type"] == "text"


def test_refusal_and_turn_limit_are_model_failures(case):
    a, _ = adapter_with([resp("refusal", [])])
    assert a.run(case, "image_only").failure == "refusal"
    loop = [resp("tool_use", [NS(type="tool_use", id=f"t{i}", name="zoom", input={"start_s": 3, "end_s": 4})])
            for i in range(CFG["execution"]["max_turns"])]
    a, _ = adapter_with(loop)
    out = a.run(case, "image_zoom")
    assert out.failure == "turn_limit" and out.turns == CFG["execution"]["max_turns"]
    assert sum(not c["ok"] for c in out.tool_calls) == out.turns - CFG["zoom"]["max_calls"]


def test_infra_error_carries_partial_usage(case):
    a, _ = adapter_with([
        resp("tool_use", [NS(type="tool_use", id="t1", name="zoom", input={"start_s": 3, "end_s": 4})]),
        InfrastructureError("overloaded"),
    ])
    with pytest.raises(InfrastructureError) as e:
        a.run(case, "image_zoom")
    assert e.value.partial_usage["input_tokens"] == 1000
