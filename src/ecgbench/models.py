"""Model adapters and the zoom tool.

An adapter receives ONLY model-facing material: the clean overview PNG, the
prompt, and (zoom condition) a handle to the clean high-resolution render. It
never receives a record number, reference labels, annotated plots, or the
dataset directory. Every attempt starts a fresh conversation.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import ROOT
from .render import zoom_crop

PROMPTS_FILE = ROOT / "config" / "prompts.yaml"
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"qrs_peaks_s": {"type": "array", "items": {"type": "number"}}},
    "required": ["qrs_peaks_s"],
    "additionalProperties": False,
}


class InfrastructureError(Exception):
    """Designated retryable failure (network, timeout, rate limit, 5xx/overloaded)."""


class SetupError(Exception):
    """Missing key, no access or unknown model: the runner aborts instead of logging per-case failures."""


@dataclass
class CaseInput:
    case_id: str
    overview_png: Path
    zoom_master_png: Path
    zoom_geom: dict


@dataclass
class AttemptOutput:
    final_text: str | None
    stop_reason: str | None
    tool_calls: list[dict] = field(default_factory=list)
    turns: int = 0
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0,
                                                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
    model_reported: str | None = None
    failure: str | None = None  # model-side failure that ends the attempt (refusal, turn limit, ...)


def load_prompts() -> dict:
    return yaml.safe_load(PROMPTS_FILE.read_text())


def build_prompt(cfg: dict, condition: str) -> tuple[str, str]:
    p = load_prompts()
    a, b = cfg["task"]["scoring_window_s"]
    user = p["task"].format(lead_name=cfg["task"]["lead_display_name"], win_start=a, win_end=b)
    if condition == "image_zoom":
        user += p["zoom_addendum"].format(max_calls=cfg["zoom"]["max_calls"])
    return p["system"].strip(), user.strip()


ZOOM_TOOL = {
    "name": "zoom",
    "description": ("Return a higher-resolution view of the same clean ECG image between start_s and end_s, "
                    "in seconds from the start of the recording. Zoomed views keep the original time labels."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}},
        "required": ["start_s", "end_s"],
        "additionalProperties": False,
    },
}


class ZoomTool:
    """Crops the clean high-resolution render. Has no access to annotations."""

    def __init__(self, case: CaseInput, cfg: dict):
        self.case, self.z, self.calls = case, cfg["zoom"], 0

    def __call__(self, args: dict) -> tuple[bool, str, bytes | None]:
        """Returns (ok, text, png_bytes). Every call, valid or not, counts toward the limit."""
        self.calls += 1
        if self.calls > self.z["max_calls"]:
            return False, f"Zoom limit of {self.z['max_calls']} calls reached. Give your final JSON answer now.", None
        try:
            s, e = float(args["start_s"]), float(args["end_s"])
        except (KeyError, TypeError, ValueError):
            return False, "start_s and end_s must be numbers.", None
        dur = self.case.zoom_geom["duration_s"]
        s, e = max(0.0, s), min(dur, e)
        if not e - s >= self.z["min_span_s"] - 1e-9 or e - s > self.z["max_span_s"] + 1e-9:
            return False, (f"Requested span must be between {self.z['min_span_s']} and {self.z['max_span_s']} s "
                           f"within 0-{dur:.0f} s."), None
        png = zoom_crop(self.case.zoom_master_png, self.case.zoom_geom, s, e, self.z["max_output_long_edge_px"])
        return True, (f"Zoomed view of {s:.2f}-{e:.2f} s. Time labels are seconds from the start of the "
                      f"recording; labeled grid lines every 0.2 s, minor grid every 0.04 s."), png


def _png_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(png).decode()}}


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, model_cfg: dict, cfg: dict):
        import anthropic  # imported lazily so offline tests don't need credentials

        self.anthropic = anthropic
        self.mc, self.cfg = model_cfg, cfg
        # SDK-level retries off: the runner owns retries so every one is logged.
        self.client = anthropic.Anthropic(max_retries=0, timeout=cfg["execution"]["request_timeout_s"])

    @property
    def model(self) -> str:
        return self.mc["model"]

    def settings(self) -> dict:
        return {k: v for k, v in self.mc.items() if k != "provider"}

    def _create(self, **kw):
        a = self.anthropic
        try:
            return self.client.messages.create(**kw)
        except (a.APIConnectionError, a.APITimeoutError, a.RateLimitError, a.InternalServerError) as exc:
            raise InfrastructureError(f"{type(exc).__name__}: {exc}") from exc
        except a.APIStatusError as exc:
            if exc.status_code in (500, 502, 503, 504, 529):
                raise InfrastructureError(f"HTTP {exc.status_code}: {exc}") from exc
            raise

    def run(self, case: CaseInput, condition: str) -> AttemptOutput:
        system, user = build_prompt(self.cfg, condition)
        messages = [{"role": "user", "content": [_png_block(case.overview_png.read_bytes()),
                                                 {"type": "text", "text": user}]}]
        kw = dict(model=self.model, max_tokens=self.mc["max_tokens"], system=system,
                  output_config={"effort": self.mc["effort"],
                                 **({"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}
                                    if self.mc.get("structured_output", True) else {})})
        tool = ZoomTool(case, self.cfg) if condition == "image_zoom" else None
        if tool:
            kw["tools"] = [ZOOM_TOOL]
        out = AttemptOutput(final_text=None, stop_reason=None)
        max_turns = self.cfg["execution"]["max_turns"] if tool else 1
        while True:
            if out.turns >= max_turns:
                out.failure = "turn_limit"
                return out
            try:
                resp = self._create(messages=messages, **kw)
            except InfrastructureError as exc:
                exc.partial_usage = dict(out.usage)  # billable turns completed before the failure
                raise
            out.turns += 1
            out.model_reported = resp.model
            for k in out.usage:
                out.usage[k] += getattr(resp.usage, k, 0) or 0
            out.stop_reason = resp.stop_reason
            if resp.stop_reason == "refusal":
                out.failure = "refusal"
                return out
            if resp.stop_reason == "max_tokens":
                out.failure = "max_tokens"
                return out
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                out.final_text = "".join(b.text for b in resp.content if b.type == "text")
                return out
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for u in uses:
                if u.name == "zoom" and tool:
                    ok, text, png = tool(u.input)
                else:
                    ok, text, png = False, f"Unknown tool {u.name!r}.", None
                out.tool_calls.append({"name": u.name, "input": u.input, "ok": ok, "message": text,
                                       "image_sha256": _sha(png) if png else None})
                content = [{"type": "text", "text": text}] + ([_png_block(png)] if png else [])
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": content,
                                **({} if ok else {"is_error": True})})
            messages.append({"role": "user", "content": results})


class OpenAICompatAdapter:
    """Any provider that speaks the OpenAI Chat Completions format over HTTP.

    Covers closed models (OpenAI; Google Gemini via its OpenAI-compatible endpoint)
    and open-weight models (hosted routers such as OpenRouter or Together, or a local
    vLLM / Ollama server). Same prompt, image and zoom tool as the Anthropic adapter.

    Chat Completions tool results are text-only for most providers, so a zoom image
    is sent as a user message right after the tool results.

    model_cfg keys: model, base_url or base_url_env (the URL read from .env),
    api_key_env (null for a local server),
    max_tokens, max_tokens_param ("max_tokens" or "max_completion_tokens"),
    structured_output (response_format json_schema), strict_tools, extra_body.
    """

    provider = "openai_compat"

    def __init__(self, model_cfg: dict, cfg: dict, transport=None):
        import os

        import httpx  # installed with the anthropic SDK

        self.httpx, self.mc, self.cfg = httpx, model_cfg, cfg
        headers = {}
        if model_cfg.get("api_key_env"):
            key = os.environ.get(model_cfg["api_key_env"])
            if not key:
                raise SetupError(f"{model_cfg['api_key_env']} is not set (add it to .env)")
            headers["Authorization"] = f"Bearer {key}"
        base_url = model_cfg.get("base_url") or os.environ.get(model_cfg.get("base_url_env") or "")
        if not base_url:
            raise SetupError(f"{model_cfg.get('base_url_env')} is not set (add it to .env)")
        self.client = httpx.Client(base_url=base_url.rstrip("/") + "/", headers=headers,
                                   timeout=cfg["execution"]["request_timeout_s"], transport=transport)

    @property
    def model(self) -> str:
        return self.mc["model"]

    def settings(self) -> dict:
        return {k: v for k, v in self.mc.items() if k not in ("provider", "api_key_env")}

    def _create(self, body: dict) -> dict:
        try:
            r = self.client.post("chat/completions", json=body)
        except (self.httpx.TimeoutException, self.httpx.TransportError) as exc:
            raise InfrastructureError(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code in (429, 500, 502, 503, 504, 529):
            raise InfrastructureError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code in (401, 403, 404):
            raise SetupError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        # Some routers name the upstream host that actually served the request.
        if r.headers.get("x-router-provider"):
            body["_served_by"] = r.headers["x-router-provider"]
        return body

    def run(self, case: CaseInput, condition: str) -> AttemptOutput:
        system, user = build_prompt(self.cfg, condition)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": [_image_url(case.overview_png.read_bytes()),
                                                 {"type": "text", "text": user}]}]
        body = {"model": self.model, self.mc.get("max_tokens_param", "max_tokens"): self.mc["max_tokens"],
                **self.mc.get("extra_body", {})}
        if self.mc.get("structured_output", False):
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "qrs_peaks", "schema": RESPONSE_SCHEMA, "strict": True}}
        tool = ZoomTool(case, self.cfg) if condition == "image_zoom" else None
        if tool:
            fn = {"name": ZOOM_TOOL["name"], "description": ZOOM_TOOL["description"],
                  "parameters": ZOOM_TOOL["input_schema"]}
            body["tools"] = [{"type": "function", "function": fn | (
                {"strict": True} if self.mc.get("strict_tools", False) else {})}]
        out = AttemptOutput(final_text=None, stop_reason=None)
        max_turns = self.cfg["execution"]["max_turns"] if tool else 1
        while True:
            if out.turns >= max_turns:
                out.failure = "turn_limit"
                return out
            try:
                resp = self._create(body | {"messages": messages})
            except InfrastructureError as exc:
                exc.partial_usage = dict(out.usage)
                raise
            out.turns += 1
            out.model_reported = resp.get("model") + (f" via {resp['_served_by']}" if resp.get("_served_by") else "")
            u = resp.get("usage") or {}
            out.usage["input_tokens"] += u.get("prompt_tokens", 0) or 0
            out.usage["output_tokens"] += u.get("completion_tokens", 0) or 0
            choice = resp["choices"][0]
            msg, out.stop_reason = choice["message"], choice.get("finish_reason")
            if msg.get("refusal") or out.stop_reason == "content_filter":
                out.failure = "refusal"
                return out
            if out.stop_reason == "length":
                out.failure = "max_tokens"
                return out
            calls = msg.get("tool_calls") or []
            if not calls:
                out.final_text = msg.get("content") or ""
                return out
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            images = []
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = None
                if name == "zoom" and tool and isinstance(args, dict):
                    ok, text, png = tool(args)
                elif name == "zoom" and tool:
                    tool.calls += 1  # malformed arguments still count toward the limit
                    ok, text, png = False, "Arguments must be a JSON object with start_s and end_s.", None
                else:
                    ok, text, png = False, f"Unknown tool {name!r}.", None
                out.tool_calls.append({"name": name, "input": args, "ok": ok, "message": text,
                                       "image_sha256": _sha(png) if png else None})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if png:
                    images += [{"type": "text", "text": f"Image for zoom call {c['id']}:"}, _image_url(png)]
            if images:
                messages.append({"role": "user", "content": images})


def _image_url(png: bytes) -> dict:
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.standard_b64encode(png).decode()}}


class MockAdapter:
    """MOCK, NOT A MODEL. Replays scripted responses to exercise the pipeline offline.

    `script` is a list of steps: {"zoom": [start, end]} calls the real zoom tool;
    {"final": "<text>"} ends the attempt; {"infra_error": True} raises an
    InfrastructureError (to test retries).
    """

    provider = "mock"

    def __init__(self, script_for_case, cfg: dict, name: str = "MOCK-scripted"):
        self.script_for_case, self.cfg, self.model = script_for_case, cfg, name

    def settings(self) -> dict:
        return {"model": self.model, "note": "scripted mock, not a model result"}

    def run(self, case: CaseInput, condition: str) -> AttemptOutput:
        out = AttemptOutput(final_text=None, stop_reason=None, model_reported=self.model)
        tool = ZoomTool(case, self.cfg) if condition == "image_zoom" else None
        for step in self.script_for_case(case.case_id, condition):
            out.turns += 1
            if step.get("infra_error"):
                raise InfrastructureError("mock infrastructure error")
            if "zoom" in step:
                if not tool:
                    out.failure = "tool_not_available"
                    return out
                ok, text, png = tool({"start_s": step["zoom"][0], "end_s": step["zoom"][1]})
                out.tool_calls.append({"name": "zoom", "input": step["zoom"], "ok": ok, "message": text,
                                       "image_sha256": _sha(png) if png else None})
                continue
            out.final_text, out.stop_reason = step["final"], "end_turn"
            return out
        out.failure = "script_ended_without_answer"
        return out


def _sha(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()[:16]


def make_adapter(model_cfg: dict, cfg: dict):
    if model_cfg["provider"] == "anthropic":
        return AnthropicAdapter(model_cfg, cfg)
    if model_cfg["provider"] == "openai_compat":
        return OpenAICompatAdapter(model_cfg, cfg)
    raise ValueError(f"unsupported provider {model_cfg['provider']!r}")


def estimate_cost_usd(cfg: dict, model: str, usage: dict) -> float | None:
    price = cfg.get("pricing_usd_per_mtok", {}).get(model)
    if not price:
        return None
    # Cache reads/writes are not used by this benchmark; they are priced at the input rate as an upper bound.
    tokens_in = sum(usage.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    return (tokens_in * price["input"] + usage.get("output_tokens", 0) * price["output"]) / 1e6
