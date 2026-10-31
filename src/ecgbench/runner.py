"""Run model attempts with checkpointing, logged retries, a spending cap, and scoring.

Each attempt is one line in <run_dir>/attempts.jsonl keyed by
(case_id, model, condition, attempt). Re-running skips keys that already have a
line, so an interrupted run resumes without repeating or silently re-paying for
completed attempts.

Failure policy (fixed before experiments):
  * InfrastructureError (network, timeout, 429, 5xx/529): retried up to
    execution.infra_retries times, each from a fresh conversation, with backoff.
    If retries are exhausted -> status "execution_failure".
  * Any other API error (e.g. 400 bad request) -> "execution_failure", no retry.
  * Model-side failures (refusal, max_tokens, turn limit, bad/invalid output)
    are NOT retried -> status "model_failure" or "ok" with an invalid response.
  * All failures stay in the denominator and are scored as zero predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from .config import PATHS, config_hash
from .models import CaseInput, InfrastructureError, build_prompt, estimate_cost_usd, load_prompts
from .responses import ParsedResponse, parse_response
from .scoring import score_case

LEDGER = PATHS["results"] / "spend_ledger.jsonl"
SETUP_ERRORS = {"AuthenticationError", "PermissionDeniedError", "NotFoundError", "SetupError"}


class BudgetExceeded(Exception):
    pass


def load_references() -> pd.DataFrame:
    df = pd.read_csv(PATHS["private"] / "references.csv")
    df["peaks_s"] = df["peaks_s"].apply(json.loads)
    return df


def case_input(case_id: str) -> CaseInput:
    d = PATHS["model_inputs"]
    return CaseInput(case_id, d / f"{case_id}.png", d / f"{case_id}_zoom_master.png",
                     json.loads((d / f"{case_id}_zoom_master.json").read_text()))


def spent_usd() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line).get("cost_usd") or 0.0 for line in LEDGER.read_text().splitlines() if line.strip())


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {tuple(json.loads(line)["key"]) for line in path.read_text().splitlines() if line.strip()}


def run_attempts(cfg: dict, adapter, case_ids: list[str], condition: str, run_dir: Path,
                 sleep=time.sleep, log=print) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "attempts.jsonl"
    done = _done_keys(out_path)
    refs = load_references().set_index("case_id")
    window, tol = tuple(cfg["task"]["scoring_window_s"]), cfg["task"]["tolerance_s"]
    system, user = build_prompt(cfg, condition)
    paid = adapter.provider != "mock"
    ex = cfg["execution"]
    costs_this_run = []
    if paid and adapter.model not in cfg.get("pricing_usd_per_mtok", {}):
        # Without a price the spend ledger would read $0 and the budget cap could not stop the run.
        raise BudgetExceeded(f"no pricing_usd_per_mtok entry for {adapter.model!r} (use 0/0 for a local server)")

    for cid in case_ids:
        for k in range(ex["attempts_per_case"]):
            key = [cid, adapter.model, condition, k]
            if tuple(key) in done:
                continue
            if paid:
                est = (sum(costs_this_run) / len(costs_this_run)) if costs_this_run else cfg["budget"]["first_attempt_estimate_usd"]
                worst = est * (1 + ex["infra_retries"])
                if spent_usd() + worst > cfg["budget"]["max_spend_usd"]:
                    raise BudgetExceeded(f"spent ${spent_usd():.2f}; next attempt could cost up to ${worst:.2f}; "
                                         f"cap ${cfg['budget']['max_spend_usd']:.2f}")
            case = case_input(cid)
            retries, out, status, error = [], None, None, None
            t0 = time.time()
            for i in range(ex["infra_retries"] + 1):
                try:
                    out = adapter.run(case, condition)
                    break
                except InfrastructureError as exc:
                    retries.append({"try": i, "error": str(exc)[:500], "partial_usage": getattr(exc, "partial_usage", None)})
                    if i < ex["infra_retries"]:
                        sleep(ex["retry_backoff_s"][min(i, len(ex["retry_backoff_s"]) - 1)])
                except Exception as exc:  # non-retryable (e.g. HTTP 400): recorded, not retried
                    if type(exc).__name__ in SETUP_ERRORS:
                        raise  # bad key, no access or unknown model: abort the run, don't log per-case failures
                    error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    break
            latency = time.time() - t0
            if out is None:
                status = "execution_failure"
                error = error or "infrastructure retries exhausted"
            elif out.failure:
                status = "model_failure"
            else:
                status = "ok"

            dur = case.zoom_geom["duration_s"]
            if out is not None and out.final_text is not None:
                parsed = parse_response(out.final_text, dur)
            else:
                parsed = ParsedResponse(False, reason=(out.failure if out else "execution_failure"))
            sc = score_case(parsed.peaks_s, refs.loc[cid, "peaks_s"], window, tol, valid=parsed.valid)

            usage = out.usage if out else None
            retry_usage = [r["partial_usage"] for r in retries if r.get("partial_usage")]
            cost = None
            if paid:
                cost = sum(estimate_cost_usd(cfg, adapter.model, u) or 0.0 for u in ([usage] if usage else []) + retry_usage)
                costs_this_run.append(cost)
            record = {
                "key": key, "case_id": cid, "model": adapter.model, "condition": condition, "attempt": k,
                "provider": adapter.provider, "model_settings": adapter.settings(),
                "model_reported": out.model_reported if out else None,
                "prompt_version": load_prompts()["version"], "system_prompt": system, "user_prompt": user,
                "input_image_sha256": hashlib.sha256(case.overview_png.read_bytes()).hexdigest()[:16],
                "status": status, "error": error, "retries": retries,
                "raw_response": out.final_text if out else None, "stop_reason": out.stop_reason if out else None,
                "model_failure": out.failure if out else None,
                "parsed": {"valid": parsed.valid, "peaks_s": parsed.peaks_s, "reason": parsed.reason, "notes": parsed.notes},
                "tool_calls": out.tool_calls if out else [], "turns": out.turns if out else 0,
                "usage": usage, "cost_usd": cost, "latency_s": round(latency, 3),
                "score": sc.to_dict(), "config_hash": config_hash(cfg), "timestamp": time.time(),
            }
            _append_line(out_path, json.dumps(record, default=str))
            if paid:
                LEDGER.parent.mkdir(parents=True, exist_ok=True)
                _append_line(LEDGER, json.dumps({"run_dir": str(run_dir), "key": key, "cost_usd": cost, "t": time.time()}))
            log(f"{cid} {adapter.model} {condition} #{k}: {status} valid={parsed.valid} "
                f"TP={sc.tp} FP={sc.fp} FN={sc.fn}" + (f" ${cost:.4f}" if cost is not None else ""))
    return out_path


def _append_line(path: Path, line: str) -> None:
    """One O_APPEND write per line, so parallel shards writing the same file never interleave."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


def load_attempts(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()])
