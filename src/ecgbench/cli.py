"""Command-line entry point: `python -m ecgbench.cli <command>`.

  prepare    manifest, split, eligibility, private references, clean + annotated images
  example    one record end to end with a clearly labelled MOCK prediction file
  baseline   XQRS raw-signal baseline on a partition
  run        model attempts (needs credentials and a budget)
  freeze     write the frozen protocol for the held-out evaluation
  summarize  pooled metrics table for one or more attempts.jsonl files
  report     final tables: case-level bootstrap CIs, paired zoom effects, subgroups
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import SUBGROUP_RULES, bootstrap, case_table, metrics, subgroups
from .baseline import xqrs_peaks_s
from .config import PATHS, ROOT, config_hash, load_config
from .coverage import eligibility_table
from .data import all_records, case_id, load_lead, reference_peaks_s
from .manifest import build_manifest
from .models import PROMPTS_FILE, make_adapter
from .render import render_annotated, render_clean, render_overlay
from .responses import parse_response
from .runner import BudgetExceeded, load_attempts, load_references, run_attempts
from .scoring import CaseScore, score_case, secondary_metrics, summarize
from .split import SPLIT_FILE, load_split, make_split

FROZEN = PATHS["results"] / "frozen" / "protocol.json"


def cmd_prepare(cfg, args):
    man = build_manifest(cfg)
    (PATHS["docs"]).mkdir(exist_ok=True)
    man.to_csv(PATHS["docs"] / "manifest.csv", index=False)
    print(f"manifest: {len(man)} files, {man.matches_official.sum()} match PhysioNet SHA256SUMS")
    if not man.matches_official.all():
        sys.exit("checksum mismatch; refusing to continue")

    if SPLIT_FILE.exists() and not args.force_split:
        split = load_split()
        if not split.equals(make_split(cfg)):
            sys.exit("existing split.csv differs from the seeded split; investigate before overwriting")
    else:
        split = make_split(cfg)
        split.to_csv(SPLIT_FILE, index=False)
    print(split.groupby("partition").agg(n=("record", "size"), pilot=("pilot", "sum")).to_string())

    elig = eligibility_table(cfg, all_records(cfg)).merge(split[["record", "partition"]], on="record")
    elig.to_csv(PATHS["results"] / "eligibility.csv", index=False)
    print("excluded:", elig[~elig.eligible][["record", "partition", "exclusion_reason"]].to_dict("records") or "none")

    PATHS["private"].mkdir(parents=True, exist_ok=True)
    rows = []
    for r in split.record:
        rows.append({"case_id": case_id(cfg, r), "record": r,
                     "partition": split.set_index("record").loc[r, "partition"],
                     "peaks_s": json.dumps([round(float(x), 4) for x in reference_peaks_s(cfg, r)])})
    pd.DataFrame(rows).to_csv(PATHS["private"] / "references.csv", index=False)
    assert len({x["case_id"] for x in rows}) == len(rows), "case_id collision"

    eligible = set(elig[elig.eligible].record)
    for i, r in enumerate(split.record):
        if r not in eligible:
            continue
        sig = load_lead(cfg, r)
        render_clean(sig, cfg, PATHS["model_inputs"], case_id(cfg, r))
        if split.set_index("record").loc[r, "partition"] == "dev":
            render_annotated(sig, reference_peaks_s(cfg, r), cfg,
                             PATHS["private"] / "annotated" / f"rec_{r:03d}.png", f"PRIVATE record {r}")
        if i % 25 == 0:
            print(f"rendered {i}/{len(split)}")
    print("clean images:", len(list(PATHS["model_inputs"].glob("c_*[!r].png"))))


def _cases(cfg, partition, pilot=False, limit=None):
    split = load_split()
    elig = pd.read_csv(PATHS["results"] / "eligibility.csv")
    sel = split[split.partition == partition]
    if pilot:
        sel = sel[sel.pilot]
    sel = sel[sel.record.isin(elig[elig.eligible].record)]
    recs = sorted(sel.record)[:limit] if limit else sorted(sel.record)
    return [(r, case_id(cfg, r)) for r in recs]


def cmd_example(cfg, args):
    """End-to-end on one dev record using a MOCK prediction file (not a model output)."""
    r = args.record
    split = load_split().set_index("record")
    if split.loc[r, "partition"] != "dev":
        sys.exit("examples must use development records")
    out = PATHS["results"] / "example"
    out.mkdir(parents=True, exist_ok=True)
    cid = case_id(cfg, r)
    sig = load_lead(cfg, r)
    refs = reference_peaks_s(cfg, r)
    render_clean(sig, cfg, PATHS["model_inputs"], cid)

    pred_file = out / f"MOCK_predictions_{cid}.json"
    if not pred_file.exists():
        # Deliberately imperfect mock built from the references: jitter, one dropped peak, one extra.
        rng = np.random.default_rng(0)
        a, b = cfg["task"]["scoring_window_s"]
        inwin = refs[(refs >= a) & (refs <= b)]
        mock = list(np.round(inwin + rng.normal(0, 0.02, len(inwin)), 3))
        mock.pop(len(mock) // 2)
        mock.append(round(float((inwin[0] + inwin[1]) / 2), 3))
        pred_file.write_text(json.dumps({"qrs_peaks_s": sorted(mock),
                                         "_label": "MOCK prediction for pipeline testing; not a model result"}))
    payload = json.loads(pred_file.read_text())
    payload.pop("_label", None)
    parsed = parse_response(json.dumps(payload), len(sig.signal) / sig.fs)
    sc = score_case(parsed.peaks_s, refs, tuple(cfg["task"]["scoring_window_s"]), cfg["task"]["tolerance_s"], parsed.valid)

    xq = xqrs_peaks_s(sig, cfg)
    sx = score_case(xq, refs, tuple(cfg["task"]["scoring_window_s"]), cfg["task"]["tolerance_s"])
    table = pd.DataFrame([
        {"method": "MOCK prediction (not a model)", **_row(sc)},
        {"method": "XQRS raw-signal baseline", **_row(sx)},
    ])
    table.to_csv(out / f"example_scores_{cid}.csv", index=False)
    render_overlay(sig, sc, refs, cfg, out / f"overlay_MOCK_{cid}.png",
                   f"record {r} | MOCK prediction (pipeline test, not a model)", parsed.peaks_s)
    render_overlay(sig, sx, refs, cfg, out / f"overlay_XQRS_{cid}.png", f"record {r} | XQRS raw-signal baseline", xq)
    print(f"record {r} -> case {cid}")
    print(f"clean image: {PATHS['model_inputs'] / (cid + '.png')}")
    print(table.to_string(index=False))
    print(f"outputs in {out}")


def _row(sc: CaseScore) -> dict:
    d = sc.to_dict()
    errs = np.abs(sc.errors_s) * 1e3
    return {k: d[k] for k in ("valid", "tp", "fp", "fn", "passed")} | {
        "median_abs_err_ms": round(float(np.median(errs)), 1) if len(errs) else None}


def cmd_baseline(cfg, args):
    win, tol = tuple(cfg["task"]["scoring_window_s"]), cfg["task"]["tolerance_s"]
    if args.partition == "test":
        _require_frozen(cfg, args)
    rows, summ = [], []
    for variant, corrected in (("xqrs_raw", False), ("xqrs_corrected", True)):
        scores, raw = [], []
        for r, cid in _cases(cfg, args.partition):
            sig = load_lead(cfg, r)
            preds, refs = xqrs_peaks_s(sig, cfg, corrected), reference_peaks_s(cfg, r)
            sc = score_case(preds, refs, win, tol)
            scores.append(sc)
            raw.append((preds, refs, True))
            rows.append({"variant": variant, "case_id": cid, "record": r, **_row(sc)})
        summ.append({"variant": variant, "partition": args.partition, **summarize(scores),
                     **secondary_metrics(raw, win, cfg["task"]["secondary_tolerances_s"])})
    out = PATHS["results"] / ("heldout" if args.partition == "test" else "dev") / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_case.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    print(s.round(3).T.to_string())


def _protocol(cfg) -> dict:
    return {
        "config_hash": config_hash(cfg),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "prompts_sha256": hashlib.sha256(PROMPTS_FILE.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(),
        "eligibility_sha256": hashlib.sha256((PATHS["results"] / "eligibility.csv").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((PATHS["docs"] / "manifest.csv").read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256((ROOT / "src/ecgbench/scoring.py").read_bytes()).hexdigest(),
    }


def cmd_freeze(cfg, args):
    if FROZEN.exists():
        sys.exit(f"{FROZEN} already exists. A changed protocol is a revision: freeze it under a new name.")
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    p = _protocol(cfg) | {"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    FROZEN.write_text(json.dumps(p, indent=1, default=str))
    print(f"frozen protocol written: {FROZEN} (config {p['config_hash']})")


def _require_frozen(cfg, args):
    if not FROZEN.exists():
        sys.exit("held-out partition is locked: run `freeze` first")
    frozen, now = json.loads(FROZEN.read_text()), _protocol(cfg)
    drift = [k for k in now if k != "config" and now[k] != frozen.get(k)]
    if drift:
        sys.exit(f"protocol differs from frozen version in: {drift}")
    if not getattr(args, "confirm_heldout", False):
        sys.exit("pass --confirm-heldout to run on the held-out partition")


def cmd_run(cfg, args):
    if args.partition == "test":
        _require_frozen(cfg, args)
    model_cfg = next((m for m in cfg["models"] if m["model"] == args.model), None)
    if model_cfg is None:
        sys.exit(f"model {args.model!r} not in config models list")
    cases = [cid for _, cid in _cases(cfg, args.partition, args.pilot, args.limit)]
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        cases = cases[k::n]  # disjoint slices: parallel shards never run the same case
    base = PATHS["results"] / ("heldout" if args.partition == "test" else "dev") / "runs"
    run_dir = base / args.run_name
    adapter = make_adapter(model_cfg, cfg)
    for cond in args.conditions or cfg["conditions"]:
        try:
            run_attempts(cfg, adapter, cases, cond, run_dir)
        except BudgetExceeded as exc:
            sys.exit(f"stopped by budget cap: {exc}")
    print(f"attempts saved to {run_dir / 'attempts.jsonl'}")


def cmd_summarize(cfg, args):
    refs = load_references().set_index("case_id")
    win = tuple(cfg["task"]["scoring_window_s"])
    rows = []
    for path in args.paths:
        df = load_attempts(path)
        for (model, cond), g in df.groupby(["model", "condition"]):
            scores = [CaseScore(**{k: v for k, v in s.items() if k != "passed"}) for s in g.score]
            raw = [(p["peaks_s"], refs.loc[c, "peaks_s"], p["valid"]) for p, c in zip(g.parsed, g.case_id)]
            n_exec = int((g.status == "execution_failure").sum())
            cost = g.cost_usd.dropna()
            rows.append({"run": Path(path).parent.name, "model": model, "condition": cond,
                         **summarize(scores, n_exec),
                         "model_failure_rate": float((g.status == "model_failure").mean()),
                         "mean_cost_usd": float(cost.mean()) if len(cost) else None,
                         "total_cost_usd": float(cost.sum()) if len(cost) else None,
                         "mean_latency_s": float(g.latency_s.mean()),
                         "mean_zoom_calls": float(g.tool_calls.apply(len).mean()),
                         **secondary_metrics(raw, win, cfg["task"]["secondary_tolerances_s"])})
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
    print(out.round(3).T.to_string())


def cmd_report(cfg, args):
    if args.partition == "test":
        _require_frozen(cfg, args)
    win = tuple(cfg["task"]["scoring_window_s"])
    cases = _cases(cfg, args.partition)
    refs = load_references().set_index("case_id")
    arm_cases = {}  # arm -> [(preds, refs, valid)] in case order
    for name, corrected in (("XQRS (raw signal)", False), ("XQRS + correct_peaks", True)):
        arm_cases[name] = [(xqrs_peaks_s(load_lead(cfg, r), cfg, corrected), refs.loc[c, "peaks_s"], True)
                           for r, c in cases]
    for path in args.runs:
        df = load_attempts(path)
        for (model, cond), g in df.groupby(["model", "condition"]):
            g = g[g.attempt == 0].set_index("case_id")
            missing = [c for _, c in cases if c not in g.index]
            if missing:
                sys.exit(f"{model} {cond}: {len(missing)} cases have no attempt in {path}")
            arm_cases[f"{model} | {cond}"] = [(g.loc[c, "parsed"]["peaks_s"], refs.loc[c, "peaks_s"],
                                               g.loc[c, "parsed"]["valid"]) for _, c in cases]
    pairs = [(f"{m} | image_only", f"{m} | image_zoom") for m in {a.split(" | ")[0] for a in arm_cases}
             if f"{m} | image_only" in arm_cases and f"{m} | image_zoom" in arm_cases]
    groups = {c: subgroups(cfg, r) for r, c in cases}
    rows, sub_rows = [], []
    for tol in [cfg["task"]["tolerance_s"], *cfg["task"]["secondary_tolerances_s"]]:
        tables = {a: case_table(cc, win, tol) for a, cc in arm_cases.items()}
        res = bootstrap(tables, cfg["report"]["n_bootstrap"], cfg["report"]["bootstrap_seed"], sorted(pairs))
        rows += [{"tolerance_ms": round(tol * 1000), "arm": a, **v} for a, v in res.items()]
        for g in [*SUBGROUP_RULES, "neither"]:
            keep = [i for i, (_, c) in enumerate(cases)
                    if (g in groups[c]) or (g == "neither" and not groups[c])]
            for a, t in tables.items():
                sub_rows.append({"tolerance_ms": round(tol * 1000), "subgroup": g, "n_cases": len(keep),
                                 "arm": a, **metrics([t[i] for i in keep])})
    out = Path(args.out) if args.out else PATHS["results"] / ("heldout" if args.partition == "test" else "dev") / "final"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "bootstrap.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(out / "subgroups.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"tables in {out}")


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from the git-ignored project .env; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser(prog="ecgbench")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--force-split", action="store_true")
    p = sub.add_parser("example"); p.add_argument("--record", type=int, default=None)
    p = sub.add_parser("baseline"); p.add_argument("--partition", choices=["dev", "test"], default="dev")
    p.add_argument("--confirm-heldout", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--partition", choices=["dev", "test"], default="dev")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--conditions", nargs="*", choices=["image_only", "image_zoom"])
    p.add_argument("--run-name", required=True)
    p.add_argument("--shard", help="k/n: run only every n-th case starting at k (parallel workers)")
    p.add_argument("--confirm-heldout", action="store_true")
    sub.add_parser("freeze")
    p = sub.add_parser("summarize"); p.add_argument("paths", nargs="+"); p.add_argument("--out")
    p = sub.add_parser("report"); p.add_argument("--partition", choices=["dev", "test"], default="dev")
    p.add_argument("--runs", nargs="*", default=[]); p.add_argument("--out")
    p.add_argument("--confirm-heldout", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    if args.cmd == "example" and args.record is None:
        args.record = int(load_split().query("pilot").record.min())
    {"prepare": cmd_prepare, "example": cmd_example, "baseline": cmd_baseline, "run": cmd_run,
     "freeze": cmd_freeze, "summarize": cmd_summarize, "report": cmd_report}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
