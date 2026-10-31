"""Build the results page: docs/site/index.html, one self-contained file (figures inlined; PNGs kept in img/).

Reads only result files already on disk: the held-out summary and bootstrap tables,
and the attempt logs (for failure counts and examples). Re-run after any new run:

    .venv/bin/python -m ecgbench.cli summarize results/heldout/runs/heldout-v1-*/attempts.jsonl \
        --out results/heldout/runs/summary.csv
    .venv/bin/python -m ecgbench.cli report --partition test --confirm-heldout \
        --runs results/heldout/runs/heldout-v1-*/attempts.jsonl
    .venv/bin/python scripts/build_site.py
"""
from __future__ import annotations

import base64
import glob
import html
import math
import re

import numpy as np
import pandas as pd

from ecgbench.analysis import subgroups
from ecgbench.baseline import xqrs_peaks_s
from ecgbench.config import PATHS, ROOT, load_config
from ecgbench.data import case_id, load_lead, reference_peaks_s
from ecgbench.models import ZoomTool
from ecgbench.render import AXES_RECT, render_overlay
from ecgbench.runner import case_input, load_attempts, load_references
from ecgbench.scoring import score_case

CFG = load_config()
SITE = PATHS["docs"] / "site"
IMG = SITE / "img"
HELDOUT = PATHS["results"] / "heldout"
WIN, TOL = tuple(CFG["task"]["scoring_window_s"]), CFG["task"]["tolerance_s"]

NAMES = {
    "claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-5.6-luna": "GPT-5.6 Luna", "grok-4.7": "Grok 4.7",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash", "muse-spark-open": "Muse Spark Open", "kimi-k3": "Kimi K3",
}
OPEN = {"deepseek-v4.1-flash", "muse-spark-open", "kimi-k3"}
COLORS = {
    "claude-opus-5-5": "#a3243b", "claude-sonnet-5-5": "#e07a8a", "gpt-6.1-sol": "#2d6aa8",
    "gpt-5.6-luna": "#8fb5dd", "grok-4.7": "#3a3540", "deepseek-v4.1-flash": "#3f8f6b",
    "muse-spark-open": "#c08a2b", "kimi-k3": "#7a5ba6",
}
COND = {"image_only": "image only", "image_zoom": "image + zoom"}
COND_SHORT = {"image_only": "image", "image_zoom": "zoom"}
FAILURES = [("invalid", "No usable answer"), ("missed", "Missed a beat"), ("added", "Added a beat"),
            ("off", "All found, some over 25 ms off"), ("ok", "All within 25 ms")]

# Plain definitions, shown on hover.
GLOSS = {
    "QRS": "The tall, sharp spike of each heartbeat on an ECG, made as the main pumping chambers contract.",
    "Lead II": "One of the twelve standard views of the heart's electrical activity; the one most often used to read rhythm.",
    "tolerance": "How far, in milliseconds, a model's mark may sit from the expert's mark and still count as the same beat.",
    "p95": "95th-percentile error: 95 out of 100 matched beats were placed at least this close to the expert mark.",
    "bootstrap": "A way to estimate uncertainty: the 100 test records are resampled many times and the score recomputed each time.",
    "held-out": "The 100 records set aside and run once, after every rule was fixed.",
    "open-weight": "Models whose trained weights are published, so anyone can run or retrain them.",
}


def esc(s) -> str:
    return html.escape(str(s))


def term(word: str) -> str:
    return f'<abbr title="{esc(GLOSS[word])}">{esc(word)}</abbr>'


def pct(v, d=0) -> str:
    return f"{v * 100:.{d}f}%"


def name(model: str) -> str:
    return NAMES.get(model, model)


# ── data ──────────────────────────────────────────────────────────────────────

def load_tables():
    summ = pd.read_csv(HELDOUT / "runs" / "summary.csv")
    boot = pd.read_csv(HELDOUT / "final" / "bootstrap.csv")
    subs = pd.read_csv(HELDOUT / "final" / "subgroups.csv")
    arms = []
    for _, r in summ.iterrows():
        key = f"{r.model} | {r.condition}"
        b = {t: boot[(boot.tolerance_ms == t) & (boot.arm == key)].iloc[0] for t in (150, 50, 25)}
        arms.append({"model": r.model, "condition": r.condition, "name": name(r.model),
                     "tp": int(r.tp), "fp": int(r.fp), "fn": int(r.fn), "invalid": float(r.invalid_rate),
                     **{f"pass{t}": b[t].case_pass_rate for t in b},
                     **{f"pass{t}_lo": b[t].case_pass_rate_lo for t in b},
                     **{f"pass{t}_hi": b[t].case_pass_rate_hi for t in b},
                     "f1_150": b[150].f1, "f1_150_lo": b[150].f1_lo, "f1_150_hi": b[150].f1_hi,
                     "p95": b[150].p95_abs_err_ms, "p95_lo": b[150].p95_abs_err_ms_lo, "p95_hi": b[150].p95_abs_err_ms_hi,
                     "median_ms": r.timing_median_abs_ms, "mean_signed": r.timing_mean_signed_ms,
                     "cost": r.mean_cost_usd, "zooms": r.mean_zoom_calls, "latency": r.mean_latency_s})
    xq = {t: boot[(boot.tolerance_ms == t) & (boot.arm == "XQRS (raw signal)")].iloc[0] for t in (150, 50, 25)}
    xqrs = {"f1_150": xq[150].f1, "p95": xq[150].p95_abs_err_ms, **{f"pass{t}": xq[t].case_pass_rate for t in xq}}
    diffs = boot[(boot.tolerance_ms == 150) & boot.arm.str.contains(" minus ")].copy()
    diffs["model"] = diffs.arm.str.split(" | ").str[0]
    return pd.DataFrame(arms), xqrs, diffs, subs


def load_all_attempts() -> pd.DataFrame:
    paths = sorted(glob.glob(str(HELDOUT / "runs" / "heldout-v1-*" / "attempts.jsonl")))
    df = pd.concat([load_attempts(p) for p in paths], ignore_index=True)
    return df[df.attempt == 0]


def classify(df: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """One outcome per attempt, in the priority order of FAILURES."""
    out = []
    for _, r in df.iterrows():
        s = r.score
        if not s["valid"]:
            k = "invalid"
        elif s["fn"]:
            k = "missed"
        elif s["fp"]:
            k = "added"
        else:
            strict = score_case(r.parsed["peaks_s"], refs.loc[r.case_id, "peaks_s"], WIN, 0.025)
            k = "ok" if strict.passed else "off"
        out.append({"model": r.model, "condition": r.condition, "case_id": r.case_id, "outcome": k})
    return pd.DataFrame(out)


# ── images from real cases ───────────────────────────────────────────────────

def overlay(record: int, preds, title: str, file: str) -> None:
    sig, rr = load_lead(CFG, record), reference_peaks_s(CFG, record)
    render_overlay(sig, score_case(preds, rr, WIN, TOL), rr, CFG, IMG / file, title, preds)


def make_images(att: pd.DataFrame, outcomes: pd.DataFrame, refs: pd.DataFrame) -> dict:
    IMG.mkdir(parents=True, exist_ok=True)
    for old in IMG.glob("*.png"):
        old.unlink()
    dev = load_attempts(PATHS["results"] / "dev" / "runs" / "dev-v1" / "attempts.jsonl")

    def dev_preds(record):
        row = dev[(dev.case_id == case_id(CFG, record)) & (dev.model == "claude-opus-5-5") & (dev.condition == "image_zoom")]
        return row.iloc[0].parsed["peaks_s"]

    # Hero and task figure: development record 4.
    cid = case_id(CFG, 4)
    overlay(4, dev_preds(4), "Claude Opus 5.5 with zoom, development record", "hero.png")
    (IMG / "input.png").write_bytes((PATHS["model_inputs"] / f"{cid}.png").read_bytes())
    _, _, png = ZoomTool(case_input(cid), CFG)({"start_s": 4.4, "end_s": 5.4})
    (IMG / "zoom.png").write_bytes(png)

    # Paced development record 90: XQRS finds nothing; the model finds every beat at another landmark.
    sig, rr = load_lead(CFG, 90), reference_peaks_s(CFG, 90)
    xq = xqrs_peaks_s(sig, CFG)
    render_overlay(sig, score_case(xq, rr, WIN, TOL), rr, CFG, IMG / "paced_xqrs.png", "XQRS on the raw signal", xq)
    overlay(90, dev_preds(90), "Claude Opus 5.5 with zoom", "paced_opus.png")

    # Held-out failure examples: the first non-paced case of each kind, in a fixed model order.
    rec_of = refs["record"].to_dict()
    order = list(NAMES)
    picks = {}
    for kind in ("missed", "added"):
        cand = outcomes[outcomes.outcome == kind].copy()
        cand = cand[~cand.case_id.map(lambda c: "paced" in subgroups(CFG, rec_of[c]))]
        if len(cand):
            cand["rank"] = cand.model.map(order.index)
            c = cand.sort_values(["rank", "condition", "case_id"]).iloc[0]
            row = att[(att.case_id == c.case_id) & (att.model == c.model) & (att.condition == c.condition)].iloc[0]
            overlay(rec_of[c.case_id], row.parsed["peaks_s"], f"{name(c.model)}, {COND[c.condition]}", f"fail_{kind}.png")
            picks[kind] = (c.model, c.condition)
    inv = att[~att.parsed.map(lambda p: p["valid"]) & att.raw_response.notna()]
    inv = inv[inv.raw_response.str.len() > 40]
    if len(inv):
        r = inv.assign(rank=inv.model.map(order.index)).sort_values(["rank", "case_id"]).iloc[0]
        text = " ".join(r.raw_response.split())
        picks["prose"] = (r.model, r.condition, text[:420] + ("..." if len(text) > 420 else ""))
    return picks


# ── SVG charts ────────────────────────────────────────────────────────────────

def svg_open(w, h, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">'


def spread_labels(labels, gap, bottom=None):
    """Nudge right-hand labels apart so they never overlap or run below `bottom`. labels: [(y, ...)]."""
    placed = []
    for item in sorted(labels, key=lambda t: t[0]):
        y = max(item[0], placed[-1][0] + gap) if placed else item[0]
        placed.append([y, *item[1:]])
    if bottom is not None:
        limit = bottom
        for item in reversed(placed):
            item[0] = min(item[0], limit)
            limit = item[0] - gap
    return [tuple(i) for i in placed]


def tolerance_chart(arms: pd.DataFrame, xqrs: dict) -> str:
    """Pass rate as the allowed error tightens, picture-only input (every model has it)."""
    a = arms[arms.condition == "image_only"]
    W, H, l, r_, t, b = 760, 420, 64, 540, 40, 370
    tols = [150, 50, 25]
    X = lambda i: l + i * (r_ - l) / (len(tols) - 1)  # noqa: E731
    lo = math.floor(min(a[["pass150", "pass50", "pass25"]].min().min(), xqrs["pass25"]) * 10) / 10
    Y = lambda v: b - (v - lo) / (1 - lo) * (b - t)  # noqa: E731
    out = [svg_open(W, H, "Pass rate as the allowed error tightens")]
    for v in np.arange(lo, 1.0001, 0.1):
        out.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{r_}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{l - 8}" y="{Y(v) + 5:.1f}" text-anchor="end" class="muted">{v:.0%}</text>')
    for i, tl in enumerate(tols):
        out.append(f'<text x="{X(i):.1f}" y="{b + 24}" text-anchor="middle" class="muted">&#177;{tl} ms</text>')
    out.append(f'<text x="{(l + r_) / 2:.0f}" y="{H - 2}" text-anchor="middle" class="muted">allowed timing error</text>')
    out.append(f'<text x="{l - 50}" y="{t - 24}" class="muted">records with every beat right</text>')
    pts = [(X(i), Y(xqrs[f"pass{tl}"])) for i, tl in enumerate(tols)]
    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '" fill="none" stroke="#1c1a1f" stroke-dasharray="4 3"/>')
    labels = [(pts[-1][1], f"XQRS {pct(xqrs['pass25'])}", "#6c6572")]
    for _, r in a.iterrows():
        col = COLORS.get(r.model, "#444")
        pts = [(X(i), Y(r[f"pass{tl}"])) for i, tl in enumerate(tols)]
        out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{col}" stroke-width="2"/>')
        out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{col}"/>' for x, y in pts]
        labels.append((pts[-1][1], f"{r['name']} {pct(r.pass25)}", col))
    for y, text, col in spread_labels(labels, 17, bottom=b):
        out.append(f'<text x="{r_ + 12}" y="{y + 5:.1f}" class="small" style="fill:{col}">{esc(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def bar_chart(arms: pd.DataFrame, xqrs: dict) -> str:
    """Horizontal bars: pass rate at ±25 ms with 95% intervals."""
    a = arms.sort_values(["pass25", "p95"], ascending=[False, True]).reset_index(drop=True)
    row_h, left, right, top = 25, 250, 715, 18
    h = top + row_h * len(a) + 44
    x = lambda v: left + v * (right - left)  # noqa: E731
    out = [svg_open(760, h, "Pass rate at 25 ms by model")]
    for v in np.arange(0, 1.01, 0.2):
        out.append(f'<line x1="{x(v):.1f}" y1="{top - 6}" x2="{x(v):.1f}" y2="{h - 34}" class="grid"/>')
        out.append(f'<text x="{x(v):.1f}" y="{h - 14}" text-anchor="middle" class="muted">{v:.0%}</text>')
    for i, r in a.iterrows():
        y = top + i * row_h
        col = COLORS.get(r.model, "#444")
        op = "1" if r.condition == "image_zoom" else "0.5"
        out.append(f'<text x="{left - 10}" y="{y + 15}" text-anchor="end">{esc(r["name"])} <tspan class="muted">{COND_SHORT[r.condition]}</tspan></text>')
        out.append(f'<rect x="{left}" y="{y + 4}" width="{max(0.5, x(r.pass25) - left):.1f}" height="15" fill="{col}" fill-opacity="{op}"/>')
        out.append(f'<line x1="{x(r.pass25_lo):.1f}" y1="{y + 11.5}" x2="{x(r.pass25_hi):.1f}" y2="{y + 11.5}" stroke="#1c1a1f" stroke-width="1.2"/>')
        out.append(f'<text x="{x(max(r.pass25, r.pass25_hi)) + 6:.1f}" y="{y + 16}" class="small">{pct(r.pass25)}</text>')
    xv = x(xqrs["pass25"])
    out.append(f'<line x1="{xv:.1f}" y1="{top - 6}" x2="{xv:.1f}" y2="{h - 34}" stroke="#1c1a1f" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{xv - 4:.1f}" y="{top - 9}" text-anchor="end" class="muted small">XQRS {pct(xqrs["pass25"])}</text>')
    out.append("</svg>")
    return "".join(out)


def scatter_cost(arms: pd.DataFrame, xqrs: dict) -> str:
    """Cost per record against pass rate at 25 ms: one dot per model, at its better input."""
    best = arms.sort_values(["pass25", "cost"], ascending=[False, True]).groupby("model").head(1)
    W, H, l, r_, t, b = 760, 420, 64, 600, 30, 360
    lo, hi = 1e-3, 1e-1
    X = lambda c: l + (math.log10(c) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (r_ - l)  # noqa: E731
    Y = lambda v: b - v * (b - t)  # noqa: E731
    out = [svg_open(W, H, "Price against accuracy")]
    for v in np.arange(0, 1.01, 0.2):
        out.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{r_}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{l - 8}" y="{Y(v) + 5:.1f}" text-anchor="end" class="muted">{v:.0%}</text>')
    for c, text in ((0.001, "$0.001"), (0.01, "$0.01"), (0.1, "$0.10")):
        out.append(f'<text x="{X(c):.1f}" y="{b + 22}" text-anchor="middle" class="muted">{text}</text>')
    out.append(f'<line x1="{l}" y1="{b}" x2="{r_}" y2="{b}" class="axis"/>')
    out.append(f'<text x="{(l + r_) / 2:.0f}" y="{H - 4}" text-anchor="middle" class="muted">cost per record</text>')
    out.append(f'<text x="{l - 50}" y="{t - 14}" class="muted">records with every beat within 25 ms</text>')
    yx = Y(xqrs["pass25"])
    out.append(f'<line x1="{l}" y1="{yx:.1f}" x2="{r_}" y2="{yx:.1f}" stroke="#1c1a1f" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{l + 6}" y="{yx + 18:.1f}" class="muted small">XQRS, free: {pct(xqrs["pass25"])}</text>')
    for _, r in best.iterrows():
        col = COLORS.get(r.model, "#444")
        x, y = X(min(max(r.cost, lo), hi)), Y(r.pass25)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{col}"/>')
        left = r.model in ("gpt-6.1-sol", "deepseek-v4.1-flash", "kimi-k3")
        dy = {"gpt-6.1-sol": 16, "kimi-k3": -10}.get(r.model, 5)
        label = f"{r['name']}{' + zoom' if r.condition == 'image_zoom' else ''}, {pct(r.pass25)}"
        out.append(f'<text x="{x + (-11 if left else 11):.1f}" y="{y + dy:.1f}" '
                   f'text-anchor="{"end" if left else "start"}" class="small" style="fill:{col}">{esc(label)}</text>')
    out.append("</svg>")
    return "".join(out)


def zoom_chart(arms: pd.DataFrame, diffs: pd.DataFrame) -> str:
    """Per model: zoom calls per record (left) and the paired change in p95 error with zoom (right)."""
    rows = []
    for _, d in diffs.iterrows():
        z = arms[(arms.model == d.model) & (arms.condition == "image_zoom")].iloc[0]
        rows.append((d.model, z.zooms, d.p95_abs_err_ms, d.p95_abs_err_ms_lo, d.p95_abs_err_ms_hi))
    rows.sort(key=lambda r: -r[1])
    row_h, top = 28, 34
    H = top + row_h * len(rows) + 46
    lname, l1, r1, l2, r2 = 190, 200, 380, 430, 740
    zmax = max(2, math.ceil(max(r[1] for r in rows)))
    vals = [v for r in rows for v in r[2:]]
    tick = next(s for s in (5, 10, 25, 50, 100) if (max(vals + [0]) - min(vals + [0])) / s <= 6)
    dlo, dhi = math.floor(min(vals + [0]) / tick) * tick, max(tick, math.ceil(max(vals) / tick) * tick)
    X1 = lambda v: l1 + v / zmax * (r1 - l1)  # noqa: E731
    X2 = lambda v: l2 + (v - dlo) / (dhi - dlo) * (r2 - l2)  # noqa: E731
    out = [svg_open(760, H, "Zoom use and its effect on timing error")]
    out.append(f'<text x="{(l1 + r1) / 2}" y="16" text-anchor="middle" class="muted">zoom calls per record</text>')
    out.append(f'<text x="{(l2 + r2) / 2}" y="16" text-anchor="middle" class="muted">change in p95 error with zoom (ms)</text>')
    for v in range(0, zmax + 1):
        out.append(f'<line x1="{X1(v):.1f}" y1="{top - 6}" x2="{X1(v):.1f}" y2="{H - 34}" class="grid"/>')
        out.append(f'<text x="{X1(v):.1f}" y="{H - 14}" text-anchor="middle" class="muted">{v}</text>')
    for v in range(dlo, dhi + 1, tick):
        out.append(f'<line x1="{X2(v):.1f}" y1="{top - 6}" x2="{X2(v):.1f}" y2="{H - 34}" class="{"axis" if v == 0 else "grid"}"/>')
        out.append(f'<text x="{X2(v):.1f}" y="{H - 14}" text-anchor="middle" class="muted">{v if v == 0 else f"{v:+d}"}</text>')
    for i, (m, zc, d, lo, hi) in enumerate(rows):
        y = top + i * row_h + 12
        col = COLORS.get(m, "#444")
        out.append(f'<text x="{lname}" y="{y + 5}" text-anchor="end">{esc(name(m))}</text>')
        out.append(f'<rect x="{l1}" y="{y - 7}" width="{max(0.5, X1(zc) - l1):.1f}" height="14" fill="{col}"/>')
        out.append(f'<text x="{X1(zc) + 6:.1f}" y="{y + 5}" class="small">{zc:.2f}</text>')
        out.append(f'<line x1="{X2(lo):.1f}" y1="{y}" x2="{X2(hi):.1f}" y2="{y}" stroke="#1c1a1f" stroke-width="1.2"/>')
        out.append(f'<circle cx="{X2(d):.1f}" cy="{y}" r="5.5" fill="{col}"/>')
    out.append("</svg>")
    return "".join(out)


# ── tables ────────────────────────────────────────────────────────────────────

def leaderboard(arms: pd.DataFrame, xqrs: dict) -> str:
    a = arms.sort_values(["pass25", "p95"], ascending=[False, True])
    rows = []
    for _, r in a.iterrows():
        kind = "open-weight" if r.model in OPEN else "closed"
        rows.append(
            f'<tr><td><span class="model"><i class="dot" style="background:{COLORS.get(r.model, "#444")}"></i>{esc(r["name"])}</span></td>'
            f'<td>{COND_SHORT[r.condition]}</td><td class="muted-cell">{kind}</td>'
            f'<td class="num">{pct(r.pass25)}</td><td class="num">{r.p95:.0f} ms</td><td class="num">{r.median_ms:.0f} ms</td>'
            f'<td class="num">{r.fn}</td><td class="num">{r.fp}</td><td class="num">{pct(r.invalid)}</td>'
            f'<td class="num">${r.cost:.3f}</td><td class="num">{r.latency:.0f} s</td></tr>')
    rows.append(f'<tr class="ref"><td>XQRS detector</td><td>signal</td><td class="muted-cell">classic</td>'
                f'<td class="num">{pct(xqrs["pass25"])}</td><td class="num">{xqrs["p95"]:.0f} ms</td><td class="num">2 ms</td>'
                f'<td class="num">48</td><td class="num">0</td><td class="num">n/a</td><td class="num">$0</td><td class="num">&lt;0.1 s</td></tr>')
    return ('<div class="table-wrap wide"><table><caption>Held-out results: 100 records, 462 expert-marked beats, '
            'one attempt per record, sorted by pass rate at &#177;25 ms.</caption><thead><tr><th>Model</th><th>Input</th><th>Type</th>'
            '<th class="num">Pass &#177;25 ms</th><th class="num">p95 error</th><th class="num">Median error</th>'
            '<th class="num">Missed</th><th class="num">Added</th><th class="num">No answer</th>'
            '<th class="num">Cost/record</th><th class="num">Time</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def failure_table(outcomes: pd.DataFrame, arms: pd.DataFrame) -> str:
    counts = outcomes.groupby(["model", "condition", "outcome"]).size().unstack(fill_value=0)
    order = arms.sort_values(["pass25", "p95"], ascending=[False, True])[["model", "condition"]].itertuples(index=False)
    head = "".join(f'<th class="num">{esc(label)}</th>' for _, label in FAILURES)
    rows = []
    for m, c in order:
        cnt = counts.loc[(m, c)] if (m, c) in counts.index else pd.Series(dtype=int)
        cells = "".join(f'<td class="num">{int(cnt.get(k, 0))}</td>' for k, _ in FAILURES)
        rows.append(f'<tr><td>{esc(name(m))}</td><td>{COND_SHORT[c]}</td>{cells}</tr>')
    return ('<div class="table-wrap wide"><table><caption>What happened on each of the 100 held-out records. '
            'Each record is counted once, under the first column that applies, reading left to right.</caption>'
            f'<thead><tr><th>Model</th><th>Input</th>{head}</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def ci_table(arms: pd.DataFrame) -> str:
    rows = []
    for _, r in arms.sort_values(["model", "condition"]).iterrows():
        rows.append(f'<tr><td>{esc(r["name"])}</td><td>{COND_SHORT[r.condition]}</td>'
                    f'<td class="num">{r.f1_150:.3f} <span class="ci">[{r.f1_150_lo:.3f}, {r.f1_150_hi:.3f}]</span></td>'
                    f'<td class="num">{pct(r.pass150)} <span class="ci">[{pct(r.pass150_lo)}, {pct(r.pass150_hi)}]</span></td>'
                    f'<td class="num">{pct(r.pass50)} <span class="ci">[{pct(r.pass50_lo)}, {pct(r.pass50_hi)}]</span></td>'
                    f'<td class="num">{pct(r.pass25)} <span class="ci">[{pct(r.pass25_lo)}, {pct(r.pass25_hi)}]</span></td>'
                    f'<td class="num">{r.p95:.0f} <span class="ci">[{r.p95_lo:.0f}, {r.p95_hi:.0f}]</span></td></tr>')
    return ('<div class="table-wrap wide"><table><caption>Full held-out scores with 95% case-level bootstrap intervals '
            '(10,000 resamples).</caption><thead><tr><th>Model</th><th>Input</th><th class="num">F1 &#177;150 ms</th>'
            '<th class="num">Pass &#177;150 ms</th><th class="num">Pass &#177;50 ms</th><th class="num">Pass &#177;25 ms</th>'
            '<th class="num">p95 error, ms</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def subgroup_table(subs: pd.DataFrame) -> str:
    rows = []
    for g, label in (("paced", "Pacemaker rhythm"), ("atrial_fibrillation", "Atrial fibrillation"), ("neither", "All other records")):
        s25 = subs[(subs.tolerance_ms == 25) & (subs.subgroup == g)]
        s150 = subs[(subs.tolerance_ms == 150) & (subs.subgroup == g)]
        mod25, mod150 = s25[s25.arm.str.contains(r" \| ")], s150[s150.arm.str.contains(r" \| ")]
        x = s25[s25.arm == "XQRS (raw signal)"].iloc[0]
        rows.append(f'<tr><td>{label}</td><td class="num">{int(s25.n_cases.iloc[0])}</td>'
                    f'<td class="num">{pct(mod25.case_pass_rate.min())} to {pct(mod25.case_pass_rate.max())}</td>'
                    f'<td class="num">{mod150.p95_abs_err_ms.min():.0f} to {mod150.p95_abs_err_ms.max():.0f} ms</td>'
                    f'<td class="num">{pct(x.case_pass_rate)}</td></tr>')
    return ('<div class="table-wrap"><table><caption>Results by rhythm group (held-out). Ranges span every model '
            'and input.</caption><thead><tr><th>Group</th><th class="num">Records</th>'
            '<th class="num">Model pass &#177;25 ms</th><th class="num">Model p95 error</th><th class="num">XQRS pass &#177;25 ms</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


# ── page ──────────────────────────────────────────────────────────────────────

def build():
    arms, xqrs, diffs, subs = load_tables()
    refs = load_references().set_index("case_id")
    att = load_all_attempts()
    outcomes = classify(att, refs)
    picks = make_images(att, outcomes, refs)
    px_ms = 10_000 / (AXES_RECT["overview"][2] * CFG["render"]["overview"]["width_px"])

    ranked = arms.sort_values(["pass25", "p95"], ascending=[False, True])
    good = arms[arms.pass25 >= 0.9]
    best_value = good.loc[(good.pass25 / good.cost.clip(lower=1e-4)).idxmax()] if len(good) else ranked.iloc[0]
    open_ranked = ranked[ranked.model.isin(OPEN)]
    n_clean = int((arms.fn == 0).sum())
    n_models = arms.model.nunique()
    opus = diffs[diffs.model == "claude-opus-5-5"]
    paced150 = subs[(subs.tolerance_ms == 150) & (subs.subgroup == "paced") & subs.arm.str.contains(r" \| ")]
    paced_err = f"{paced150.p95_abs_err_ms.min():.0f} to {paced150.p95_abs_err_ms.max():.0f} ms"
    css = (ROOT / "scripts" / "site.css").read_text()

    open_line = ""
    if len(open_ranked):
        o = open_ranked.iloc[0]
        open_line = f"Among the open-weight models, {esc(o['name'])} ({COND[o.condition]}) did best, passing {pct(o.pass25)}."
    zoom_line = ""
    if len(opus):
        d = opus.iloc[0]
        zoom_line = (f"Claude Opus 5.5 zoomed on most records, and its 95th-percentile error fell by "
                     f"{-d.p95_abs_err_ms:.0f} ms (95% interval {-d.p95_abs_err_ms_hi:.0f} to {-d.p95_abs_err_ms_lo:.0f} ms).")

    def arm(model, cond):
        return arms[(arms.model == model) & (arms.condition == cond)].iloc[0]
    grok_o, grok_z = arm("grok-4.7", "image_only"), arm("grok-4.7", "image_zoom")
    ds_o, ds_z = arm("deepseek-v4.1-flash", "image_only"), arm("deepseek-v4.1-flash", "image_zoom")
    muse_o, muse_z = arm("muse-spark-open", "image_only"), arm("muse-spark-open", "image_zoom")
    kimi = arm("kimi-k3", "image_only")
    luna = arm("gpt-5.6-luna", "image_only")
    sol_z, opus_z = arm("gpt-6.1-sol", "image_zoom"), arm("claude-opus-5-5", "image_zoom")
    priciest = arms.loc[arms.cost.idxmax()]

    gallery = []
    for kind, label in (("missed", "(a) Missed beat"), ("added", "(b) Added beat")):
        if kind in picks:
            m, c = picks[kind]
            gallery.append(f'<div><p class="panel-label">{label}: {esc(name(m))}, {COND[c]}</p>'
                           f'<img src="img/fail_{kind}.png" alt="{label[4:]}"/></div>')
    prose = ""
    if "prose" in picks:
        m, c, text = picks["prose"]
        prose = (f'<figure class="figure"><blockquote class="raw">{esc(text)}</blockquote>'
                 f'<figcaption>A reply from {esc(name(m))} ({COND[c]}) that described the beats in words instead of '
                 f'returning the list of times. The benchmark counts this as no answer, so every beat on that record counts as missed.</figcaption></figure>')

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ECG Landmark Bench</title>
<style>{css}</style>
</head>
<body>
<header class="topbar"><nav class="topbar-inner"><span>ecg-landmark-bench</span>
<div class="links"><a href="#results">results</a><a href="#failures">failures</a><a href="#appendix">appendix</a></div></nav></header>
<main>
<header class="masthead flow">
  <h1>ECG Landmark Bench</h1>
  <div class="byline"><span class="author">Colin Budries</span><span class="date">October 2026</span></div>
  <figure class="figure wide hero"><img src="img/hero.png" alt="A ten-second ECG strip. Gray triangles show where cardiologists marked each beat; green ticks show where a model marked them." />
  <figcaption>The task in one picture. Gray triangles are the cardiologists' marks; green ticks are a model's answers. Only beats in the shaded middle section are scored.</figcaption></figure>
</header>

<article class="post">
<section id="summary" class="flow">
  <p>I had free credits with a service that gives access to many AI models, and I wanted to use them to see how leading closed and open-weight models handle a task I care about. Heart health matters a great deal to me. ECGs once needed a clinic; now they are built into wearables such as Whoop, and before long many people will have a daily or even continuous ECG. If people are going to ask AI models to read those recordings, it is worth knowing whether an ordinary model can do the most basic part of the job. This does not test wearable ECGs directly. It tests a simpler first step: whether a general-purpose model can precisely locate beats in a clean ECG image.</p>
  <div class="keybox">
    <p class="keytitle">In brief</p>
    <ul>
      <li>We asked {n_models} AI models to mark every heartbeat on pictures of real ECG recordings, and compared their marks with those of expert cardiologists.</li>
      <li><strong>How close is close enough.</strong> Every mark is a moment in the recording. A model's mark counts as correct if it falls within a set time of the cardiologists' mark, the allowed timing error. At 150 milliseconds (ms) a model only has to point at the right beat. At 25 ms it must place the mark very near the cardiologists' mark: within about a quarter of the width of the spike, three fifths of one small square on ECG paper, or five to six pixels in the image.</li>
      <li>On these clean ten-second strips, finding the beats was not the main difficulty for the strongest models; placing them precisely was. On the 100 test records, {n_clean} of the {len(arms)} model and input combinations missed no beat at all. XQRS, a basic open-source beat detector that reads the raw signal rather than a picture, missed 48, including every beat on the pacemaker recordings.</li>
      <li>The models differ in precision. GPT-6.1 Sol and Claude Opus 5.5, both with zoom, placed every beat within 25 ms on {pct(sol_z.pass25) + ' of records each' if pct(sol_z.pass25) == pct(opus_z.pass25) else pct(sol_z.pass25) + ' and ' + pct(opus_z.pass25) + ' of records'}. For context, one horizontal pixel of the image is about {px_ms:.1f} ms, so the best models' errors are on roughly the scale of a pixel.</li>
      <li>The largest changes with a zoom tool available came for the least precise models: Grok 4.7 went from {pct(grok_o.pass25)} to {pct(grok_z.pass25)} of records passed at 25 ms, and DeepSeek V4.1 Flash from {pct(ds_o.pass25)} to {pct(ds_z.pass25)}.</li>
      <li>Some models fail simply by not answering in the form asked for. With zoom offered, Muse Spark Open gave no usable answer on {pct(muse_z.invalid)} of records.</li>
      <li>On clean ECG images, the strongest general-purpose models locate beats to within roughly a pixel. Their remaining failures come less from missing heartbeats than from where exactly a mark belongs, image resolution, tool use and answers in the wrong format. Whether this carries over to noisy wearable ECGs, other parts of the waveform or clinical interpretation is still an open question.</li>
    </ul>
  </div>
</section>

<section id="why" class="flow">
  <h2><span class="num">1</span> Why This Matters</h2>
  <p>An electrocardiogram (ECG) records the heart's electrical activity as a wavy line. Locating each beat is one of the foundational steps in reading it. It gives the heart rate and the time between beats, and it is the anchor for finding the start and end of the other waves, which doctors measure to check for rhythm problems and drug side effects. Software does this well when it has the raw recording. In practice, many ECGs are stored and shared only as images or paper printouts, and general-purpose AI models are increasingly asked to read them.</p>
  <p>This benchmark tests that first, checkable step: can a model look at the picture and say where each beat is? It does not ask whether the heart is healthy, and nothing here is a diagnosis.</p>
</section>

<section id="task" class="flow">
  <h2><span class="num">2</span> The Task</h2>
  <p>The recordings come from the Lobachevsky University ECG Database (LUDB, version 1.0.1, published on PhysioNet): 200 people, one ten-second recording each. Two cardiologists marked every wave by eye and agreed on a single set of labels. We use one view of the heart, {term("Lead II")}, and the cardiologists' annotated peak of each {term("QRS")} complex, the main spike of each beat.</p>
  <p>Each model receives one picture of the full recording on standard ECG grid paper and must reply with the times, in seconds, of every beat between 3 and 7 seconds. The experts left the first and last second or two of every recording unlabeled, so only this middle window is scored; it is shaded on the picture. In the second version of the task the model may also call a zoom tool, up to six times, to see any stretch in more detail.</p>
  <figure class="figure"><div class="cells"><div><p class="panel-label">(a) What every model receives</p><img src="img/input.png" alt="The overview image"/></div><div class="zoom-panel"><p class="panel-label">(b) One zoom request: 4.4 to 5.4 seconds</p><img src="img/zoom.png" alt="A zoomed view"/></div></div>
  <figcaption>The pictures a model sees for one development record. They carry no record number, labels or other clues.</figcaption></figure>
  <p><strong>Scoring.</strong> A model's mark counts as a beat found if it lies within a set time, the {term("tolerance")}, of an expert mark, and each expert mark can be claimed only once. A record passes if every beat is found and no extra beats are added. We score at three tolerances, 150, 50 and 25 milliseconds; the strictest is the most informative.</p>
  <p><strong>A fair test.</strong> The 200 records were split at random into 100 for development and 100 for the final test. All rules were settled on the development records and then frozen. Only after that were the {term("held-out")} records run, once. New models are tested under exactly the same rules.</p>
</section>

<section id="results" class="flow">
  <h2><span class="num">3</span> Results</h2>
  <p>At a loose tolerance almost every model looks perfect. The differences appear as the allowed error shrinks.</p>
  <figure class="figure">{tolerance_chart(arms, xqrs)}
  <figcaption>Share of the 100 test records on which a model got every beat right, picture alone. A record counts only if all of its four or five scored beats fall within the allowed timing error and no beat is added; one bad beat fails the whole record. A model can therefore place most beats well and still score low. The dashed line is XQRS, a basic open-source beat detector that reads the raw signal rather than a picture, run with its default settings. It is included as a reference point, not as a state-of-the-art baseline; commercial ECG software is likely to do better.</figcaption></figure>
  <figure class="figure">{bar_chart(arms, xqrs)}
  <figcaption>Share of the 100 test records with every beat within 25 ms and none added, for every model and input. Solid bars used the zoom tool; pale bars had the picture alone. Black lines show 95% {term("bootstrap")} intervals, reflecting the sampling uncertainty from 100 test records.</figcaption></figure>
  {leaderboard(arms, xqrs)}
  <p><strong>Found, but late.</strong> With the picture alone, Grok 4.7 and Kimi K3 found most beats but placed their marks after the expert's, by {grok_o.mean_signed:.0f} and {kimi.mean_signed:.0f} ms on average. Because the error is a shift in one direction rather than random scatter, nearly every record fails at 25 ms even though the beats themselves were found.</p>
  <p><strong>Cost and precision.</strong> Price is a poor guide to precision. {esc(priciest['name'])} cost the most per record (${priciest.cost:.3f}) and passed {pct(priciest.pass25)} at 25 ms; GPT-5.6 Luna cost ${luna.cost:.3f} and passed {pct(luna.pass25)}. The chart below sets each model's accuracy against its cost.</p>
  <figure class="figure">{scatter_cost(arms, xqrs)}
  <figcaption>Price against accuracy. Each dot is one model in its better version, with or without the zoom tool. Higher is more accurate; further left is cheaper. The dashed line is XQRS, the free detector that reads the raw signal.</figcaption></figure>
  <p><strong>Which model to use.</strong> For precision, GPT-6.1 Sol and Claude Opus 5.5 were effectively tied: both passed {pct(sol_z.pass25)} of records at 25 ms with zoom. GPT-6.1 Sol had the lower 95th-percentile timing error, {sol_z.p95:.0f} ms against {opus_z.p95:.0f} ms, but this experiment is too small to treat that difference as meaningful. For value, {esc(best_value['name'])} ({COND[best_value.condition]}) passed {pct(best_value.pass25)} of records at ${best_value.cost:.3f} each. {open_line}</p>
</section>

<section id="failures" class="flow">
  <h2><span class="num">4</span> What Goes Wrong</h2>
  <p>Scores alone hide how models fail. The table counts what happened on each test record.</p>
  {failure_table(outcomes, arms)}
  <figure class="figure"><div class="cells">{"".join(gallery)}</div>
  <figcaption>Real failures from the test records. Gray triangles: expert marks; green ticks: correct answers; red crosses: beats the model missed; orange circles: beats the model added.</figcaption></figure>
  {prose}
  <p><strong>Pacemaker beats expose a problem with the answer key as well as with the models.</strong> A beat driven by a pacemaker often has no upward spike. On such beats the cardiologists placed their mark partway down the slope, while every model chose either the bottom of the beat or its start, {paced_err} away. The models found every one of these beats; the open-source detector found none. We have kept the experts' labels and report pacemaker records separately.</p>
  <figure class="figure"><div class="cells"><div><p class="panel-label">(a) XQRS on a pacemaker recording: no beats found</p><img src="img/paced_xqrs.png" alt="XQRS on a paced record"/></div><div><p class="panel-label">(b) Claude Opus 5.5 on the same recording</p><img src="img/paced_opus.png" alt="Opus on a paced record"/></div></div>
  <figcaption>A pacemaker recording from the development set. The model finds every beat but marks a different point on each beat from the one the experts chose.</figcaption></figure>
</section>

<section id="zoom" class="flow">
  <h2><span class="num">5</span> What Changed When Zoom Was Available?</h2>
  <p>It depends on the model. The largest observed improvements came for the two models that were least precise with the picture alone. Grok 4.7's pass rate at 25 ms rose from {pct(grok_o.pass25)} to {pct(grok_z.pass25)}, and DeepSeek V4.1 Flash's from {pct(ds_o.pass25)} to {pct(ds_z.pass25)}; both zoomed five or six times on a typical record. Models that were already precise gained a few milliseconds. {zoom_line} Claude Sonnet 5.5 almost never zoomed, so for it the zoom version of the task amounts to the same picture with a longer prompt, and the change cannot be told apart from zero. Each record was run only once in each version, so these differences include the ordinary variation between runs of the same model as well as any effect of zoom.</p>
  <p>Muse Spark Open shows both sides. With zoom offered, its pass rate at 25 ms rose from {pct(muse_o.pass25)} to {pct(muse_z.pass25)}, and every answer it gave was within 25 ms. But on {pct(muse_z.invalid)} of records it never returned a usable list of times. At the loose 150 ms tolerance, where it had passed {pct(muse_o.pass150)} with the picture alone, those missing answers leave it worse off.</p>
  <figure class="figure">{zoom_chart(arms, diffs)}
  <figcaption>Left: how often each model used the zoom tool. Right: the change in 95th-percentile timing error when zoom was offered, with 95% intervals; a negative value means more precise. Kimi K3 is absent because the service we used to reach it offers no tool support for it.</figcaption></figure>
</section>

<section id="appendix" class="flow">
  <h2><span class="num">A</span> Technical Appendix</h2>
  {ci_table(arms)}
  {subgroup_table(subs)}
  <p><strong>Matching.</strong> For each record, predictions and expert marks are paired one to one so as to maximise the number of pairs within the tolerance, then to minimise the total distance (scipy <code>linear_sum_assignment</code>, in whole microseconds). Predictions outside the 3 to 7 s window are not counted as extra beats. Answers that are not valid JSON of the form <code>{{"qrs_peaks_s": [...]}}</code>, or that contain times outside the recording, are scored as no answer.</p>
  <p><strong>Uncertainty.</strong> Intervals are 95% percentile intervals from 10,000 bootstrap resamples of the 100 records. The zoom effect is a paired difference computed on the same resamples.</p>
  <p><strong>Models and settings.</strong> Claude models through the Anthropic API; all others through a commercial service that gives access to many models. Reasoning effort high throughout; at most 10 model turns and 6 zoom calls per record; network errors retried up to three times. The three open-weight models (DeepSeek V4.1 Flash, Muse Spark Open and Kimi K3) have publicly released weights but were run through hosted providers, not locally.</p>
  <p><strong>Limitations.</strong> The test set is small: 100 records with four or five scored beats each, and only three pacemaker records, so the intervals above are wide. Each model answered each record once; when development records were run twice, marks moved by a median of 1.5 to 10 ms, so differences of a few milliseconds between models should not be read as meaningful. LUDB has been public since 2020 and has only 200 records, so this benchmark cannot show that the models are generalizing to unseen ECGs rather than benefiting from some exposure to these recordings during training. The two cardiologists agreed on a single set of marks, so disagreement between experts cannot be measured. Settings differ slightly: DeepSeek ran without the required answer format, which it could not combine with the zoom tool, and Kimi K3 ran without the zoom tool. Clean ten-second strips from one device are easier than photographs of paper ECGs, and nothing here tests clinical use.</p>
  <p><strong>Records.</strong> The protocol was fixed, and its settings recorded by hash, before the test records were run. Every request, reply, zoom call and cost is logged.</p>
  <p><strong>Data.</strong> LUDB 1.0.1, Open Data Commons Attribution License. Kalyakulina et al., IEEE Access, 2020; Goldberger et al., Circulation, 2000.</p>
</section>
</article>
</main>
<footer class="foot">ecg-landmark-bench &middot; October 2026</footer>
</body>
</html>
"""
    # inline the figures so the page is one self-contained file
    page = re.sub(r'src="img/([\w.]+)"', lambda m: 'src="data:image/png;base64,'
                  + base64.b64encode((SITE / "img" / m.group(1)).read_bytes()).decode() + '"', page)
    (SITE / "index.html").write_text(page)
    print(f"wrote {SITE / 'index.html'}")


if __name__ == "__main__":
    build()
