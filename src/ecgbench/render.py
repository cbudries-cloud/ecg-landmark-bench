"""Deterministic ECG rendering.

Model-facing images (clean): waveform, time axis (s from recording start),
voltage axis (mV), lead name, the fixed scoring window. Nothing else: no
reference markers, counts, diagnoses, record numbers, or metadata.

Private images (annotated / overlay) go under data/private or results/ and are
never passed to a model.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from PIL import Image  # noqa: E402

from .data import LeadSignal  # noqa: E402

# Fixed axes position (figure fractions) so pixel <-> time mapping is exact and identical across cases.
AXES_RECT = {"overview": (0.055, 0.13, 0.935, 0.78), "zoom_master": (0.012, 0.08, 0.985, 0.86)}
PNG_META = {"Software": None}  # drop matplotlib's default metadata


def y_limits(sig: LeadSignal, cfg: dict) -> tuple[float, float]:
    r = cfg["render"]
    step = r["major_grid_mv"]
    lo = math.floor((sig.signal.min() - r["y_margin_mv"]) / step) * step
    hi = math.ceil((sig.signal.max() + r["y_margin_mv"]) / step) * step
    return lo, hi


def _draw(sig: LeadSignal, cfg: dict, kind: str):
    r = cfg["render"]
    size = r[kind]
    fig = plt.figure(figsize=(size["width_px"] / size["dpi"], size["height_px"] / size["dpi"]), dpi=size["dpi"])
    ax = fig.add_axes(AXES_RECT[kind])
    duration = len(sig.signal) / sig.fs
    ax.set_xlim(0, duration)
    ax.set_ylim(*y_limits(sig, cfg))
    ax.xaxis.set_minor_locator(MultipleLocator(r["minor_grid_s"]))
    ax.yaxis.set_minor_locator(MultipleLocator(r["minor_grid_mv"]))
    ax.xaxis.set_major_locator(MultipleLocator(r["major_grid_s"]))
    ax.yaxis.set_major_locator(MultipleLocator(r["major_grid_mv"]))
    ax.grid(which="minor", color="#f1d4d4", linewidth=0.5)
    ax.grid(which="major", color="#e09c9c", linewidth=0.9)
    label_every = r["x_tick_label_every_s"] if kind == "overview" else r["major_grid_s"]
    ticks = np.round(np.arange(0, duration + 1e-9, r["major_grid_s"]), 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" if abs(t / label_every - round(t / label_every)) < 1e-6 else "" for t in ticks])
    fs_label = 11 if kind == "overview" else 13
    ax.tick_params(labelsize=fs_label)
    if kind == "overview":  # zoom crops get their axis description in the tool's text reply
        ax.set_xlabel("Time (s) from start of recording", fontsize=fs_label)
    ax.set_ylabel("mV", fontsize=fs_label)
    if r["shade_scoring_window"]:
        a, b = cfg["task"]["scoring_window_s"]
        ax.axvspan(a, b, color="#9ec5ff", alpha=0.18, zorder=0, linewidth=0)
    ax.plot(sig.times_s, sig.signal, color="black", linewidth=r["line_width"], zorder=3)
    ax.text(0.003, 1.02, cfg["task"]["lead_display_name"], transform=ax.transAxes,
            fontsize=fs_label + 3, fontweight="bold", va="bottom")
    if r["shade_scoring_window"] and kind == "overview":
        a, b = cfg["task"]["scoring_window_s"]
        ax.text((a + b) / 2, 1.02, f"scoring window {a:.1f}-{b:.1f} s", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=fs_label, color="#2b5ea8")
    return fig, ax


def _save(fig, path: Path | None = None) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, metadata=PNG_META)
    plt.close(fig)
    data = buf.getvalue()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def render_clean(sig: LeadSignal, cfg: dict, out_dir: Path, case_id: str) -> dict:
    """Write {case_id}.png (overview) and {case_id}_zoom_master.png plus geometry for the zoom tool."""
    fig, _ = _draw(sig, cfg, "overview")
    overview = _save(fig, out_dir / f"{case_id}.png")
    fig, _ = _draw(sig, cfg, "zoom_master")
    _save(fig, out_dir / f"{case_id}_zoom_master.png")
    m = cfg["render"]["zoom_master"]
    left, bottom, width, height = AXES_RECT["zoom_master"]
    geom = {  # geometry only; derived from fixed layout, contains no annotation information
        "duration_s": len(sig.signal) / sig.fs,
        "master_width_px": m["width_px"], "master_height_px": m["height_px"],
        "axes_left_px": left * m["width_px"], "axes_right_px": (left + width) * m["width_px"],
        "axes_bottom_px": (1 - bottom) * m["height_px"],
    }
    (out_dir / f"{case_id}_zoom_master.json").write_text(json.dumps(geom, indent=1))
    return {"overview_bytes": overview}


def zoom_crop(master_png: Path, geom: dict, start_s: float, end_s: float, max_long_edge: int) -> bytes:
    """Crop the clean high-resolution render to [start_s, end_s].

    The left margin (voltage axis labels) is kept and pasted beside the crop so
    the crop still shows mV; the master's own time-axis labels (every 0.2 s)
    remain visible under the crop, so original time coordinates are preserved.
    """
    img = Image.open(master_png).convert("RGB")
    px_per_s = (geom["axes_right_px"] - geom["axes_left_px"]) / geom["duration_s"]
    x0 = int(round(geom["axes_left_px"] + start_s * px_per_s)) - 30   # small pad so edge tick labels aren't cut
    x1 = int(round(geom["axes_left_px"] + end_s * px_per_s)) + 30
    x0, x1 = max(x0, int(geom["axes_left_px"])), min(x1, img.width)
    margin = img.crop((0, 0, int(geom["axes_left_px"]), int(geom["axes_bottom_px"]) + 14))  # y labels only
    body = img.crop((x0, 0, x1, img.height))
    out = Image.new("RGB", (margin.width + body.width, img.height), "white")
    out.paste(margin, (0, 0))
    out.paste(body, (margin.width, 0))
    scale = min(1.0, max_long_edge / max(out.size))
    if scale < 1.0:
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def render_annotated(sig: LeadSignal, refs_s: np.ndarray, cfg: dict, path: Path, title: str) -> None:
    """PRIVATE: overview with expert reference markers. Never shown to a model."""
    fig, ax = _draw(sig, cfg, "overview")
    idx = np.clip(np.round(refs_s * sig.fs).astype(int), 0, len(sig.signal) - 1)
    ax.plot(refs_s, sig.signal[idx], "v", color="#d62728", markersize=9, zorder=5, label="expert QRS reference")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(title, fontsize=11, loc="right")
    _save(fig, path)


def render_overlay(sig: LeadSignal, score, refs_s: np.ndarray, cfg: dict, path: Path, title: str,
                   preds_s=()) -> None:
    """Results figure: matches (green), misses (red X), extras (orange)."""
    fig, ax = _draw(sig, cfg, "overview")
    amp = lambda t: sig.signal[np.clip(np.round(np.asarray(t) * sig.fs).astype(int), 0, len(sig.signal) - 1)]  # noqa: E731
    ymax = ax.get_ylim()[1]
    for p, r in score.matched:
        ax.plot([r, r], [amp(r), ymax], color="#2ca02c", lw=0.8, ls=":")
        ax.plot(p, ymax - 0.05 * (ymax - ax.get_ylim()[0]), "|", color="#2ca02c", ms=14, mew=2)
    unscored = [t for t in preds_s if not (cfg["task"]["scoring_window_s"][0] <= t <= cfg["task"]["scoring_window_s"][1])]
    handles = [
        ax.plot(refs_s, amp(refs_s), "v", color="#555555", ms=7, zorder=5, label="expert reference")[0],
        ax.plot([], [], "|", color="#2ca02c", ms=14, mew=2, label=f"matched prediction ({score.tp})")[0],
    ]
    if score.false_negatives:
        handles.append(ax.plot(score.false_negatives, amp(score.false_negatives), "X", color="#d62728",
                               ms=13, zorder=6, label=f"missed reference ({score.fn})")[0])
    if score.false_positives:
        handles.append(ax.plot(score.false_positives, amp(score.false_positives), "o", mfc="none",
                               mec="#ff7f0e", mew=2.5, ms=15, zorder=6, label=f"extra prediction ({score.fp})")[0])
    if unscored:
        handles.append(ax.plot(unscored, amp(unscored), "o", mfc="none", mec="#999999", mew=1.5, ms=11,
                               zorder=6, label=f"prediction outside window, not scored ({len(unscored)})")[0])
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=11, loc="right")
    _save(fig, path)
