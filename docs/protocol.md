# Evaluation protocol (v1, frozen 2026-10-28)

Every number below comes from `config/benchmark.yaml` or `config/prompts.yaml`.
`python -m ecgbench.cli freeze` writes their hashes, plus the split, eligibility,
manifest and scorer hashes, to `results/frozen/protocol.json`. After that, held-out
runs refuse to start if anything changed.

## Task

- **Input:** one PNG (2400×600 px) of LUDB Lead II, full 10 s, raw signal in mV
  (no filtering, resampling or detrending). ECG-style grid (0.2 s / 0.5 mV major,
  0.04 s / 0.1 mV minor), time axis labeled every 0.5 s from the start of the
  recording, the scoring window lightly shaded, title "Lead II". Every record
  gets a fixed layout; the y-range is the record's signal range ± 0.3 mV, rounded
  to 0.5 mV.
- **Target:** times of LUDB QRS reference peaks (symbol `N`) inside 3.0–7.0 s.
- **Output:** `{"qrs_peaks_s": [...]}` in seconds from the start of the recording.
  The prompt does not give the expected number of peaks.

## Partitions (`config/split.csv`)

Seed 42, `numpy.random.default_rng`. 100 dev / 100 test records; one record per
subject, so no subject spans partitions. Pilot: 20 dev records drawn with the same
generator. Scored references: dev 438 (2–7 per record), test 462 (3–8).

## Eligibility (`results/eligibility.csv`), rule set v1

Fixed before any model or baseline result existed:

- **E1 coverage:** the record's first and last Lead II annotation of any kind must
  bracket 2.85–7.15 s (window ± tolerance). Actual worst case over all 200:
  first ≤ 2.716 s, last ≥ 7.35 s.
- **E2 completeness:** a labeled RR gap > 1.6× the record's median RR overlapping
  2.85–7.15 s needs a manual "complete" verdict in `config/coverage_review.yaml`.
  Records 51, 96 and 112 were flagged and reviewed: all are real pauses or irregular
  rhythm with no unlabeled QRS.
- **E3:** at least one reference inside the window.

**Result: 200/200 eligible, no exclusions.** Excluded records, if any appear
later, would stay in their partition and be reported.

Why a fixed window rather than per-record coverage: a per-record window derived from
annotation positions would hint at where labeled beats are. A fixed 3–7 s window is
identical for every record and every method and reveals nothing.

## Response validation (`src/ecgbench/responses.py`)

| Case | Handling |
|---|---|
| JSON object alone, or inside one ```json fence | valid (fence noted) |
| Empty list | valid, zero predictions |
| Duplicate timestamps | valid; kept as separate predictions (extra copies become FPs) |
| Timestamp in [0, 10] s outside the window | valid; not scored (see matching) |
| Any timestamp < 0 or > 10 s | **invalid** (likely milliseconds/samples), never rescaled |
| Malformed JSON, extra/missing keys, non-list, non-numeric/boolean/null, NaN/Infinity | **invalid** |
| Refusal, max_tokens, turn limit, infrastructure failure | **invalid** (no answer) |

Invalid responses are scored as zero predictions, so every scored reference is a
false negative. They stay in all denominators and are also reported as rates.
Answers are never repaired, and never using reference labels.

Anthropic runs use structured outputs (`output_config.format`, JSON schema) so
format errors are rare. Validation still applies in full.

## Matching and metrics (`src/ecgbench/scoring.py`)

1. A prediction–reference pair may match only if |error| ≤ 0.150 s (inclusive,
   computed in integer microseconds).
2. Maximize the number of matches, then minimize total |error|. Solved exactly with
   `scipy.optimize.linear_sum_assignment` on a bonus-weighted cost; deterministic.
3. Predictions are matched against **all** references of the record. Scored
   references are those in [3.0, 7.0] (closed). Unmatched predictions in [3.0, 7.0]
   are FPs. Predictions outside the window are never FPs. A prediction in the window
   that matches a reference just outside it is neither TP nor FP.

Reported per model × condition (pooled over cases): TP, FP, FN, precision, recall,
F1 at the primary 150 ms tolerance, plus precision, recall, F1 and case pass rate at
secondary tolerances of 50 ms and 25 ms (same matching rule, re-run per tolerance); matched-error mean signed / median |error| / 95th-percentile |error| (ms); case
pass rate (valid, FN = 0 and FP = 0); invalid rate; execution-failure rate;
model-failure rate; cost and latency per attempt; zoom calls per attempt.

Empty denominators: precision is NaN with no predictions; F1 is NaN if precision or
recall is NaN.

Uncertainty: **case-level bootstrap** (the recording/subject is the unit), never
resampling individual heartbeats or attempts. 10,000 replicates, seed 20261028,
95% percentile intervals for F1, case pass rate and p95 |error| at every tolerance.
Zoom effect = paired difference (image_zoom − image_only, same model) computed on the
same resampled cases. Implemented in `src/ecgbench/analysis.py`, run by `cli report`.

Pre-specified subgroups (error analysis only, from private header diagnoses):
**paced** (any "… pacing." diagnosis line; 9 records: 6 dev, 3 held-out),
**atrial_fibrillation** ("Rhythm: Atrial fibrillation"; 7 dev, 8 held-out), and
**neither**. Held-out subgroups are small, so they get point estimates only.

## Models are not part of the frozen protocol

The frozen hash covers the task, images, prompts, split, eligibility, data
manifest and scorer. It deliberately excludes the model roster and prices, so a
model added later (another provider, an open-weight model) is evaluated under the
identical protocol. Every attempt record logs the model's exact settings. Model
adapters: Anthropic Messages API, and any OpenAI-Chat-Completions-compatible
endpoint (`docs/models.md`).

## Conditions

- **image_only:** one request: system prompt + image + task text. No tools.
- **image_zoom:** same image and text, plus a `zoom(start_s, end_s)` tool. The tool
  crops a clean 12000×900 render of the same signal (same layout, no annotations),
  keeps the y-axis labels and original time labels (every 0.2 s), and returns it at
  ≤ 2400 px. Span 0.4–10 s; **max 6 calls**; invalid and over-limit calls count
  toward the limit and return an error message. At most 10 model turns.
- A fresh conversation for every case × condition × attempt. Adapters receive only
  the case's clean images; they never see record numbers, references, annotated
  plots, header diagnoses or the dataset directory.

## Failure and retry policy (`src/ecgbench/runner.py`)

- **Designated infrastructure errors** (connection, timeout, HTTP 429, 500/502/503/
  504/529): retried up to 3 times from a fresh conversation, with backoff of 5, 20
  and 60 s. Every try is logged with its error and any partial token usage. If all
  retries fail, the status is `execution_failure`.
- **Other API errors** (e.g. 400): `execution_failure`, no retry.
- **Model failures** (refusal, max_tokens, turn limit) and bad output: no retry.
- 1 attempt per case. Results are checkpointed per attempt; reruns skip finished keys.
- Spending: estimated from reported tokens × `pricing_usd_per_mtok` (including
  partial usage from failed tries), appended to `results/spend_ledger.jsonl`. Before
  each attempt the runner stops if spent + (estimate × 4) would exceed
  `budget.max_spend_usd`.

## Conventional baseline

WFDB `XQRS` (wfdb 4.3.1, default settings) on the numerical Lead II signal, scored
identically. Variant `xqrs_corrected` adds `wfdb.processing.correct_peaks`
(±50 ms, `peak_dir="compare"`). It is a **raw-signal** baseline: it sees exact
sample values, not an image, so it is not an equal-input comparison.

## Change log (development partition only)

| Date | Change | Evidence |
|---|---|---|
| 2026-10-28 | Prompt v1 wording changed before any model run: "main peak, may be upward or downward" → "top of the upward deflection (R wave), even if small; pacemaker spikes are not QRS peaks." | Dev analysis: 865/889 references at local maximum, 0 at local minimum (docs/dataset.md). |
| 2026-10-28 | Zoom renders: dropped the x-axis title and clipped tick fragment from crops. | Visual check of a crop. |
| 2026-10-28 | Added secondary tolerances 50 ms and 25 ms; 150 ms remains primary. | Dev pilot hit a ceiling at 150 ms (3 of 4 model × condition arms perfect); see docs/report.md. |
| 2026-10-28 | Development run extended from 20-record pilot to all 100 dev records (`results/dev/runs/dev-v1/`). | Pilot had only 1 of the hard (paced/AF) records. |
| 2026-10-28 | Frozen. Added before freezing: bootstrap/subgroup analysis rules (above), model roster excluded from the protocol hash, OpenAI-compatible adapter. Held-out: 1 attempt per case (brief default). | Author decision. |
| 2026-10-29 | Roster (not part of the hash): models via an OpenAI-compatible router: gpt-6.1-sol, gpt-5.6-luna, grok-4.7, deepseek-v4.1-flash, muse-spark-open, kimi-k3, all at reasoning_effort high. DeepSeek runs with `structured_output: false`; kimi-k3 image_only only (router lists no tool support). Runner gained `--shard k/n` and atomic appends; no scoring change. | Lineup chosen by the author; DeepSeek returned empty replies with tools plus response_format on dev. |
