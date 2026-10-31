# Report: VLM QRS-peak localization on LUDB (final, 2026-10-28)

> **Status:** complete for protocol v1 (frozen 2026-10-28, `results/frozen/protocol.json`).
> The held-out results below come from 100 records that were never used to make any
> decision. Development results follow for context.
> This measures where a model places waveform landmarks. It is **not** diagnosis.

## All models (held-out)

The full lineup (eight models, fifteen model and input arms) is reported in
`docs/site/index.html`, built from `results/heldout/runs/summary.csv` and
`results/heldout/final/bootstrap.csv`. The Claude-only sections below are unchanged.

## Held-out results (100 records, 462 scored QRS peaks, 1 attempt per case)

Sources: `results/heldout/runs/` (`attempts.jsonl` per model, `summary.csv`),
`results/heldout/final/bootstrap.csv` and `subgroups.csv`, `results/heldout/baseline/`.
There were 400 model attempts, with no invalid responses, refusals or execution
failures. Model cost was $7.17. Brackets are 95% case-level bootstrap intervals
(10,000 replicates).

| Method | Condition | F1 @150 ms | F1 @25 ms | Pass @25 ms | p95 \|err\| (ms) | Zooms | $/case |
|---|---|---|---|---|---|---|---|
| Opus 5.5 | image only | 1.000 [1.000, 1.000] | 0.989 [0.968, 1.000] | 0.98 [0.95, 1.00] | 10 [8, 12] | 0 | 0.012 |
| Opus 5.5 | image + zoom | 1.000 [1.000, 1.000] | 0.991 [0.973, 1.000] | 0.99 [0.97, 1.00] | 6 [5, 10] | 1.6 | 0.047 |
| Sonnet 5.5 | image only | 0.999 [0.997, 1.000] | 0.962 [0.931, 0.987] | 0.91 [0.85, 0.96] | 18 [14, 38] | 0 | 0.006 |
| Sonnet 5.5 | image + zoom | 0.998 [0.995, 1.000] | 0.961 [0.930, 0.986] | 0.89 [0.83, 0.95] | 20 [13, 32] | 0.01 | 0.007 |
| XQRS (raw signal) | n/a | 0.945 [0.907, 0.976] | 0.943 [0.905, 0.974] | 0.88 [0.81, 0.94] | 6 [4, 8] | n/a | 0 |
| XQRS + correct_peaks | n/a | 0.945 [0.907, 0.976] | 0.938 [0.898, 0.972] | 0.88 [0.81, 0.94] | 4 [4, 8] | n/a | 0 |

p95 |err| is computed over peaks matched at 150 ms. "Pass" means every scored peak in
the record was found, with no extra peaks.

**Zoom effect (paired: image + zoom minus image only, same cases):**

| Model | Δ p95 \|err\| | Δ F1 @25 ms | Δ pass @25 ms |
|---|---|---|---|
| Opus 5.5 | **−4 ms [−5.8, −1.0]** | +0.002 [0.000, 0.007] | +0.01 [0.00, 0.03] |
| Sonnet 5.5 | +2 ms [−6.0, +5.4] | −0.001 [−0.012, 0.011] | −0.02 [−0.08, 0.03] |

**Subgroups** (point estimates; the paced and AF groups are too small for intervals):

| Subgroup | n | XQRS F1 @150 | Models F1 @150 | Models p95 \|err\| |
|---|---|---|---|---|
| Paced | 3 | 0.000 (no detections) | 1.000 (all four arms) | 52–60 ms |
| Atrial fibrillation | 8 | 0.929 | 1.000 (all four arms) | 6–18 ms |
| Neither | 89 | 0.962 | 0.998–1.000 | 6–16 ms |

### Conclusions

1. **Finding QRS peaks in a clean 10-s image is solved for these models at ±150 ms.**
   Opus found all 462 held-out peaks in both conditions. Sonnet found all of them
   too, adding 1–2 extra peaks. The raw-signal XQRS detector missed 48 peaks; it
   found nothing at all in the 3 paced records.
2. **Zoom helps only the model that uses it.** Opus called zoom 1.6 times per case,
   and that tightened its timing (p95 10 → 6 ms; the interval excludes zero).
   Detection did not change. Sonnet almost never zoomed (0.01 calls per case) and
   showed no effect. Its dev-set improvement did not replicate, so it was noise.
3. **Timing precision separates the models.** At ±25 ms Opus passes 98–99% of
   records and Sonnet 89–91%, about the level of XQRS (88%).
4. **Paced beats remain the hard case, and the cause is the reference convention.**
   Every model found every paced beat, but placed it about 50 ms from the expert mark.
   This matches the dev finding (`docs/dataset.md`): on beats with no upward R wave,
   the expert marks sit on the downslope.
5. **These results confirm the development findings.** Dev and held-out results agree
   to within the noise. The one exception is Sonnet's zoom effect, which did not
   replicate.

### Limitations

- **One attempt per case.** The intervals cover case-to-case variation, not
  run-to-run variation. On dev, repeated runs shifted peak times by a median 1.5–10 ms
  (below), which is about the size of the Opus zoom effect.
- **Public data.** LUDB may have been seen in pretraining, by these models or by
  others.
- **One consensus label set.** There is no estimate of expert-to-expert disagreement,
  so the 150 ms tolerance is a matching rule, not a clinical standard.
- **Only about 4–6 scored beats per record.** The paced and AF subgroups are very small.
- **A human spot-check of the three coverage verdicts** (`config/coverage_review.yaml`)
  is still pending; those verdicts were reviewed by the assistant.
- **Two models from one provider.** Other providers and open-weight models can run under
  the same frozen protocol (`docs/models.md`).

---

# Development results (used to settle the protocol)

## What has been measured

### Raw-signal baseline (XQRS, development partition, 100 records, 438 scored references)

| Variant | Precision | Recall | F1 | Case pass rate | Median \|err\| | 95th pct \|err\| |
|---|---|---|---|---|---|---|
| XQRS (default) | 1.000 | 0.920 | 0.958 | 0.90 | 2.0 ms | 27.8 ms |
| XQRS + correct_peaks | 1.000 | 0.920 | 0.958 | 0.90 | 2.0 ms | 6.0 ms |

Source: `results/dev/baseline/summary.csv`, per case in `per_case.csv`.

**Where XQRS fails.** All 35 misses come from 10 records. XQRS returned **no
detections at all inside the window** for 8 of them: records 34, 90 and 104
(ventricular pacing), 95 (AF + pacing), 83 and 109 (atrial fibrillation), 127 and
133 (sinus bradycardia). Its start-up learning phase evidently fails on these
10-second strips. It never produced a false positive. This is the expected
behavior of a classical detector with default settings. It was not tuned, and it
won't be tuned on the test set.

Its timing is tight (median 2 ms), but it reads exact sample values. A model
reading a 2400-px image has ~4 ms per pixel, so a near-perfect image reader would
still show a few ms of error.

### Pipeline correctness

- 57 automated tests pass (`pytest -q`): matching edge cases, response validation,
  answer-leak checks on images, zoom limits, retry/resume/budget logic, and the
  Anthropic tool loop against a fake client.
- One record (4, case `c_5f75ddcf0f`) runs end to end: `results/example/` holds a
  scores table and overlays for a clearly labelled MOCK prediction and for XQRS.

## Full development run (100 dev records, 438 scored references, 2026-10-28)

Source: `results/dev/runs/dev-v1/` (`attempts.jsonl`, `summary.csv`). 400 attempts, no
invalid responses, refusals or execution failures. Cost $7.00.

| Model | Condition | F1 @150 | F1 @50 | F1 @25 | Pass @25 | Median \|err\| | p95 \|err\| | Zooms | $/att | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| opus-5-5 | image only | 1.000 | 0.991 | 0.961 | 0.93 | 2 ms | 16.3 ms | 0 | 0.012 | 2.6 s |
| opus-5-5 | image + zoom | 1.000 | 0.986 | 0.973 | 0.97 | 2 ms | 6.0 ms | 1.6 | 0.044 | 9.2 s |
| sonnet-5-5 | image only | 0.997 | 0.985 | 0.958 | 0.91 | 4 ms | 20.0 ms | 0 | 0.006 | 2.0 s |
| sonnet-5-5 | image + zoom | 1.000 | 0.986 | 0.971 | 0.95 | 4 ms | 14.0 ms | 0.09 | 0.008 | 3.2 s |
| XQRS (raw signal) | n/a | 0.958 | 0.951 | 0.904 | 0.83 | 2 ms | 27.8 ms | n/a | 0 | <0.1 s |

**Findings on development data:**
1. **Detection is near-saturated at 150 ms.** Both models found every scored QRS peak
   in all 100 records, including the 8 records where XQRS found nothing (paced, AF,
   bradycardia). Sonnet without zoom added 3 extra peaks.
2. **Zoom mainly improves timing precision**: p95 error 16.3 → 6.0 ms (Opus) and
   20.0 → 14.0 ms (Sonnet). Sonnet improved despite almost never calling zoom, so part
   of the zoom-condition effect may come from the prompt rather than the tool.
3. **Paced records are where errors concentrate**: median worst error per record is
   23-43 ms there vs 4-8 ms elsewhere. In paced beats without an upward R wave (e.g.
   record 90) the expert mark sits on the downslope, between the onset and the trough.
   Every model and condition marked the trough or onset instead (35-80 ms off). This is
   ambiguity in the reference convention, not a clear model error.
4. **Run-to-run variability** (20 records run twice): case pass/fail agreed 100%
   (Opus) and 90-95% (Sonnet); peak times shifted by a median 1.5-10 ms. Sonnet's
   record-192 T-wave failure in the pilot did not recur. Single-attempt differences of
   a few ms between conditions are within this noise.

## Development pilot (20 dev records, 89 scored references, 1 attempt each, 2026-10-28)

Source: `results/dev/runs/pilot-v1/` (`attempts.jsonl`, `summary.csv`, `overlays/`).

| Model | Condition | P | R | F1 | Pass | Median \|err\| | p95 \|err\| | Zoom calls | $/attempt | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | image only | 1.00 | 1.00 | 1.00 | 20/20 | 2.0 ms | 8.0 ms | 0 | 0.012 | 3.8 s |
| claude-opus-5-5 | image + zoom | 1.00 | 1.00 | 1.00 | 20/20 | 2.0 ms | 4.6 ms | 1.8 | 0.047 | 9.3 s |
| claude-sonnet-5-5 | image only | 1.00 | 1.00 | 1.00 | 20/20 | 4.0 ms | 12.0 ms | 0 | 0.006 | 2.2 s |
| claude-sonnet-5-5 | image + zoom | 0.94 | 0.96 | 0.95 | 18/20 | 4.0 ms | 19.2 ms | 0.1 | 0.008 | 2.9 s |

No invalid responses, refusals or execution failures.

**Failures (both Sonnet, zoom condition, no zoom calls made):**
- Record 192 (noisy baseline): marked the four **T waves** instead of the narrow QRS
  spikes (all 4 extra, all 4 missed). Without the zoom option the same model got it right.
- Record 63: one extra point 100 ms after a correctly marked peak.

**What the pilot taught us (to resolve before freezing):**
1. **Ceiling effect at ±150 ms.** Three of four arms are perfect on detection. At this
   tolerance the benchmark mostly separates models by *timing precision* (p95 error),
   not by finding peaks. Zoom's visible effect for Opus is tighter timing (p95 8.0 →
   4.6 ms).
2. **The pilot has few hard cases.** Only 1 of 20 pilot records (83, AF) is one XQRS
   failed on; paced and AF strips are where difficulty is expected.
3. **Sonnet rarely uses zoom** (0.1 calls/attempt vs Opus 1.8), so for Sonnet the zoom
   condition mostly measures "same image, longer prompt."

## Limitations noted during development

LUDB is public, so held-out records may have been seen in pretraining. There is a
single consensus label set and no expert-variability estimate; the 150 ms
tolerance is our matching rule, not a clinical standard. Only ~4 scored beats per
record. The population is enriched for disease. Results describe waveform
localization only, not diagnosis.
