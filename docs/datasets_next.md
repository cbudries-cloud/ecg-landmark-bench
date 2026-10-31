# Next datasets: plan

LUDB gives only 200 records and about 4 scored beats each, and frontier models are
near the ceiling at 150 ms. Larger public sets would add statistical power, harder
cases and, later, training data for RL that stays separate from any test set.

Facts below come from primary PhysioNet pages and dataset papers, checked
2026-10-28 by a research pass. **Re-verify each one on download**, as was done for
LUDB in `docs/dataset.md`.

## Candidates

| Dataset | License | Size | Beat mark | Fit |
|---|---|---|---|---|
| **MIT-BIH Arrhythmia** (mitdb 1.0.0) | ODC-By, open | 48 records / 47 subjects, 30 min, 360 Hz, 2 ch; 73 MB zip | Detector output corrected by two cardiologists, placed at the "major local extremum" (usually R) | **Best next set.** About 110k beats over whole records; thousands of 10-s windows |
| **INCART** (incartdb 1.0.0) | ODC-By, open | 75 records / 32 patients, 30 min, 257 Hz, 12 leads; 564 MB zip | Algorithmic "middle of QRS". Beat labels were corrected by hand, **beat locations were not** | **Good, 150 ms only.** Lead setup closest to LUDB |
| CPSC 2019 | Unclear (only a third-party mirror found; it says CC BY 4.0) | 2,000 × 10 s, 500 Hz, 1 lead | R-peak indices; annotator provenance unverified | Conditional: same shape as LUDB, but provenance must be confirmed first |
| European ST-T (edb) | ODC-By, open | 90 records / 79 subjects, 2 h, 250 Hz | Two cardiologists; where the mark sits is unverified | Fallback for CPSC 2019 |
| QT Database | ODC-By | 105 × 15 min | Partial coverage; automated R fiducial; subjects overlap mitdb and edb | Not recommended |
| PTB-XL / PTB-XL+ | CC BY 4.0 | 21,799 × 10 s | No per-beat labels / algorithmic only | At most "silver" labels for training, never for scoring |
| Chapman, Icentia11k, nstdb | various | | No beat labels / non-commercial license, 1.1 TB / mitdb copies | Not recommended (nstdb only as a noise-robustness add-on) |

All PhysioNet sets above are WFDB-readable, so `wfdb` already reads them. Every one
of them is public and widely used, so the pretraining-contamination caveat applies;
for mitdb it is the strongest of any option.

## What changes in the code

The pipeline currently assumes LUDB's shape: one 10-s strip per record, 500 Hz, a
named lead, a `.ii` annotation file, the `N` symbol and a fixed 3–7 s window.
Generalizing it takes the following work, in order.

1. **Dataset registry.** Each dataset entry gives its raw directory, annotation
   extension, lead to use, beat symbols, sampling rate and subject map. LUDB becomes
   the first entry. `data.py` reads from the entry instead of top-level config keys.
2. **Beat symbols, not one symbol.** mitdb uses many beat codes (N, L, R, V, A, /,
   …) plus non-beat annotations (rhythm, noise, comments) that must be filtered.
   Keep the list of beat codes per dataset in config, and test it against the
   dataset's documented label table.
3. **Windowing.** A case becomes (record, start time). Cut 10-s windows from long
   records on a fixed grid, and keep the same image layout and the same 3–7 s scoring
   window inside each window. Eligibility moves to the window level. For example,
   drop windows overlapping noise or unreadable-signal annotations, and windows near
   a record's unannotated stretches.
4. **Split by subject.** mitdb records 201 and 202 come from one person. QTDB and
   nstdb reuse mitdb subjects. Split on subject, never on window.
5. **Sampling rate.** Render at the native rate (360 / 257 Hz). The image layout
   stays in seconds, so prompts don't change. Store the rate in the geometry json.
6. **Prompt wording.** LUDB marks the upward R wave. mitdb marks the major local
   extremum, which can be a downward deflection. A shared prompt must either be
   neutral ("the main peak of each QRS complex") or dataset-specific. Either way that
   is a **new protocol version**, piloted on that dataset's dev split only.
7. **Tolerances per dataset.** INCART positions were never hand-corrected, so only
   the 150 ms result is defensible there.
8. **Sampling cases.** Thousands of windows exist. Sample a fixed, seeded number per
   subject so that long records don't dominate, and oversample paced and arrhythmia
   windows as a labeled subgroup.

## Cost at scale

Measured on LUDB: about $0.07 per case for all four arms (Opus and Sonnet × image /
zoom), $0.044 of it for Opus with zoom. So 1,000 cases × 4 arms ≈ $70, and
10,000 ≈ $700. Use cheaper models and image-only to screen; run zoom arms on a sample.

## Reinforcement learning, later

- Use separate training data (for example mitdb windows from training subjects).
  **Never** use the LUDB held-out records, or any test split.
- The reward is the scorer itself: matches within tolerance, minus false positives,
  plus a timing term at 25 ms. 150 ms is saturated for frontier models.
- Paced beats have ambiguous references (`docs/dataset.md`). Exclude them from the
  reward or treat them separately.
- Open-weight models can be trained directly. Closed models depend on each provider's
  fine-tuning options, which need checking at the time.
