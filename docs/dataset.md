# Dataset provenance: LUDB 1.0.1

| Item | Value |
|---|---|
| Name | Lobachevsky University Electrocardiography Database (LUDB) |
| Version | 1.0.1 |
| Source | https://physionet.org/content/ludb/1.0.1/ (files from https://physionet.org/files/ludb/1.0.1/) |
| Accessed | 2026-10-28 |
| License | Open Data Commons Attribution License v1.0 (ODC-By), `LICENSE.txt` in the release |
| Required attribution | Kalyakulina et al., IEEE Access 2020 (doi:10.1109/ACCESS.2020.3029211), plus the standard PhysioNet citation (Goldberger et al., Circulation 2000). See README. |
| Integrity | All 2,805 files match PhysioNet's `SHA256SUMS.txt`; our per-file SHA-256 list is `docs/manifest.csv`. Original files are not modified. |

## Verified against the downloaded files (not just the documentation)

| Claim | Documentation | Verified |
|---|---|---|
| Records | 200 | 200 (`RECORDS`) |
| Subjects | 200, one record each | One `.hea` per ID; no subject ID other than record number, so record = subject |
| Leads | 12: i, ii, iii, avr, avl, avf, v1–v6 | Yes, all 200 headers; Lead II is channel `ii` (index 1 in every file; code selects by name) |
| Sampling rate | 500 Hz | 500 Hz in all 200 headers |
| Duration | 10 s | 5,000 samples in all 200 |
| Units | mV | `mV` for Lead II in all 200 |

## How the expert annotations were made

From the LUDB paper (arXiv:1809.03393, Section II) and README:

- Two certified, practicing cardiologists marked wave **onsets, peaks and offsets**
  for P waves, QRS complexes and T waves "by an eye inspection of each ECG signal
  and independently for each of 12 leads."
- The markup was **joint, by consensus**, "to decrease subjective influence and
  provide the end user with a definite annotation." So there is **one** label set,
  not two independent ones; inter-annotator variability cannot be measured from LUDB.
- Each lead has its own annotation file; Lead II labels are in `<record>.ii`.

## Annotation encoding (verified on all 200 Lead II files)

- Each wave is `(` onset, a peak symbol, `)` offset. Peak symbols: `p` (P),
  `N` (QRS), `t` (T). **`N` is the only QRS peak symbol** (1,831 on Lead II). In
  LUDB it marks a QRS peak; it is *not* a "normal beat" diagnosis, and we never
  interpret it as one.
- Where the reference sits: on development records, 865/889 Lead II QRS references
  are within 10 ms of the local signal maximum and **none** at a local minimum. The
  rest are R waves next to a pacemaker spike (record 34) or small positive r waves
  between larger negative deflections (record 83). So the reference marks the
  **R-wave (upward) peak**, not the largest absolute deflection. The prompt says this.
  Exception found in the full dev run: in paced beats with no upward R wave (e.g.
  record 90) the mark sits on the downslope of the wide negative complex, between onset
  and trough. The convention is ambiguous there; paced records (6 dev, 3 held-out; 9 records carry a "pacing." diagnosis) are
  reported as a separate subgroup and the labels are not altered.
- Quirks (none affect peak times): records 95 and 104 have QRS peaks without an
  onset `(`; record 111 has one out-of-order marker sequence at 7.84 s.

## Coverage: annotations do not span the whole recording

The paper doesn't say so, but **the first and last parts of every recording are
unannotated**: on Lead II the first annotation of any kind falls between 0.37 and
2.72 s, and the last between 7.35 and 9.63 s (median 1.32 and 8.64 s). Visible
heartbeats near the edges have no label (e.g. record 1: QRS at 0.02 s and 9.25 s
unlabeled). Scoring the full 10 s would therefore count correct answers as false
positives. See `docs/protocol.md` for the scoring window that handles this.

## Known limitations for this benchmark

- **Public data.** LUDB has been public since 2018–2020. Holding records out of our
  development process does not show a model never saw them (or the paper's figures)
  during pretraining.
- **Single consensus label set**: no measure of expert disagreement; the 150 ms
  tolerance is our event-matching rule, not an estimate of expert variability or
  clinical acceptability.
- **Population**: 200 adults from one region and one device type (Schiller
  Cardiovit AT-101), enriched for cardiac abnormalities (15 atrial fibrillation,
  9 paced, many hypertrophy). Not representative of any clinical population.
- **10-s strips**: only ~4–6 beats per record fall in the scoring window.
- **Header diagnoses** (`#` comment lines) are used only for private error
  analysis, never in prompts or images.
