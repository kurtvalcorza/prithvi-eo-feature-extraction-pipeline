# DIMER Philippines Rice Mapping Capstone — Specification

Prepared: 2026-10-11.
Status (as prepared 2026-10-11): design, with dataset files and checksums verified locally and on Kaggle and a Colab T4 compatibility smoke recorded in §4. The notebook has since been implemented and run on hosted T4; `docs/release-verification.md` holds its current status.

## 1. Learning artifact and scope

**Title:** Rice From Space: What Does a Satellite Foundation Model Add?

**Driving question:** Given 32 labelled Sentinel-2 chips, does a head trained on frozen Prithvi-EO-2.0 features map Philippine rice better than a hand-built spectral-temporal baseline, and what can 8 validation chips actually establish?

- Artifact: `tutorials/DIMER_Philippines_Rice_Mapping_Capstone.ipynb`.
- Proposed host: `kurtvalcorza/prithvi-eo-feature-extraction-pipeline` (it carries the verified Prithvi-EO-2.0-300M base model, pinned revision `9eb1b1102806593963daa333bcc491b1c6f8562f`, checkpoint SHA-256 `faab2c8b…6e7`).
- Preserve the existing tutorial and the EO & Climate workshop; this is a separate capstone reusing the repo's verified loading semantics.
- Normative basis: DIMER Notebook Specification (current; 2.4 since the 2026-10-11 review fixes); profile `E2E`, mode `WORKSHOP`, kind `CAPSTONE`, standalone (D1).
- Placement: optional capstone after the "Rice from space" course lessons L3 (Sentinel-2 LightGBM) and L3B (AlphaEarth LightGBM). It links to them rather than re-teaching feature engineering.
- Default: public pinned data, credential-free, no uploads, no manual restart, fresh Colab T4.
- GPU execution only on authorised hosted compute (Colab or Kaggle). Local verification is CPU-only and must not run while the rice study's memory-heavy jobs are active.
- Structure and pedagogy follow the fleet reference notebook `DIMER_Philippines_Flood_Mapping_Capstone` (NOTEBOOK_SPEC §25.15).

## 2. Learning outcomes

Learners will explain Sentinel-2 bands, 20 m pixels, cloud gaps and temporal compositing; explain why rice is a temporal signal (flooding, then green-up, then harvest); trace how the input a pretrained model expects is built without resizing; run the training of a small head on frozen features; compare it with a baseline under a frozen protocol; interpret pooled precision, recall, F1 and IoU on a minority class; run one controlled experiment; and write a bounded conclusion.

Shared sequence: orient -> Input / Model / Output -> predict -> run -> notice -> interpret -> change one thing -> compare -> conclude.

## 3. Dataset (verified)

Dataset: Kaggle `kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching`, version 1, **public**, CC BY 4.0 (the notebook pins version 1. Versions 2 and 3, published 2026-10-11, change only the card and rights wording with identical data, but Kaggle built no download archive for them; anonymous access to version 1 depends on the dataset setting that keeps all versions accessible, enabled by the maintainer on 2026-10-11). Verified 2026-10-11: anonymous page HTTP 200; `INDEX.jsonl` SHA-256 `e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d`; all 10 `PACKAGE.sha256` entries match the local copy; Kaggle lists the archives expanded into `data-train/`, `data-validation/`, `development-example/`.

- 40 chip-years, 2019–2021: 32 train, 8 validation. No test split in the teaching package.
- Each chip 128 x 128 px at 20 m, EPSG:32651.
- `S2` float32 [24, 6, 128, 128], bands B2/B3/B4/B8A/B11/B12, surface reflectance (DN x 0.0001). NaN where `S2_valid` = 0. 24 equal-duration calendar bins, median of masked observations.
- `AlphaEarth` float32 [64, 128, 128] — not used by the canonical systems (see §5).
- `label` uint8 [128, 128]: 0 non-rice, 1 rice, 255 ignore. Use `label`, never `label_raw`.
- **Label lineage (issue #22 M2; researcher pass 2026-10-12, `.agent/backups/rice-capstone-review-2026-10-11/issue22/M2_label_lineage.md`, confirmed against the study's build code).** Target: annual **paddy-rice occurrence according to the source map**. Source: Chen et al., Zenodo 10.5281/zenodo.15124936 — the annual `PADDYINTENSITY{year}` raster of the same year as the chip-year (2019/2020/2021); the four-year composite (the only layer with the uncertain code 5) and the planting-date layers are not used. Codes 1/2/3 (single/double/triple paddy rice) → 1; 0 → 0 (non-rice includes water and sea: the per-pixel land mask is all-true, the regional footprint is enforced at chip selection); any other code fails the build; 255 in `label_raw` only where the destination has no source coverage. Geometry: nearest-neighbour `rasterio.warp.reproject` from the 1/5400° EPSG:4326 crop onto the chip's 20 m EPSG:32651 grid (corners on multiples of 2,560 m) with a one-pixel halo. Ignore band: on the halo array, every pixel whose 3 × 3 neighbourhood holds both rice and non-rice is set to 255 — one pixel on each side of each edge (≈40 m across), diagonals included; then the halo is cropped. Scores cover retained labelled pixels only, so the edge band removes part of the small-field/edge-delineation problem from scoring. Immutable references: label recipe SHA-256 `7adf68fe9d23622a1cf008f71cacd7c5006ed1377e34a5fc8c17c046953dc63e` (every INDEX row and chip; `read_index` refuses any other); rebuild kit `recipe-kit.zip` SHA-256 `976530ffa324f0d701051d757214a5dca977621d5a3e752b4f11bac0dbf7f032` in Kaggle `kurtvalcorza/rice-philippines-s2-alphaearth-v1` version 2 (`recipe-kit/build_samples.py` `c6a25527561b492a1702a8cbc802f3196674f07c5cfd6e66cc4888cc70d1154a`, `recipe-kit/dataset_contract.py` `fc6512f1dee02bfe2a0fbbd6cbcc03ca36c88974dbd55c05e16efe347326b4a0`). Build-time QA re-derived `label_raw` and `label` for all 1,200 parent chips and matched (`QA.json` `spatial.label_reprojection_and_edge_band_verified: true`).
- **Open lineage gaps (not resolved by this capstone):** (1) whole-file integrity of the Chen rasters was not verified (Philippine crops were HTTP range-read from the authors' mirror; ZIP CRC/MD5 not checked); (2) the teaching dataset carries no builder or recipe — the link is the recipe hash and parent index digest, and parent version 1's kit is not anonymously downloadable (HTTP 404; confirmed in version 2 only); (3) per-chip records omit the label crop filename/hash (identity = `year` + recipe hash + build-time manifest assertion); (4) label verification reused the same function, not an independent implementation, and no Philippines-specific accuracy of the Chen maps is known beyond the authors' Monsoon-Asia figures; (5) the GDAL/rasterio versions governing nearest-neighbour behaviour are pinned only in the kit's `environment.txt`, not inspected.
- Validation chips are all from PH-07/PH-08. Two have no rice (`ph_242_425` 2019 and `ph_293_513` 2020, INDEX `raw_rice_fraction` 0); per-chip recall is undefined there, and F1 too when a system predicts no rice on them.
- The `development-example` is a parent **training** sample. It must not be used as a holdout.
- Attribution block (from `RIGHTS.md`, all four components): Chen et al. rice maps (Zenodo 10.5281/zenodo.15124936, CC BY 4.0); "Contains modified Copernicus Sentinel data 2019–2021"; AlphaEarth Foundations Satellite Embedding (Google / Google DeepMind, CC BY 4.0, Brown et al. 2025) if shown; geoBoundaries PHL ADM1 (CC BY 3.0 IGO). Plus the dataset compilation credit.
- Known stale text: the card and `RIGHTS.md` still say "privately staged". The dataset is public; the notebook must not repeat the stale sentence. Correcting the card is a separate dataset-version decision for the study owner.

## 4. Compatibility gate

Facts already established from the host repo source (`pipeline.py` on `origin/main`): the model is built at `img_size=224`, `num_frames=4`, patch 16, and the encoder accepts any H, W that are multiples of 16 (64–1024) and fewer than 4 frames (the positional embedding is sliced). Band order matches the dataset exactly. Pinned normalisation: reflectance x 10 000, then the config means/stds.

Colab T4 smoke, 2026-10-11 (pre-build feasibility, not release evidence; Colab CLI 0.7.4 sequential execution on a fresh Tesla T4, run 3, notebook SHA-256 `8d34e56b…3ce7`, 146 s wall): **PASS**.
- Runtime: Python 3.13.15, torch 2.11.0+cu130. Colab is now Python 3.13; the repo pins 3.12 and torch 2.14, so the capstone's isolated environment must pin its own interpreter.
- Checkpoint SHA-256 matched; anonymous `kagglehub` download of the public dataset in 29–33 s; `INDEX.jsonl` digest matched; Colab expands the archives (`data-train/` etc.).
- Real chip `ph_98_740` 2021: quarterly rule picked bins 0, 6, 17, 19 (valid 1.0/1.0/0.998/0.998); NaN 0.11% of selected frames; reflectance 0.003–0.545; normalised −1.46 to 3.22.
- Strict-equivalent load: 0 missing, 0 unexpected keys. Encoder at 128 px returned 1 + T·64 tokens for T = 4, 3, 1 (257/193/65), all finite. No resizing.
- Frozen encoder + head, batch 8, fp16 autocast: one step 0.62 s, peak 1.40 GiB. Unfrozen encoder backward, batch 4: peak 2.69 GiB (without optimizer state; AdamW on 300M parameters would add roughly 3–4 GiB — inferred, not measured).
- Failure log (kept): run 1 upgraded numpy inside the live kernel and the model import crashed (NOTEBOOK_SPEC §25.13 anti-pattern); run 2 pinning numpy made pip refuse terratorch 1.2.13. Run 3 ran the model stages in a fresh interpreter. `colab exec` exited 0 on the failing runs, so results must be judged from outputs.

Remaining gate items before build:
1. Resolution mismatch: Prithvi-EO-2.0 was pretrained on 30 m HLS; this data is 20 m Sentinel-2 surface reflectance. State it as a domain shift; the capstone keeps the native grid (resampling or harmonisation would be a different design choice, not tested here — issue #22 S1).
2. Processing level: Prithvi expects HLS-style surface reflectance; this dataset is S2 surface reflectance. Close match; disclose.
3. Frame selection: four frames, the clearest valid bin in each calendar quarter, earliest on ties (study protocol line 58). Record selected bins and valid fractions per chip.
4. Missing data: NaN after standardisation -> 0 (the pretraining mean). Record the filled fraction per chip and frame. Pixels with no valid observation in any selected frame stay scored, as in the study.
5. Output resolution: 16 px patches = 320 m at 20 m. A head on patch tokens upsamples a coarse grid; field edges will blur. Distinguish output pixel spacing (20 m), token spacing (320 m) and resolved detail; attribute the limit to this head (no skip connection or learned sub-patch decoder), not to every Prithvi-based segmentation design (issue #22 S1).

## 5. Systems compared

### A. All-non-rice baseline
Predicts non-rice everywhere. Shows why accuracy hides failure on a minority class (rice is 11.3% of the retained labelled pixels of the teaching validation chips: 12,889 of 113,821).

### B. Spectral-temporal LightGBM baseline
The study's M1 feature recipe (70 axes: per-band observed statistics, quarterly means, quarterly valid fractions) on a declared, class-stratified pixel subsample of the fit chips. One fixed configuration from the study protocol (num_leaves 31). Threshold selected on the development chips. Links to course lesson L3.

### C. Frozen Prithvi-EO-2.0 + small segmentation head
Encoder frozen. Head: tokens of encoder block 12 (the repo's validated `FEATURE_LAYER` = 11; D3), temporal mean, 1x1 conv 1024->256, GELU, 1x1 conv 256->2, bilinear upsample to 128 px. Cross-entropy with `ignore_index=255`. Fixed epochs, seed, learning rate; threshold selected on dev. Training on Colab T4.

Reference only (not re-run): course lessons report teaching-scale F1 on the same 8 validation chips — Sentinel-2 LightGBM 0.760 (L3) and AlphaEarth LightGBM 0.928 (L3B). Quote them as lesson results with their caveats. Full-study M1/M2/M5 numbers stay out until the study's test report is published, and only with Kurt's approval.

## 6. Protocol (frozen before validation)

1. Split the 32 train chips into **fit** and **development** by whole region, using a declared seeded rule, targeting about 25% of chips in development and at least two rice-containing development chips. Region grouping reduces leakage risk but does not guarantee distance across a boundary, so `prepare` writes `spatial_audit.json` (issue #22 S2): unique sites, repeated years, sites or footprints shared/touching across roles, minimum cross-role ground distance. For the pinned index (computed offline 2026-10-12): 40 chips, 40 unique sites, no repeated years, no shared or touching footprints across roles; minimum distances development–fit 28,621.7 m (`ph_172_604_2021` / `ph_183_598_2021`), fit–validation 190,268.5 m, development–validation 384,000 m. No pixel bootstrap or confidence interval is added for the eight clustered validation chips.
2. All choices (B's threshold, C's threshold, C's epochs, the §8 experiment) are made on development only and written to `selection.json`.
3. Then score A, B and C once on the 8 validation chips. Preserve the result even if B wins.
4. The development example is never used for selection or scoring.

## 7. Guided cell sequence

1. Orient: question, intended use, limits, runtime, access.
2. Infrastructure: isolated hash-locked environment, hardware checks, anonymous downloads, checksum verification (collapsed `@title Infrastructure:` cells).
3. Data inspection: true/false-colour composites, NDVI time series for a rice and a non-rice pixel, label overlay, valid-observation counts by bin.
4. Predict: where will a pretrained model and a hand-built baseline disagree?
5. Declare protocol: fit/dev/validation roles, selection rules, scoring conventions.
6. Baselines A and B, threshold chosen on dev.
7. Prithvi input building: quarterly frame selection, NaN fill, normalisation; show what was filled.
8. Train head C on fit chips; dev curves; threshold on dev.
9. Controlled activity on dev (§8); freeze the record.
10. Single validation scoring of A, B, C.
11. Interpretation: pooled metrics, per-chip table (undefined values stated), disagreement and error maps, the 320 m patch effect.
12. Export, fresh reload of the head artifact, integrity verification, the run's own answer to the driving question (computed from its validation table, with its evidence level) followed by a conclusion scaffold, attribution, References (IEEE numbered citations, each entry with a DOI or a link pinned to the revision used; NOTEBOOK_SPEC 2.4 DOC14–DOC16). The opening carries the 2.4 use notice, a problem statement and a data context (DOC1–DOC8, DOC17–DOC21).

## 8. Controlled learner experiment

Change one thing on development chips only: **number of frames, T = 1 (best single quarter) vs 4**. Keep the frozen weights, the head's recipe (architecture, optimiser, seed), the epoch count and the threshold fixed; train one head per frame count with that recipe (Kurt, 2026-10-11: the activity measures the information in one frame versus four, not C's robustness to fewer frames). Ask learners to predict first, with a neutral prompt: a single quarter may fall below, match or exceed T = 4. This is a **fixed-recipe, development-only comparison** (issue #22 M1): the T = 1 heads borrow the epoch count and threshold selected for T = 4, so it is not an equally tuned comparison of one-frame and four-frame systems, and a T = 4 advantage does not isolate seasonal progression from improved observation coverage or from settings better suited to four frames. Canonical validation uses T = 4. Optional future work (not done here): tune each variant under the same search budget; coverage-matched or temporal-order controls.

## 9. Metrics and evidence boundaries

- Aggregate TP/FP/FN/TN over pixels with `label != 255`; report pooled rice F1 (primary), IoU, precision, recall. Undefined ratios are null with a reason.
- Observation support (issue #22 S3): B receives quarterly observed-fraction features; C receives mean-filled standardised inputs without a validity channel, and pixels unobserved in all selected frames stay scored. `observation_support.csv` splits B's and C's pooled metrics by the number of observed selected frames (0–4) as a secondary diagnostic; the primary common evaluation mask is unchanged. `<chip>_observed_frames.tif` exports the count.
- Per-chip table for diagnosis only. Chips and pixels are spatially clustered (many chips touch); no pixel-level or chip-level significance tests or narrow confidence intervals.
- 8 validation chips from two regions: no national or operational claim.
- B and C differ in temporal information, missing-data handling, class sampling and prediction architecture: a whole-pipeline comparison, not an isolation of pretraining (issue #22 S1). Keep three claims apart: some reference-rice signal (F1 above A), comparator performance (C vs B under the declared metric), and practical usefulness (not addressed) (S4).
- Labels are published map products, not field observations. Disagreement is not proof of model error.
- Prithvi pretraining data overlap with these locations is unknown; say so.

## 10. Artifacts and reproducibility

`results.zip`: `run_summary.json`, `data_manifest.json`, `preprocessing.json` (frames chosen, filled fractions), `selection.json`, `metrics.csv`, `per_chip_metrics.csv`, `activity.csv`, the head adapter (safetensors + manifest), prediction GeoTIFFs per validation chip (the scored map 0/1/255, where 255 marks pixels the reference ignores; a 0/1 decision for every pixel; float32 rice scores so a user can re-threshold), all three checked by the reload, figures, `conclusion.md`, `checksums.json`, attribution text. No base weights or source imagery in the bundle. Fresh-process reload must reproduce validation predictions exactly (or within declared tolerance with identical binary output) and verify each GeoTIFF's placement — shape, CRS GeoKeys, pixel-is-area, full transform (origin and 20 m pixel size) and bounds — against the chip's recorded corner (issue #22 m1). The CPU tests also parse the written GeoKey tags with an independent struct-based reader; a Rasterio/GDAL round trip is not part of CI (neither is a CI dependency) and was run once locally as supporting evidence.

## 11. Runtime and scope controls

Target: fresh Colab T4. Measure actual runtime and peak VRAM before publishing estimates. Download only the teaching package (about 350 MB) and the base checkpoint (1.33 GB). Embed reference code at generation time; no repo clone, no worker calls.

## 12. Owner decisions (Kurt, 2026-10-11)

- **D1 Profile: `E2E`.** The notebook trains a head and exports an adapter artifact. BYOD is required: a user supplies HDF5 chips in this dataset's schema (same keys, shapes and band order), the notebook validates them, and the exported head scores them. Because NOTEBOOK_SPEC DAT14 forbids an inference-only BYOD in an `E2E` adaptation workflow, BYOD also re-adapts a head on the user's labelled chips under the same protocol (site-grouped fit/development/test, epoch and threshold on development, export, then a head-serialization parity check: the saved head is read back in the same process and re-applied to the cached test features; scores within 1e-4 and identical decisions are required, otherwise the stage fails with a verification error, discards the adapted head and writes no BYOD bundle or receipt — issue #22 M3. This is not the fresh-process, input-to-prediction reload used for the canonical head) when at least 6 labelled chips from at least three sites are supplied. Site ids come from `group_id`/`group`; a file-name fallback is recorded as unverified; identical-content chips or overlapping footprints with different site ids stop the adaptation (prediction still runs), and the adapted split's spatial audit is recorded (issue #22 S2).
- **D2 Encoder: frozen head as the canonical system, plus an optional fine-tuning extension.** The canonical comparison (A, B, C) uses the frozen encoder. A clearly separated, opt-in section fine-tunes the encoder (system C-FT: the patch embedding, the CLS token and encoder blocks 1–12) under the same fit/dev protocol; its epochs and learning rate are fixed in advance or chosen on dev only, and it is scored on validation only if selected before validation scoring. Smoke headroom: unfrozen backward at batch 4 peaked at 2.69 GiB without optimizer state; measure the full step with AdamW before fixing the batch size.
- **D3 Feature block: fixed block 12** (the repo's validated `FEATURE_LAYER` = 11, zero-based). No sweep.

## 13. Acceptance and handoff

CPU checks: frame selection ties, NaN fill accounting, normalisation idempotence, label mapping (255 ignored), fit/dev split determinism and disjointness, metric edge cases (no-rice chip), GeoTIFF/CSV/ZIP round trips, notebook schema, generator parity. Hosted check: exact generated notebook, fresh T4, complete default path, exports and fresh-process reload. Status stays Candidate until hosted execution is recorded. Work on a feature branch; this specification does not authorise commit, push, PR, merge or DIMER upload.

## Sources

- Notebook standard: https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md
- Reference capstone: https://github.com/kurtvalcorza/prithvi-flood-segmentation-pipeline/blob/main/tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb
- Host pipeline: https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline
- Model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
- Dataset: https://www.kaggle.com/datasets/kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching
- Rice labels: https://doi.org/10.5281/zenodo.15124936
