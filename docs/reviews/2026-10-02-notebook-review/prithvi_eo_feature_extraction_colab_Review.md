# Prithvi-EO-2.0-300M Feature-Extraction E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/prithvi-eo-feature-extraction-pipeline`  
**Notebook:** `tutorials/prithvi_eo_feature_extraction_colab.ipynb`  
**Reviewed commit:** `9380af24d8414afc3c8eeda91f9c4fc8549c09e6` (`main`, confirmed with `gh api repos/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/commits/main`)  
**Notebook Git blob:** `6a12fd2ed9fa00c7090ca94a80303806e16aceb5`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-25 (commit `e611e93`). No commit since `e611e93` touches `src/`, `tools/build_notebook.py`, `tools/notebook_template.py`, `pyproject.toml` or this notebook; the later commits concern the separate EO workshop notebook.  
**Finding prefix:** `FX` (`EO` is already used in this repository by the workshop review)  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The default path is careful and honestly framed. The notebook statically audits the pickled checkpoint against a three-global allow-list, unpickles it once through torch's weights-only loader into a digest-pinned safetensors file, builds the model from the pinned `terratorch`, embeds and reconstructs the four upstream example tiles against a mean-fill baseline, streams exactly 88 digest-pinned members out of a digest-pinned 2.6 GB tarball without `extractall`, takes roles from the burn-scar model repository's splits, scores a zero-head linear probe that *is* the majority baseline, trains a 2,050-parameter probe with validation-loss epoch selection, and reloads the exported probe with exact parity. The prose reads the result against the right reference (the burn-scar row's pixel-level fine-tune on the same 12 test scenes, confirmed identical by name) and says what the patch-level number does not establish.

| Measure | This review (CPU, stub encoder, synthetic chips) | Kaggle T4 record (blob `6a12fd2e`) |
|---|---|---|
| Code cells completed | default-path cells 13–21 executed against the repository's stub encoder: 5/5 ok | 10/10 on pass 2; pass 1 stopped at the install guard |
| Test patch-level burn-scar IoU, baseline / zero head / probe | not meaningful on a stub | 0 / 0 / **0.8149** (F1 0.898, accuracy 0.7865 → 0.9539) |
| Masked reconstruction MSE vs mean fill (four-date series) | — | 0.0773 vs 0.1294 (every single date also below its baseline) |
| BYOD path, `USE_BYOD = True` re-run from Section 4 as documented | **crashes** in Section 8 with `KeyError: 'source_id'`; no probe maps, artifact, reload or `result.json` for the learner's data | not run |
| Section 5 "frozen zero head" when Section 5 is re-run after Section 6 | **the trained probe** (stub test burn IoU 0.9822 vs 0 for a true zero head; `adapted: True`) | not run |
| BYOD comparison table after the documented re-run | `frozen_zero_head` 0.9856 vs `probe` 0.9825 — the "frozen" column is the sample-trained probe | not run |
| BYOD smallest dataset accepted | **7 chips** (stated minimum of 4 refused; 5 and 6 refused too) | not run |
| Optional experiments `FEATURE_LAYER = 23`, `CLASS_BALANCE = True` re-run from Section 6 | ok; `adapt()` starts from a fresh zero head each time and the frozen column stays correct | not run |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (FX-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3) and passed only after a restart. `docs/release-verification.md` calls the restart "expected", and the repository marks the blob `Release-grade` on that run. This repository's own EO workshop notebook already uses the uv isolated environment that removes the restart.
2. **The documented BYOD path cannot finish (FX-M2).** Section 8 indexes `record['source_id']`, which only the sample loader sets, so no probe map, artifact, reload parity or `result.json` is produced for the learner's data, and the default run's exports stay in `outputs/` beside the BYOD evaluation report.
3. **Section 5's "frozen" numbers are not frozen on a re-run (FX-M3).** Nothing detaches the probe before Section 5, so re-running Section 5, or following the BYOD instruction "re-run from that cell" in Section 4, scores the already-trained probe and prints it as `frozen_test_zero_head`; Section 7's comparison then shows the "frozen" model beating the probe.
4. **Guided layer largely absent (FX-M4).** The notebook is declared `GUIDED` but has no audience statement, how-to-use section, roadmap, glossary, prediction, checkpoint, troubleshooting or conclusion template; 2,033 carried-module lines sit in three unlabelled, uncollapsed cells.

FX-M1 and FX-M4 match the sibling Prithvi reviews (burn scar BS-M1/BS-M3, crop CR-M1/CR-M4), which use the same template family. FX-M2 matches CR-M2 (crop). FX-M3 is the narrower form of BS-M2/CR-M3: here `adapt()` itself re-initialises the head to zero, so Section 6's epoch 0 and re-runs of Section 6 are correct; only the Section 5 evaluation is stale.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. The Prerequisites ("Knowledge") assume the reader knows multispectral surface reflectance (bands, scaling, no-data), what a masked autoencoder reconstructs, token embeddings and linear probes, and how IoU, precision and recall are read against a majority baseline |
| Supported runtime | "a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12)"; "about 2.7 GB of GPU memory"; about 6 GB of disk |
| Promised outcomes | Pinned install; carried package (3 modules); 7-file snapshot staged and digest-verified; pickle audited and converted once; example tiles embedded as a series and per date, reconstructed under a 75 % mask against mean fill; 88 pinned members extracted, 44 scenes validated and assigned 24 / 8 / 12 with three refusals; majority baseline = zero head; bounded linear probe (2,050 parameters) on frozen block-12 tokens; paired held-out comparison; probe maps; safetensors probe export and fresh reload with parity; BYOD zip through "the same contract — validation, majority baseline, probe training, held-out evaluation, probe maps, artifact export and reload parity" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `1eb24297` (an ancestor of `main`) |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 23 cells (10 code). Cells 5, 7 and 9 carry `pipeline.py` (1,211 lines), `metrics.py` (76) and `samples.py` (746). Also read: `pipeline.py` (`validate_dataset`, `predict`, `evaluate`, `_probe_logits`, `adapt`, `save_artifact`, `load_artifact`, `check_artifact_manifest`, `from_artifact`), `samples.py` (`SAMPLE_RECORDS`, `read_corpus`, `split_dataset`, `load_byod_dataset`, `write_sample_pair`, `write_dataset_csv`, `dataset_manifest`), `README.md`, `tutorials/README.md`, `docs/release-verification.md`, `STATUS.md`, `tests/test_adaptation.py`, `tests/conftest.py`, `tools/notebook_template.py`. `docs/execution-evidence/` holds only the workshop notebook's record.
- **Documented execution evidence.** `docs/release-verification.md`, row 2026-09-25 (`e611e93` / `6a12fd2e`), and the workspace archive it cites (`.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-eo-feature-extraction/v2/evidence/`: `run_summary.json`, `executed.ipynb`, `executed-pass1.ipynb`, `outputs/`). Kaggle Tesla T4, **the reviewed blob** (`fetched_blob_verified: true`), clean Hugging Face cache. Pass 1 (22:46:53Z, 205.2 s) raised `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.3; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Pass 2 (22:50:19Z, 232.1 s) completed 10/10. The earlier 2026-09-20 run of blob `28e2298e` shows the same restart.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), the shared `eo-notebook-test` conda env (Python 3.12.14, torch 2.13.0+cpu, numpy 2.5.3, tifffile 2026.9.20, safetensors 0.8.0; no `terratorch`, `timm` or `lightning`; nothing installed). No weights were downloaded and no tarball was fetched; the four real example tiles were read from a local copy of the pinned snapshot.
  - **How the cells were run:** the three carried-module cells were executed from the notebook itself; Section 1 ran with the documented `DIMER_NOTEBOOK_CI_PREINSTALLED=1` switch (install skipped; `timm`/`lightning` replaced by version-only stand-ins); Section 3 was not run. `PrithviFeaturePipeline.from_pretrained` (also used by `from_artifact`) was replaced by a factory returning the repository's own stub masked autoencoder (copied from `tests/test_adaptation.py`), `verify_converted` by a dummy; `fetch_sample_dataset` returned 8 / 4 / 4 synthetic chips (`tests/conftest.py`) carrying `source_id` and `region` like the real loader; `google.colab.files.upload` returned a synthetic BYOD zip. Cells 13, 15, 17, 19 and 21 were then executed verbatim (form defaults), except `importlib.metadata.version('terratorch')` in cell 21 replaced by a constant and the form substitutions named per probe.
  - **Probes (about 74 s in total):**
    - P1: notebook parse, every code cell compiles, guided-layer markers, display calls, `source_id` uses, stated BYOD minimum, threshold/calibration wording, whether Section 5 resets the probe.
    - P2: `tools/build_notebook.py --check` (OK), `tools/validate_release_assets.py` (PASS), offline suite with `PYTHONPATH=src` (65 passed, 1 skipped `test_model_backed.py`, 3 failed — all Windows-host artefacts: two `pass_fds not supported on Windows` in the workshop bridge tests, one `OSError [Errno 22]` overwriting a memory-mapped `adapter.safetensors`; CI runs on Linux and none of the three touches this notebook's cells).
    - P3: the carried `load_byod_dataset` + `split_dataset(seed=0)`, exactly as Section 4 calls them, on 8 synthetic zips.
    - P5: default cells 13–21 on the stub (5/5 ok).
    - P6: Section 5 re-run after a complete run; its "frozen zero head" vs a true zero head.
    - P7: `FEATURE_LAYER = 23` and `CLASS_BALANCE = True`, re-run from Section 6 (documented optional experiments).
    - P8: the documented BYOD route (`USE_BYOD = True`, re-run from Section 4) after a default run.
    - P9: BYOD on a fresh state with 4 of 10 chips lacking burn pixels (ran through Section 5).
    - P10: the sample's tiles per role, and the exported `…_sample_pairs.csv` names vs the extracted member names.
- **Not verified:** any notebook cell on real weights here; any Colab run (the stated runtime has no record for this notebook); the real upload dialog; the optional experiments on real weights; the "about 2.7 GB of GPU memory" figure.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| **Technical correctness** | Strong on the default path: provenance, audit, conversion, extraction, metrics, epoch selection and reload are sound and recorded, and `adapt()` is transactional and always starts from a zero head. Three state/branch defects: the install needs a restart (FX-M1), the BYOD branch crashes on a sample-only key and leaves the previous run's exports in place (FX-M2), and Section 5 scores whatever probe is attached while labelling it the zero head (FX-M3). BYOD error handling is partly unactionable (FX-m1). |
| **Promise fulfilment** | The default promises are delivered on the recorded run. The BYOD promise ("probe maps, artifact export and reload parity") is not delivered (FX-M2), and the BYOD comparison it reaches is mislabelled (FX-M3). |
| **Learner experience** | Each stage is explained and Sections 4–7 have "Look for" notes that match the record (val loss 0.6931 → 0.1996, val F1 0.7577, test IoU 0.8149, MSE 0.0773 vs 0.1294). But there is no guided layer (FX-M4), no figure is ever shown in an imagery tutorial (FX-m4), and the interpretation and optional experiments state sample outcomes without saying how to read them on other data (FX-m6). |
| **Spec conformance** | Open `MUST`s: RUN1, RUN10, ENV6, REL2/REL11 (FX-M1); DAT10, DAT14, REL12 (FX-M2, FX-M3); DAT12, DAT19 (FX-m1); SPL5 (FX-m2, BYOD); UNC4 (FX-m5). Open `SHOULD`s: GDL1–GDL14, UX8 (FX-M4); UX3, UX11 (FX-m4); UX10 (FX-m1); UNC3 (FX-m5); SPL10 (FX-m2); GDL8/GDL10/GDL14 (FX-m6); EXE5 (FX-S3). |

## 3. Findings

### FX-M1 — Major: `Run all` needs a manual restart after the install cell

**Location:** Section 1, install cell (cell 3); `docs/release-verification.md` procedure step 4 (line 81: "an interpreter restart after the install is expected") and Current status; `STATUS.md`; generator `tools/build_notebook.py` lines 61–69 (install cell body).

**Observed issue:** The install cell `pip install`s the seven pins into the running kernel and raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')` whenever a pin replaced an already-imported distribution. In a fresh Kaggle T4 image that always happens (`numpy` 2.0.2 preloaded vs pin 2.5.3; `cuda-bindings` 12.9.4 vs 13.4.3 pulled in by `torch==2.14.0`).

**Consequence:** A learner who chooses `Run all` stops at cell 3 after about 3.5 minutes of installation and must restart and run again. The release record treats this as expected and marks the blob `Release-grade` on a two-pass run, which RUN1, RUN10 and ENV6 forbid.

**Evidence:** Documented execution evidence: `run_summary.json` passes `[{attempt 1, ok false, 205.2 s}, {attempt 2, ok true, 232.1 s}]`, `restarted_after_install_cell: true`, error text above; `executed-pass1.ipynb`. Release record: "10/10 ok (1 restart after install cell)" for both recorded runs. Source inspection: the guard in cell 3.

**Recommended correction:** Replace the in-kernel install with the uv isolated-environment pattern: a carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart is needed. The nearest reference is this repository's own workshop notebook (`tutorials/DIMER_AI_for_Earth_Observation_and_Climate_Applications_Workshop.ipynb`, generated by `tools/eo_workshop/build_workshop.py`, uv move in `b110907`, one-pass Colab T4 record 2026-10-04 with `tutorials/requirements-eo-workshop.lock.txt`); the fleet reference is `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (origin/main). Make the change in the generator and regenerate; remove "restart is expected" from the release procedure and return the status to `Candidate` until a one-pass run is recorded.

**Acceptance check:** On a fresh Colab or Kaggle GPU runtime, `Run all` on the regenerated blob completes every code cell in one pass with no error output and no restart; the release record shows one pass for that blob.

**Spec:** RUN1, RUN10, ENV6, REL2, REL11.

### FX-M2 — Major: the BYOD branch crashes in Section 8 (`KeyError: 'source_id'`) and leaves the previous run's exports in `outputs/`

**Location:** Section 8 (cell 21): `tifffile.imwrite(f'outputs/…_probe_map_' + record['source_id'] + '.tif', …)` and `print({'scene': record['source_id'], …})`; `load_byod_dataset` in `samples.py`; generator `tools/notebook_template.py` lines 346 and 348.

**Observed issue:** Only the sample loader (`read_corpus`) sets `source_id` and `region`. `load_byod_dataset` returns `{id, image, label}`, so in BYOD mode the first statement of Section 8's loop raises `KeyError: 'source_id'`. Nothing after it runs: no probe map, no `predictions.json`, no `save_artifact`, no `from_artifact` reload, no parity assertion, no `result.json`. Because the BYOD route is a re-run after the default path, the default run's `result.json`, `predictions.json`, four probe maps and adapter stay in `outputs/` beside a BYOD `evaluation_report.json`.

**Consequence:** The opening cell promises that BYOD chips "flow through the same contract — validation, majority baseline, probe training, held-out evaluation, probe maps, artifact export and reload parity". A learner who follows the BYOD instruction gets a traceback in the last section, no reusable probe for their own data, and an `outputs/` folder whose `result.json` and adapter describe the sample run while the evaluation report describes theirs. The default sample path is unaffected. Section 4 also prints `'tiles': []` for BYOD splits (it uses `.get`, so it does not crash).

**Evidence:** Direct execution (P8): the actual cells 13 → 21 with `USE_BYOD = True` and a valid synthetic 8-chip zip (splits 4 / 2 / 2): cells 13–19 ok, cell 21 `KeyError: 'source_id'` with nothing printed; afterwards `outputs/…_result.json` carries the sample `data_source` and `outputs/…_evaluation_report.json` carries `BYOD (byod8.zip)`. P3: BYOD record keys are `['id', 'image', 'label']`. P1: `source_id` is indexed twice in cell 21. The offline tests never run the tutorial cells on BYOD records.

**Recommended correction:** In the template, use `record.get('source_id', record['id'])` (or have `load_byod_dataset` set `source_id = id` and accept an optional `region` column); clear the run's export names at the start of Section 4 (as Section 8 already does for the adapter directory) or write BYOD exports under a run-specific prefix; add a unit test that executes the Section 4–8 cell bodies on `load_byod_dataset` output with the stub pipeline already in `tests/test_adaptation.py`.

**Acceptance check:** With a stub pipeline and a valid 8-chip BYOD zip, executing cells 13–21 with `USE_BYOD = True` after a default run completes without error, and every file in `outputs/` that records a data source records the BYOD zip; a hosted BYOD run (REL12) reaches the reload-parity print.

**Spec:** DAT10, DAT14, REL12.

### FX-M3 — Major: Section 5 scores whatever probe is attached and labels it the frozen zero head

**Location:** Opening cell ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); Section 5 (cell 15, `frozen_test = pipe.evaluate(test_records)`, `frozen_val = pipe.evaluate(val_records)`, printed as `frozen_test_zero_head` / `frozen_validation_zero_head`); Section 7 (cell 19, comparison column `frozen_zero_head`, `validation_burn_iou.zero_head`, evaluation report `frozen` block); generator `tools/notebook_template.py` line 237.

**Observed issue:** `evaluate()` uses the attached probe; the zero head exists only while `pipe.probe is None`. Section 5 never detaches the probe, so once Section 6 has run, re-running Section 5 — alone, or as part of the documented BYOD route from Section 4 — evaluates the trained probe and prints it under the zero-head labels. `adapt()` itself is not affected: it builds a fresh zero-initialised head each time, so Section 6's epoch 0 (`"zero head (majority baseline)"`) and re-runs of Section 6 alone stay correct.

**Consequence:** On the learner's own data the comparison the notebook builds is between the sample-trained probe (labelled "frozen zero head") and the BYOD probe. In the stub run the table read `frozen_zero_head 0.9856, probe 0.9825` — a learner would conclude that training made the probe worse than an untrained head. The same section prints validation burn IoU `zero_head 0.9888` beside validation loss `zero_head 0.6931` (the true zero head from `adapt()`'s history), an internal contradiction with no explanation. Only the separate `baseline_not_burned` column stays correct.

**Evidence:** Direct execution on the stub with the actual cells. P6: after a complete default run, re-running cell 15 prints `frozen_test_zero_head` burn IoU 0.9822 (= the probe's test IoU; a true zero head scores 0.0) and `evaluate()` reports `adapted: True`. P8: the BYOD re-run from Section 4 prints `frozen_test_zero_head` burn IoU 0.9856 (true zero head on those chips: 0.0) and Section 7 the comparison above. P7: `FEATURE_LAYER = 23` and `CLASS_BALANCE = True` re-run from Section 6 complete with epoch 0 noted as the zero head and the frozen column unchanged. P1: no reset in cell 15. The stub's magnitudes are not the real network's; the mechanism is.

**Recommended correction:** At the start of Section 5, detach the probe (`pipe.probe = pipe.feature_stats = pipe.adapter = None`, `pipe.feature_layer = FEATURE_LAYER`) or score the zero head through a dedicated call (for example `pipe.evaluate(records, probe=None)`), and take Section 7's zero-head validation numbers from `adapt_result['history'][0]`; state each optional experiment's rerun scope ("re-run Sections 6–8"). Fix in `pipeline.py` and/or `tools/notebook_template.py`, then regenerate.

**Acceptance check:** After a complete default run, re-running from Section 4 (sample or BYOD) prints Section 5 zero-head metrics equal to `baseline_not_burned` (burn-scar IoU 0, accuracy equal to the not-burned fraction) and `adapted: False`, and Section 7's `frozen_zero_head` column equals that of a fresh run on the same data.

**Spec:** DAT14, REL12 (GDL10 for the rerun instruction).

### FX-M4 — Major (learner-facing): the declared `GUIDED` layer is largely absent

**Location:** Whole notebook; carried module cells 5, 7, 9; generator `tools/notebook_template.py`.

**Observed issue:** The notebook declares mode `GUIDED` but has no intended-audience statement (GDL1), no **How to use this notebook** (GDL2), no roadmap (GDL3), no Input → Model → Output task contract near the opening (GDL4; the Prerequisites' data contract is the closest), no glossary although HLS, surface reflectance, masked autoencoder, CLS token, mean token, patch token, encoder block, standardised MSE, mean fill, linear probe, majority baseline, patch-level IoU, ignore class, pickle audit and safetensors all appear (GDL6), no prediction before any principal result (GDL7), no interpretation checkpoints or sample answers (GDL9), no Predict → Change → Run → Observe → Explain activity (GDL10; the optional experiments are one sentence), no Infrastructure labelling or collapsed carrier cells (GDL11; 2,033 lines in three cells, no `cellView`/collapsed metadata on any code cell), no troubleshooting (GDL13) and no conclusion template (GDL14). Sections end without a synthesis (UX8). "Look for" notes exist for Sections 1, 4, 5 and 6 (GDL8 partly met).

**Consequence:** A self-paced learner meets 2,033 lines of carrier code before any lesson, gets no help separating essentials from infrastructure, and is never asked to predict, check or explain anything, so objectives such as "read the CLS and mean-token embeddings" and "compare the probe with the majority baseline" are exercised only by reading printed dictionaries.

**Evidence:** Source inspection; P1 markers (`How to use`, `Glossary`, `Troubleshoot`, `Roadmap`, `Infrastructure`, `Check your`, `audience`, `What to notice`, `Expected result`, `conclusion` all absent; the only "predict" hits are the `pipe.predict` API name and prose about the probe).

**Recommended correction:** Add the NOTEBOOK_SPEC 2.2 guided layer in `tools/notebook_template.py`: audience, how-to-use, roadmap, task contract, collapsible glossary, a prediction before Sections 5 and 7 (for example "which block's tokens will separate burned patches better, the last or a middle one?" before the `FEATURE_LAYER` experiment), "What to notice" notes, collapsible "Check your reasoning" answers, one bounded PCROE activity with its rerun scope (the `FEATURE_LAYER = 23` comparison is a natural one), `# @title Infrastructure: …` with `cellView: form` on the carrier cells, a troubleshooting section (install/restart, Hub and tarball download, disk, GPU memory, BYOD errors) and a conclusion template.

**Acceptance check:** The GDL1–GDL14 checklist in NOTEBOOK_SPEC 2.2 passes item by item on the regenerated notebook, and the three carrier cells open collapsed in Colab.

**Spec:** GDL1–GDL14, UX8.

### FX-m1 — Minor: BYOD stated minimum is wrong, several failures are not actionable, and the refusal probes misfire on small BYOD sets

**Location:** Opening cell ("at least four chips with some positive pixels"); Section 4 (cell 13) upload branch and refusal probes; `split_dataset` and `load_byod_dataset` in `samples.py`; generator `tools/notebook_template.py` lines 68, 157, 179–180.

**Observed issue:** `split_dataset(seed=0)` takes 25 % for test and 20 % for validation and requires 4 training chips, so 4, 5 and 6 chips are refused and 7 is the true minimum. A `pairs.csv` row naming a file missing from the zip raises a bare `KeyError: 'c3.tif'`; a non-TIFF file raises `TiffFileError: not a TIFF file`; cancelling the upload dialog raises `StopIteration` from `next(iter(uploaded.items()))`. In BYOD mode the first two refusal probes build their lists from `test_records[1:4]`; with fewer than 4 test chips (any BYOD set under 14 chips) both are rejected with `2 records; 4..2000 are required` instead of the band-count and label-class checks they are meant to show.

**Consequence:** A learner following the stated contract with 4–6 chips is refused after uploading; three common mistakes give errors that do not name the contract or the fix; and the refusal demonstration silently stops demonstrating anything while still printing "rejected".

**Evidence:** Direct execution (P3): 4 → `split leaves 2 training chips; at least 4 are required`, 5 and 6 → 3 training chips, 7 → 4 / 1 / 2, 8 → 4 / 2 / 2; missing member → `KeyError: 'c3.tif'`; bad image → `TiffFileError`; no `pairs.csv` → actionable `ValueError`; empty upload expression → `StopIteration`. P8: the BYOD re-run printed both shape/label probes as `'2 records; 4..2000 are required'`. The cancelled dialog itself was not exercised (source).

**Recommended correction:** State "at least 7 labelled chips (the seeded split keeps 4 for training)" or make the split keep 4 training chips first; wrap the BYOD branch to turn a missing member, an unreadable TIFF and an empty upload into messages that name the row and the fix; build the refusal probes from `validate_dataset(..., min_records=1)` or from four copies of one valid record so they test the intended rule.

**Acceptance check:** A 7-chip BYOD zip that meets the stated contract is accepted; a missing member, a non-TIFF file and a cancelled upload each produce a message naming the file or step and the fix; in BYOD mode with 2 test chips the probes report the band-count and label-value failures.

**Spec:** DAT12, DAT19, UX10.

### FX-m2 — Minor: two sample test scenes share an HLS tile with training scenes, unstated; BYOD split is per scene with no group key

**Location:** Section 4 markdown and print (`'tiles'` per split); Interpretation ("Split by site or tile, not by scene"); `split_dataset` in `samples.py`.

**Observed issue:** The 12 test scenes include tiles `T10TFQ` and `T10TGS`, which also appear among the 24 training scenes (validation shares none). The notebook prints the tile lists but never says they overlap, and its own advice is to split by site or tile. The BYOD split is a seeded per-scene shuffle with no group column; the docstring and the interpretation tell the learner to group by fire or tile, but `pairs.csv` has no field to do so.

**Consequence:** The headline 0.8149 includes two test scenes whose tile was seen in training, a mild optimism the learner cannot see; a BYOD learner with several scenes per fire gets exactly the near-duplicate split the notebook warns against.

**Evidence:** Direct execution (P10) on the real `SAMPLE_RECORDS` constants: test tiles also in train `['T10TFQ', 'T10TGS']`, validation none. Documented execution evidence: the recorded Section 4 print lists `T10TFQ` and `T10TGS` among the test tiles. Source inspection of `split_dataset`.

**Recommended correction:** Say in Section 4 which test tiles also occur in training (or print the overlap), and accept an optional `group` column in `pairs.csv` that `split_dataset` keeps within one split.

**Acceptance check:** The notebook prints or states the train/test tile overlap for the sample; a BYOD zip with a `group` column yields splits in which no group spans two roles.

**Spec:** SPL5 (BYOD), SPL10.

### FX-m3 — Minor: the exported `…_sample_pairs.csv` ("the BYOD shape") names files that exist nowhere

**Location:** Section 4 (cell 13, `write_dataset_csv(test_records, …)`); `write_dataset_csv` in `samples.py`; README / `tutorials/README.md` ("a sample pair is written to `outputs/` in that shape").

**Observed issue:** The CSV lists 12 rows with image names `<source_id>_merged.tif` (for example `T10SDH.2020248.v1_merged.tif`). The extracted members are named `subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4_merged.tif`, and `outputs/` holds only one pair named `prithvi_eo_feature_extraction_sample_chip.tif` / `_sample_label.tif`. A zip built from `outputs/` with that CSV fails `load_byod_dataset` on the first row.

**Consequence:** A learner who uses the exported files as the BYOD template ends with a zip the loader refuses with a bare `KeyError` (FX-m1).

**Evidence:** Direct execution (P10): the CSV's names do not match the extracted member names; documented execution evidence: the recorded `outputs/…_sample_pairs.csv` has the same 12 names and `outputs/` the single renamed pair.

**Recommended correction:** Write a `pairs.csv` that names the written sample pair (or write all listed pairs under the listed names), so `outputs/` zipped as-is is a valid BYOD upload.

**Acceptance check:** Zipping the sample files and CSV from `outputs/` and passing the zip to `load_byod_dataset` loads every listed row.

**Spec:** DAT12.

### FX-m4 — Minor: an imagery tutorial never displays an image, reconstruction or probe map

**Location:** Sections 4, 5 and 8 (cells 13, 15, 21).

**Observed issue:** The reconstruction of the first date and its mask, the sample chip and label, and the four probe maps are written as GeoTIFFs only; no tutorial cell plots anything (P1: 0 display calls outside the carried modules).

**Consequence:** The learner is told to compare a patch-level map "next to the pixel labels" and to read a reconstruction against mean fill, but sees only numbers; downloading and opening GeoTIFFs is outside the stated prerequisites.

**Evidence:** Source inspection; P1.

**Recommended correction:** Show a small figure per stage: RGB of the first date, its 75 % mask and reconstruction; one test scene's RGB, label and probe map side by side.

**Acceptance check:** Sections 5 and 8 each render at least one labelled figure inline in Colab.

**Spec:** UX3, UX11.

### FX-m5 — Minor: decision rule, threshold and calibration ownership are not stated in the notebook

**Location:** Sections 7–8 and Interpretation; `tutorials/README.md` Conformance notes ("the notebook says the threshold and its cost trade-off are the deployment's to set").

**Observed issue:** The argmax decision rule appears only inside `predictions.json` (`decision_rule`) and the carried code; no markdown cell says the probe's softmax scores are uncalibrated, that the default rule is argmax with no threshold, or who owns thresholding and calibration for deployment (P1: "threshold", "calibrat" and "deployment" absent from the markdown). The repository README claims the notebook says it.

**Consequence:** A learner reading the probe maps as burn probabilities, or carrying the 0.5 argmax into an operational setting, is not warned; the README overstates what the notebook delivers.

**Evidence:** Source inspection; P1.

**Recommended correction:** Add one paragraph in Section 8: scores are uncalibrated softmax outputs of a linear head, the map is the argmax, and the threshold and its cost trade-off (missed burns vs false alarms) belong to the deployment, which must calibrate on its own data.

**Acceptance check:** The regenerated notebook's markdown states the decision rule, that scores are uncalibrated, and who owns threshold and calibration.

**Spec:** UNC3, UNC4.

### FX-m6 — Minor: interpretation and experiments state sample outcomes without guidance for other data; brace typo in the data contract

**Location:** Interpretation (cell 22: "reaches a patch-level burn-scar IoU near 0.8"; "Optional experiments … set `CLASS_BALANCE = True` and watch precision and recall trade places"); Section 6 markdown ("F1 about 0.77 … against 0.90 for block 12"); Prerequisites ("`{{id, frames}}`", "`{{id, image, label}}`"); generator `tools/notebook_template.py` lines 120, 397, 422.

**Observed issue:** The interpretation is fixed text quoting the sample result; after a BYOD run it still says "near 0.8" and "0.925 on the same scenes". The optional experiments pre-state their outcome ("precision and recall trade places", "F1 about 0.77") and give no rerun scope. The Prerequisites show doubled braces (`{{id, frames}}`), a template-escaping leak.

**Consequence:** A learner cannot tell whether their own result agrees, and the experiments ask them to confirm a given answer rather than predict and explain one.

**Evidence:** Source inspection; P1 (`brace_typo_present`, `interpretation_hardcodes`).

**Recommended correction:** Phrase the interpretation as "on the default sample, …" and add a BYOD-conditional line; turn the experiments into prediction prompts with "re-run Sections 6–8"; fix the escaping in the template.

**Acceptance check:** The regenerated Prerequisites show `{id, frames}` and `{id, image, label}`; each optional experiment names its rerun scope and asks for a prediction; the interpretation distinguishes the sample result from a BYOD run.

**Spec:** GDL8, GDL10, GDL14.

### Suggestions

- **FX-S1:** Regenerate against NOTEBOOK_SPEC 2.2 (the notebook declares 2.0).
- **FX-S2:** Record a Colab run of the notebook; both hosted records are Kaggle T4, while the notebook names Colab as the supported runtime.
- **FX-S3:** Document `DIMER_NOTEBOOK_CI_PREINSTALLED` (read by cell 3) in the notebook (EXE5).
- **FX-S4:** Print per-scene burn IoU for the 12 test scenes beside the pooled number; the notebook already says pooled metrics let large burns dominate.
- **FX-S5:** Print peak GPU memory, to back the "about 2.7 GB" prerequisite (no record states it).
- **FX-S6:** Say that the kept epoch was the last one (epoch 30 of 30; validation loss still falling, 0.2140 → 0.1996 over the last five epochs), so validation-loss selection did not stop early on the sample.

## 4. Readiness

**Needs revision.** No Blocker. Four Majors are open (FX-M1 one-pass `Run all`; FX-M2 BYOD crash and stale exports; FX-M3 stale "frozen" evaluation; FX-M4 guided layer), and the open `MUST`s RUN1, RUN10, ENV6, REL2/REL11, DAT10, DAT12, DAT14, DAT19, SPL5, UNC4 and REL12 fail release under the spec regardless of severity. The repository's `Release-grade` status rests on a two-pass run and should return to `Candidate`. Remaining gates after fixes: a one-pass hosted run of the regenerated blob (Colab preferred, FX-S2), and a hosted BYOD run that reaches export and reload parity (REL12).

## 5. Verified vs inferred

- **Verified by direct execution (CPU, stub encoder, synthetic chips, real example tiles; the notebook's own cells):** generator parity (`--check` OK), validator PASS, 65 offline tests passed / 1 skipped (3 Windows-only failures, unrelated); default cells 13–21 wire up end to end on the stub; BYOD `KeyError: 'source_id'` in Section 8 with the sample `result.json` left in place; Section 5 "frozen zero head" equal to the trained probe after a re-run, on the sample and on BYOD; optional `FEATURE_LAYER`/`CLASS_BALANCE` re-runs correct; BYOD minimum of 7, unactionable `KeyError`/`TiffFileError`, misfiring refusal probes; train/test tile overlap; CSV names vs extracted members.
- **Verified from documented execution evidence (Kaggle T4, this blob):** pass-1 restart error; default-path metrics, history, 4-tensor 16,720-byte artifact and exact reload parity; the recorded tile lists and CSV.
- **Inferred, not executed:** behaviour on real weights of every re-run and BYOD probe above (the mechanisms are in the carried code and do not depend on the encoder); the cancel-upload `StopIteration` (expression only); GPU memory; behaviour on Colab.
- **Most likely to be wrong:** FX-M3's severity. The `baseline_not_burned` column, the cell's assertions and Section 6's epoch 0 stay correct, so a careful learner could spot that the "frozen zero head" is no longer zero; if the fix cycle judges the mislabelled column as localised friction, it is a Minor. The mechanism (Section 5 evaluates the attached probe and labels it the zero head) is certain from source and was reproduced with the actual cells.
