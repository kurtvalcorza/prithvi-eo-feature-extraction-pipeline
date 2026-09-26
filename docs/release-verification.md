# Release verification

`tutorials/prithvi_eo_feature_extraction_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions; the dataset
  revision `1864285e…` is the only other 40-hex commit the documents may name;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `PrithviFeaturePipeline.from_pretrained(weights_dir=..., device=..., report=print)` so the pickle audit and the
  conversion are printed before the model loads, `fetch_sample_dataset` from the pinned cache path,
  `load_byod_dataset`, `dataset_manifest`, `write_sample_pair`, `validate_stacks` on the example tiles,
  `validate_dataset` with the refusal probes, `pipe.embed` and `pipe.reconstruct` on the example stacks,
  `pipe.evaluate` on the zero head and after the probe with the procedural and directional assertions, `pipe.adapt`
  with its explicit hyperparameters and feature layer, `pipe.predict` probe maps, `pipe.save_artifact`,
  `PrithviFeaturePipeline.from_artifact` and the reload-parity assertion, and the provenance fields
  `served_from_pickle: False`, `remote_code_executed: False` and the `data_tarball` record), the nine expected
  `outputs/` paths, the learner-facing statements (the asset is a pickle unpickled once, a foundation model has no
  task head, the linear probe, the mean-fill and majority baselines, the tarball is streamed with no `extractall`,
  split by site or tile, CC BY 4.0) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `huggingface_hub` / `safetensors` / `urllib` / `terratorch` / `tarfile.open(` / `Unpickler` / `PrithviMAE(` /
  `forward_features(` use or `torch.load(` / `pickle.load` **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load` or `torch.load(` without `weights_only=True` anywhere, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub masked autoencoder, skipped
without torch), `tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`,
`tests/test_model_backed.py` (skipped without the staged weights or `terratorch`); crafted pickles, temporary
manifests, synthetic scenes and stacks, a synthetic tarball with a decoy member and an injected fetcher, no weights).
These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/prithvi-eo-2.0-300m/` or the data cache `weights/hls-burn-scars/` (the standalone path writes
   the manifest itself, stages all seven listed files from the Hub, audits and converts the checkpoint, fetches the
   2.6 GB tarball from the Hub dataset at its immutable revision, hashes it and streams out the 88 pinned members, so
   neither directory may be seeded); the runtime needs about 6 GB of free disk for the snapshot, the converted file,
   the tarball and the extracted members;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `EPOCHS = 30`, `LEARNING_RATE = 1e-2`, `BATCH_SIZE = 4096`, `FEATURE_LAYER = 11`,
   `CLASS_BALANCE = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `terratorch==1.2.13`, `tifffile==2026.9.15`,
   `numpy==2.5.3`, `safetensors==0.8.0`, `huggingface-hub==1.32.0` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `PrithviFeaturePipeline`, `audit_pickle`, `convert_model`,
     `build_model`, `verify_snapshot`, `verify_converted`, `stage_missing_files`, `validate_inputs`,
     `validate_dataset`, `validate_stacks`, `patch_labels`, `read_chip`, `read_mask`, `read_stack`, `fetch_tarball`,
     `extract_pinned_members`, `fetch_sample_dataset`, `load_byod_dataset`, `write_sample_pair`,
     `write_dataset_csv`, `dataset_manifest`, `segmentation_metrics`, `majority_baseline`) with no import of the
     repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 7 entries fetched from `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` at the immutable revision and
     `verify_snapshot` reporting 7 verified files;
   - the model cell printing the **conversion record** with the static audit (globals `collections.OrderedDict` /
     `torch.FloatStorage` / `torch._utils._rebuild_tensor_v2`, 0 violations, audit digest `e7b998d0…`), the
     checkpoint record (398 tensors: 294 encoder + 104 decoder) and the converted file (`15b8ed6d…`,
     1,326,543,456 bytes), then the load report on `cuda` with source "converted from the manifest-verified source
     checkpoint";
   - the tarball fetched and hashed (`4e6f99a7…`, 2,645,552,531 bytes), the 88 pinned members extracted under
     `weights/hls-burn-scars/chips/`, the dataset manifest with 24 / 8 / 12 scenes, burn fractions about
     0.15 / 0.12 / 0.21 at the pixel level and patch-positive fractions about 0.15 / 0.12 / 0.21, the written sample
     pair and `outputs/prithvi_eo_feature_extraction_sample_pairs.csv`, the five validated example stacks
     ((6, 4, 448, 560) and four (6, 1, 448, 560)), and three refusals (five-band scene, unknown label class, five-date
     stack);
   - `pipe.embed` on the example stacks (token grids [4, 28, 35] and [1, 28, 35]; cosine similarities 0.98–0.99
     between the four dates' mean-token embeddings) and `outputs/prithvi_eo_feature_extraction_embeddings.json`
     written; `pipe.reconstruct` reporting a masked-patch MSE below the mean-fill baseline for the series (about
     0.077 vs 0.129) and for every date, with the RGB reconstruction and mask of the first date written;
   - the majority baseline and the zero head on the test scenes (accuracy ≈ 0.79, burn-scar IoU 0) and the
     validation scenes (burn-scar IoU 0);
   - `pipe.adapt` printing epoch 0 as the zero head, 2,050 trainable parameters, 330,419,712 frozen, about 24,400
     training patches (positive fraction ≈ 0.15), 180 steps, and a 30-epoch history with validation loss ≈ 0.693 →
     ≈ 0.20 at the kept epoch and validation burn-scar F1 ≈ 0.76;
   - `pipe.evaluate` on the test scenes with the paired comparison and
     `outputs/prithvi_eo_feature_extraction_evaluation_report.json` written (the cell asserts the kept epoch's
     validation loss is no higher than the zero head's, that the validation burn-scar IoU matches the history within
     0.01, and that the probe's test burn-scar IoU is above the baseline's 0 — on the sample ≈ 0.81, F1 ≈ 0.90,
     accuracy ≈ 0.95 vs 0.79);
   - four test scenes mapped by the probe with `outputs/prithvi_eo_feature_extraction_predictions.json` and one
     32 × 32 map GeoTIFF per scene written;
   - `pipe.save_artifact` writing `outputs/prithvi_eo_feature_extraction_adapter/{adapter.safetensors,manifest.json}`
     (4 tensors, about 17 KB), and `PrithviFeaturePipeline.from_artifact` reloading it with held-out metrics and
     score maps matching the probe (the cell asserts a burn-scar IoU difference below 10⁻³ and a maximum score
     difference below 10⁻³);
   - `outputs/prithvi_eo_feature_extraction_result.json` written with `NOTEBOOK_SOURCE`, the model identity, the
     provenance block (`served_from_pickle: false`, `remote_code_executed: false`, the audit digest, the converted
     digest, the `data_tarball` record), the runtime versions, the reconstruction table, the comparison and the reload
     parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the data cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `prithvi_eo_feature_extraction_colab.ipynb` (`E2E`) | `e611e93` / `6a12fd2e` | 2026-09-25 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-eo-feature-extraction` v2`; image `torch 2.10.0+cu128` before the pinned install, pinned `torch 2.14.0+cu130` / `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | **PASSED** — 10/10 code cells ok (1 restart after install cell); 110 files, 5611 MB fetched into a clean runtime; comparison test patch-level burn-scar IoU / F1 (majority baseline = zero head 0 / 0, accuracy 0.7865): probe 0.8149 / 0.8980, accuracy 0.9539; reload parity exact; identical probe metrics to the 2026-09-20 run, as expected (no pinned chip is affected by the idempotent-scaling fix); evidence archived under `.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-eo-feature-extraction/v2/evidence/` in the workspace |
| `prithvi_eo_feature_extraction_colab.ipynb` (`E2E`) | `81bebf7` / `28e2298e` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-eo-feature-extraction` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `timm 1.0.26`, `lightning 2.6.6`, `tifffile 2026.9.15`, `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | **PASSED** — 10/10 code cells ok (1 restart after install cell); 110 files, 5611 MB fetched into a clean runtime (Hub snapshot + the pinned data); comparison test patch-level burn-scar IoU / F1 (majority baseline = zero head 0 / 0, accuracy 0.7865): probe 0.8149 / 0.8980, accuracy 0.9539, precision 0.8504, recall 0.9512 (best epoch 30, validation loss 0.6931 → 0.1996, validation F1 0.7577); reconstruction MSE vs mean-fill baseline 0.0773 vs 0.1294 on the four-date series; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-prithvi-eo-feature-extraction/v1/evidence/` in the workspace |
| `prithvi_eo_feature_extraction_colab.ipynb` | generated, pre-commit | 2026-09-20 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/prithvi_eo_feature_extraction_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/prithvi_eo_feature_extraction_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-25 | `e611e93` / `6a12fd2e` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-eo-feature-extraction` v2`; image `torch 2.10.0+cu128` before the pinned install, pinned `torch 2.14.0+cu130` / `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 437.5 s | **PASSED** — 10/10 code cells ok (1 restart after install cell); 110 files, 5611 MB fetched into a clean runtime; comparison test patch-level burn-scar IoU / F1 (majority baseline = zero head 0 / 0, accuracy 0.7865): probe 0.8149 / 0.8980, accuracy 0.9539; reload parity exact; identical probe metrics to the 2026-09-20 run, as expected (no pinned chip is affected by the idempotent-scaling fix); evidence archived under `.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-eo-feature-extraction/v2/evidence/` in the workspace |
| 2026-09-20 | `81bebf7` / `28e2298e` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-eo-feature-extraction` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `timm 1.0.26`, `lightning 2.6.6`, `tifffile 2026.9.15`, `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the checkpoint audited and converted in the notebook, the portraits rendered and the photographs fetched by the notebook | 392.9 s | **PASSED** — 10/10 code cells ok (1 restart after install cell); 110 files, 5611 MB fetched into a clean runtime (Hub snapshot + the pinned data); comparison test patch-level burn-scar IoU / F1 (majority baseline = zero head 0 / 0, accuracy 0.7865): probe 0.8149 / 0.8980, accuracy 0.9539, precision 0.8504, recall 0.9512 (best epoch 30, validation loss 0.6931 → 0.1996, validation F1 0.7577); reconstruction MSE vs mean-fill baseline 0.0773 vs 0.1294 on the four-date series; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-prithvi-eo-feature-extraction/v1/evidence/` in the workspace |
| 2026-09-20 | generated, pre-commit | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti, `terratorch 1.2.13`, `timm 1.0.29`, `lightning 2.6.6`) | Default sample path (stage → verify → load the already-converted file → tarball hash → pinned-member extraction → validate → refusal probes → embeddings + reconstruction + zero-head baseline → linear probe → evaluate → probe maps → export → reload); the Hub files, the converted safetensors and the tarball were pre-staged, so `stage_missing_files` fetched 0 of 7 entries and the tarball was hashed and streamed from the cache | 89.5 s | **PASSED** — 10/10 code cells; probes refused; reconstruction MSE 0.077 vs mean-fill 0.129 on the four-date series (0.054–0.105 vs 0.069–0.172 per date); test patch-level burn-scar IoU 0 (zero head = majority baseline, accuracy 0.787) → 0.815 (probe; F1 0.898, precision 0.850, recall 0.951, accuracy 0.954), validation F1 0.758, best epoch 30 (validation loss 0.693 → 0.200), adaptation 4.8 s / 180 steps; probe 16,720 bytes; reload parity identical (positive_iou_diff 0.0, max_abs_score_diff 0.0) |

## Supplemental EO and climate workshop (separate status)

`tutorials/DIMER_AI_for_Earth_Observation_and_Climate_Applications_Workshop.ipynb` is a `MULTI-CAPABILITY` / `WORKSHOP`
notebook (DIMER Notebook Specification 2.1) across four live Prithvi-EO-2.0 profiles, generated by
`tools/eo_workshop/build_workshop.py`. It carries its own status, **Candidate**, independent of this repository's `E2E`
tutorial above. Evidence so far: a CPU offline `Run all` on 2026-09-25 (186 s, Hugging Face cache seeded from the local
repository clones) — a pre-flight, not a clean-runtime execution, and of the earlier layout that installed the pins into
the kernel (with one automatic restart on hosted runtimes). Since 2026-09-26 the notebook installs the pins into an
isolated environment (`dimer_eo_env/`, created with `uv`) and routes every code cell after the two kernel cells to one
worker process there, so a single `Run all` completes without a restart; the bridge is unit-tested in
`tests/test_eo_workshop.py` and was exercised locally under IPython 7.34 / ipykernel 6.17 and IPython 9.17 / ipykernel
7.3 with NumPy 2.1.3 pre-imported in the kernel (output, figures, errors, interrupt, worker crash). Its release gate is one clean Kaggle Tesla T4
`Run all` of the exact committed blob, recorded here with the per-task metrics against their baselines; until then its
registry row stays Candidate.

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| — | — | — | — | No clean-runtime execution recorded yet. | — |

## Current status

**Release-grade.** The `E2E` notebook blob `6a12fd2e` (committed at `e611e93`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-25 (10/10 ok (1 restart after install cell), 437.5 s, 110 files, 5611 MB fetched and digest-verified inside the notebook, the checkpoint converted in the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.
