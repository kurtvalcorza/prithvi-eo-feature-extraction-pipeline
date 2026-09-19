# Prithvi-EO Feature Extraction Pipeline

DIMER-oriented pipeline for **Prithvi-EO-2.0-300M** (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`, the NASA/IBM Earth-observation foundation model — a ViT-L masked autoencoder for Harmonized Landsat Sentinel-2 imagery), pinned to an immutable Hugging Face revision. The repository exposes token and scene embeddings of six-band HLS reflectance stacks (1–4 dates), masked-patch reconstruction scored against a mean-fill baseline, a labelled-chip contract with explicit ceilings, a bounded **linear-probe** contract on frozen patch tokens with a portable safetensors artifact, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`
- Revision: `9eb1b1102806593963daa333bcc491b1c6f8562f`
- Source asset: `Prithvi_EO_V2_300M.pt` (1,326,658,396 bytes, SHA-256 `faab2c8b…`) — a torch state-dict pickle (encoder + decoder, 398 tensors), converted once to safetensors and never served (see below)
- Upstream weight license: Apache-2.0
- Upstream task: self-supervised masked-autoencoder pretraining on 4.2 M HLS samples (six bands: blue, green, red, narrow NIR, SWIR 1, SWIR 2; 3-D patches of 1 × 16 × 16 over up to four dates); the model is a backbone, and the fleet's flood and burn-scar rows are fine-tunes of it
- Runtime: `terratorch==1.2.13` + `torch==2.14.0` + `torchvision` + `tifffile` — the model class comes from PyPI, **no Hub-hosted code and no served pickle**
- Repository adaptation: **E2E** (a bounded linear probe — a 1024 → 2 head on the frozen patch tokens of encoder block 12 — on labelled scenes at the patch level, with a portable safetensors artifact)

## Two things to know before you start

**The checkpoint is a pickle, and the pipeline converts it once.** `audit_pickle` lists every global the torch archive's pickle would import with `pickletools` (no execution) and refuses anything outside `collections.OrderedDict` / `torch._utils._rebuild_tensor_v2` / `torch.FloatStorage`; `convert_model` unpickles it once through `torch.load(weights_only=True)`, loads the 398 tensors strictly into `terratorch`'s `PrithviMAE` built from the pinned configuration, and writes `prithvi-eo-2.0-300m.safetensors` (1,326,543,456 bytes, SHA-256 `15b8ed6d…`), the only file the model is ever loaded from.

**A foundation model has no task head, so the adaptation is a linear probe.** `adapt()` runs the frozen encoder once per labelled scene, takes the patch tokens of encoder block 12 (`FEATURE_LAYER = 11`, chosen on the validation split during the build — the middle blocks separate burned from unburned patches better than the last), and trains a 2,050-parameter head with validation-loss epoch selection; the untrained (zero) head is exactly the majority baseline. On 12 held-out HLS Burn Scars scenes the probe reaches a patch-level burn-scar IoU of 0.81 (F1 0.90) against 0 for the baseline — an embedding-quality check at 16-pixel resolution, not a replacement for the burn-scar row's pixel-level fine-tune (0.93).

## Quick start

```python
from prithvi_eo_feature_extraction_pipeline import PrithviFeaturePipeline, fetch_sample_dataset, read_stack

pipe = PrithviFeaturePipeline.from_pretrained()          # verifies the snapshot, audits + converts the pickle once, loads safetensors
tiles = sorted(pipe.weights_dir.glob("examples/*.tif"))   # the four upstream Mexico tiles (T13REM, 2018, four dates)
series = {"id": "mexico", "frames": [str(t) for t in tiles]}
print(pipe.embed([series])["embeddings"][0]["mean"].shape)     # (1024,) mean patch-token embedding; "cls" is the CLS token
print(pipe.reconstruct([series])["results"][0])               # masked-patch MSE at a 75 % mask vs the mean-fill baseline
splits = fetch_sample_dataset()                                # 24 / 8 / 12 pinned HLS Burn Scars scenes, extracted from the digest-verified tarball
print(pipe.evaluate(splits["test"])["model"])                  # the zero head: the majority baseline at the patch level
pipe.adapt(splits["train"], splits["validation"])              # linear probe on frozen block-12 tokens, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["model"])                  # the probe, same scenes
pipe.save_artifact("outputs/probe")
```

`embed()` and `reconstruct()` take records `{id, frames}` — a (6, T, H, W) reflectance stack with 1–4 dates and sides that are multiples of 16 in [64, 1024], or 1–4 GeoTIFF paths in chronological order (`read_stack` builds the array). `predict()`, `evaluate()` and `adapt()` take `{id, image, label}` — a (6, 512, 512) chip (or a GeoTIFF path; 13-band Sentinel-2 L1C files are reduced to the six HLS-equivalent bands) and a (512, 512) mask with 0 / 1 / −1, reduced to 32 × 32 patch labels by `patch_labels`. HLS scenes are already reflectance in [0, 1]; values above 1 are scaled by 10⁻⁴, no-data (0 or −9999) is replaced by 0, and validation is structural: nothing checks that the bands are the right six in the right order or that the reflectance is corrected.

## Weights layout

```
weights/prithvi-eo-2.0-300m/   README.md  config.json  Prithvi_EO_V2_300M.pt  examples/*.tif  dimer-base-manifest.json
                               prithvi-eo-2.0-300m.safetensors  (git-ignored, converted)
weights/hls-burn-scars/        hls_burn_scars.tar.gz (2.6 GB, digest-pinned) and chips/ with the 88 extracted members (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of all 7 manifest entries and of the converted file when present), converts the pickle when the safetensors file is absent, and refuses on the first mismatch. With `require_source=False` the digest-verified converted file is accepted without the checkpoint — the DIMER-hosted shape. `docs/WEIGHTS.md` records the provenance, the audit, the conversion and the DIMER hosting notes.

## Sample data

`fetch_sample_dataset()` assembles 44 labelled HLS scenes of the HLS Burn Scars dataset — 24 from the training split, 8 from the validation split and 12 from the test split that the upstream authors publish beside the burn-scar fine-tune, drawn with a fixed seed from the scenes whose mask is at least 60 % valid and 3 % burn scar. `fetch_tarball` downloads the 2.6 GB `hls_burn_scars.tar.gz` from the Hugging Face dataset `ibm-nasa-geospatial/hls_burn_scars` at an immutable revision and refuses it on a size or SHA-256 mismatch; `extract_pinned_members` streams through it once and copies out exactly the 88 members pinned by path, byte size and SHA-256 in `SAMPLE_RECORDS` — each refused on its own mismatch, written under its base name, no `extractall`. The four example tiles that ship in the model snapshot are the label-free embedding sample. The dataset is CC BY 4.0 (NASA IMPACT / UAH).

## Probe artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the probe's weight and bias plus the feature mean and standard deviation, about 17 KB) and a `manifest.json` recording the artifact format, the exact base model id and revision, the converted-base digest, the adaptation scope and feature layer, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `PrithviFeaturePipeline.from_artifact(dir)` re-verifies the base file, checks the manifest, scope, layer and digest before deserialising, rebuilds the model and attaches the probe.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: crafted pickles, temporary manifests, synthetic scenes and stacks, a synthetic tarball with a decoy member, an injected fetcher and a stub masked autoencoder with a 24-layer `forward_features`, never the weights or `terratorch`; `tests/test_model_backed.py` runs the real converted weights when they and `terratorch` are present (strict load, embeddings of the example tiles, reconstruction against the baseline, a short probe, reload parity) and skips otherwise. The model-backed smoke is recorded in `MODEL_CARD.md` (*Runtime*).

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/blob/main/tutorials/prithvi_eo_feature_extraction_colab.ipynb)

`tutorials/prithvi_eo_feature_extraction_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It needs a GPU runtime. It downloads and converts the pinned checkpoint in the runtime (the audit and conversion records are printed before the model loads), embeds and reconstructs the example tiles, fetches the pinned tarball and extracts the pinned scenes, and runs the sample path: validation, the majority baseline, the linear probe, held-out evaluation, probe maps, artifact export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Candidate** — the `E2E` notebook has executed top-to-bottom on the local pre-flight harness only (WSL, RTX 5070 Ti, weights and tarball pre-staged; 10/10 cells, 89.5 s); the clean-runtime Kaggle execution that promotes it is pending and will be recorded in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but are not the evidence; the hosted run is.

## Licensing

- Upstream weights: Apache-2.0 (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`; the Prithvi-EO-2.0 code and TerraTorch are Apache-2.0), staged from the pinned revision and converted, not modified, into the served safetensors.
- Tutorial data: HLS Burn Scars (NASA IMPACT / University of Alabama in Huntsville, CC BY 4.0, attribution required); fetched at run time as a digest-pinned tarball, never committed.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
