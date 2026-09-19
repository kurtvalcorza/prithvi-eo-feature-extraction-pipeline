# Weight provenance, the pickle audit, the conversion, the pinned dataset tarball and DIMER hosting

This repository pins **one** model snapshot with its own `dimer-base-manifest.json` and **one** dataset tarball. The checkpoint is a pickle, which this pipeline audits and converts but never serves; the tarball is streamed once for exactly the pinned members and never `extractall`-ed.

## Prithvi-EO-2.0-300M weights

- Upstream: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`
- Immutable revision: `9eb1b1102806593963daa333bcc491b1c6f8562f` (2025-10-09, "Update README.md"); the checkpoint was first published in commit `ebfff1d8` (2024-12-03, "added model weights") and the model code in the repository was last changed in `b2f2520a` (2025-03-24, "Update model code" — that module is never imported here); the checkpoint bytes are identical at every revision since.
- Source format: `Prithvi_EO_V2_300M.pt` — torch zip archive (`archive/data.pkl`) holding an `OrderedDict` state dict of 398 float32 tensors: 294 under `encoder.` (the ViT-L masked-autoencoder encoder: 3-D patch embedding, CLS token, fixed positional embedding, 24 blocks, final norm) and 104 under `decoder.` (mask token, fixed positional embedding, decoder embedding, 8 blocks, norm and the prediction head). No prefix, no hyper-parameters, no optimiser state.
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter).
- Local layout: `weights/prithvi-eo-2.0-300m/` holds the 7 manifest entries (the checkpoint, upstream `README.md`, `config.json`, the four example tiles `examples/Mexico_HLS.S30.T13REM.2018{026,106,201,266}T*.v2.0_cropped.tif`; 1,338,724,072 bytes total) with byte size and SHA-256 for each, plus the converted file described below. `verify_snapshot()` in `src/prithvi_eo_feature_extraction_pipeline/pipeline.py` checks the manifest entries, asserts the checkpoint's digest against the package constant, and checks the converted file against its pinned digest when present.
- Cross-check: the manifest's checkpoint digest `faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7` equals the `oid sha256` of the Hub LFS pointer at the pinned revision.

## What the pickle would execute, and how it is audited

Under the fleet asset specification (§11) a pickle is executable serialization. `audit_pickle()` disassembles the file with `pickletools.genops` — every `.pkl` inside the torch zip archive — collects every `GLOBAL` / `STACK_GLOBAL` it would import, and refuses anything outside the allow-list, executing nothing:

| File | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|
| `Prithvi_EO_V2_300M.pt` | `collections.OrderedDict`, `torch.FloatStorage`, `torch._utils._rebuild_tensor_v2` | exactly those three | `e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932` |

The audit reports 0 violations and its digest is pinned in `PICKLE_AUDIT_SHA256` (the digest covers the sorted set of global names; this checkpoint names three, one fewer than the Lightning checkpoints of the flood and burn-scar rows, which also carry `torch.LongStorage`); `convert_model()` refuses a file whose audit digest differs. Tests craft a torch archive carrying `os.system` and a plain pickle of a `complex` number, and assert that the audit refuses each before anything is constructed.

An allow-list bounds what the unpickler can name; the loader below bounds what it can construct. The digest pins tie the audited bytes to the loaded bytes, and the unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs size check → SHA-256 check against the package constant → static audit and audit-digest check, and only then:

- `torch.load(map_location="cpu", weights_only=True)` — torch's restricted unpickler, which constructs tensors and containers and nothing else — must return a dict of exactly 398 tensors;
- the tensors are loaded with `strict=True` into `terratorch.models.backbones.prithvi_mae.PrithviMAE(img_size=224, patch_size=(1, 16, 16), num_frames=4, in_chans=6, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, coords_encoding=[], coords_scale_learn=False, mask_ratio=0.75, norm_pix_loss=False)` — the pinned `config.json`'s `pretrained_cfg` — and the model's own state dict is saved as safetensors.

Serving file (both identities recorded, `derived_from_sha256` = the source digest above):

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `prithvi-eo-2.0-300m.safetensors` | 1,326,543,456 | 398 (331,625,472 elements: 330,419,712 parameters + 1,205,760 elements of fixed sin/cos positional-embedding buffers) | `15b8ed6dacf8b3ca02c542bb8eaffc92c7a7addc5351b3d967873ad3f73a24b0` | no (regenerated) |

The conversion is deterministic: the digest was reproduced on two consecutive build conversions and by the executed tutorial notebook, which converts the file it downloads. `verify_converted()` checks size and digest; `from_pretrained()` loads the safetensors with `strict=True` and asserts the parameter count. The strict load matched every key with 0 missing, 0 unexpected and 0 shape mismatches.

## Fidelity

No upstream regression fixture is published for this checkpoint. The evidence is the strict key-and-shape match against the architecture built from the pinned configuration under `terratorch 1.2.13`, and the pretraining task itself: on the four upstream example tiles (HLS S30 T13REM, Mexico, 2018, 448 × 560 pixels), the masked-patch reconstruction MSE at a 75 % mask ratio is 0.077 for the four-date series against 0.129 for a mean-fill baseline, and 0.054 / 0.105 / 0.095 / 0.076 against 0.069 / 0.128 / 0.172 / 0.142 per date (standardised units, seed 0; `MODEL_CARD.md`, *Runtime*). Behaviour under the exact TerraTorch version the authors used was not measured; the upstream `inference.py` (which imports the Hub-hosted `prithvi_mae.py`) was not run.

## Runtime facts

- The model is float32 as shipped and runs in float32 on CPU and CUDA (no autocast; the token embeddings are the product and are kept exact); `embed` and `reconstruct` run under `torch.inference_mode()` one record at a time and move results to the CPU; `adapt` trains only the 2,050-parameter probe, on cached tokens, and never stores a gradient for the encoder.
- Inputs are standardised with the six-band means and standard deviations of the pinned `config.json` (HLS digital-number scale): a chip in [0, 1] reflectance is multiplied by 10 000 first, a chip arriving as reflectance × 10 000 is left as is (values above 1 are recognised and scaled by `CONSTANT_SCALE = 1e-4` on validation, then back), no-data (0 or −9999) is replaced by 0 — the same preprocessing as the upstream inference script. Stacks may hold 1–4 dates; with fewer than four the 3-D positional embedding is interpolated by `terratorch`.
- Embeddings: `embed` returns the CLS token and the mean of the patch tokens of the final, normalised encoder layer (every patch visible); `adapt` reads the patch tokens of encoder block `FEATURE_LAYER = 11` (zero-based; block 12 of 24) through `forward_features`, which returns every block's output in spatial order. The encoder's masking path (`encoder(x, mask_ratio=0.0)`) returns tokens in shuffled order and is not used for features.
- `terratorch` pulls `torchgeo`, `lightning`, `timm`, `segmentation-models-pytorch`, `albumentations`, `rasterio`, `geopandas`, `lightly`, `h5py`, `tensorboard` and others; the pipeline uses `terratorch.models.backbones.prithvi_mae.PrithviMAE` only, and reads scenes with `tifffile` (no rasterio, no georeferencing).

## The tutorial data: the pinned HLS Burn Scars tarball

`samples.py` fetches `hls_burn_scars.tar.gz` from the Hugging Face dataset `ibm-nasa-geospatial/hls_burn_scars` at the immutable revision `1864285e25010d346a842e4f068b1a1d4248ed6d` (2,645,552,531 bytes, SHA-256 `4e6f99a75cb2c500547b20662a15cbd531dc421376f815e91846ea542798e8e6`, hashed once per `fetch_tarball` call and refused on a mismatch), then streams through it **once** with `extract_pinned_members` and copies out exactly the 88 pinned members — 44 six-band float32 `_merged.tif` scenes (6,295,168 bytes each) and their single-band `.mask.tif` masks (525,272 bytes each; values −1 / 0 / 1), 300,099,360 bytes uncompressed — each pinned by member path, byte size and SHA-256 in `SAMPLE_RECORDS`. No `extractall`, no path taken from the archive: every member is written under its base name in `weights/hls-burn-scars/chips/` (git-ignored). The scenes, their roles (24 train / 8 validation / 12 test from the burn-scar model repository's splits) and the selection rule (≥ 60 % valid, ≥ 3 % burn, fixed seed, 2026-09-19) are the same as the fleet's burn-scar row, whose `samples.py` this module is; the pixel masks are reduced to 16 × 16 patch labels by `pipeline.patch_labels` for the probe (a patch is labelled when at least half of its pixels are, positive when at least half of its labelled pixels are burn scar). The dataset is CC BY 4.0 (NASA IMPACT / UAH).

## Files deliberately not staged

The upstream repository at the pinned revision also carries `prithvi_mae.py` (the authors' model module, superseded by the `terratorch` package), `inference.py` (their rasterio inference script), `requirements.txt` and two `assets/*.png` figures; none is listed in the manifest, fetched or executed. Of the tarball, only the 88 pinned members are ever written to disk.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the licence and notices. DIMER may host the converted safetensors in its model store under those terms; it is derived from, and recorded beside, the unmodified upstream checkpoint.
- Upload set: `prithvi-eo-2.0-300m.safetensors`. **The `.pt` file must not be uploaded** — a profile that carries it would reintroduce the executable-serialization boundary this conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path; the model class comes from `terratorch==1.2.13` on PyPI, the served state dict is safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified file without the manifest or the checkpoint.
- Serving shape: an embedding needs the 1.33 GB weights and one (6, T, H, W) stack; on an RTX 5070 Ti laptop GPU a 512 × 512 single-date scene embeds in about 0.05 s (2.7 GB of GPU memory) and a four-date 448 × 560 series in about 0.6 s; a laptop CPU embeds a scene in about 2 s. A probed profile needs the weights plus a 17 KB probe artifact.
- One review item is open: whether the one-time restricted unpickle (in the build and, for the tutorial, in the runtime) meets the DIMER deserialization-trust bar or whether DIMER hosts only maintainer-converted files. The served artifact is the same file either way.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
