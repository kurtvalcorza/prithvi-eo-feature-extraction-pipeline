"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E feature-extraction workflow: the pinned Prithvi-EO-2.0-300M masked autoencoder
(a torch pickle) is digest-verified, statically audited and converted once into safetensors, the four upstream HLS
example tiles are embedded as a time series and reconstructed under a 75 % mask against a mean-fill baseline, 44
labelled HLS scenes are extracted from the digest-pinned HLS Burn Scars tarball and assigned the model repository's
roles, a bounded linear probe on frozen patch tokens is trained in the kernel, the held-out scenes are scored at the
patch level against the majority baseline, embeddings and probe maps are exported, and the probe artifact is reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "prithvi-eo-feature-extraction-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/prithvi_eo_feature_extraction_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ibm--nasa--geospatial%2FPrithvi--EO--2.0--300M-ffcc4d?style=flat",
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-NASA--IMPACT%2FPrithvi--EO--2.0-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/NASA-IMPACT/Prithvi-EO-2.0",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2412.02732-b31b1b.svg", "https://arxiv.org/abs/2412.02732"),
]

TEMPLATE = {
    "package": "prithvi_eo_feature_extraction_pipeline",
    "repo_name": REPO,
    "stem": "prithvi_eo_feature_extraction",
    "notebook_name": "prithvi_eo_feature_extraction_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime installs the pinned dependencies (torch, torchvision, terratorch and its "
        "stack, tifffile, numpy, safetensors, huggingface-hub), stages and digest-verifies the pinned Prithvi-EO-2.0-300M snapshot "
        "(1.34 GB: the checkpoint and the four upstream example tiles) from the Hub, statically audits the pickle against an "
        "allow-list, converts it once into safetensors with a pinned digest, builds the masked autoencoder from the installed "
        "`terratorch` package and loads it strictly, embeds the four example tiles as one time series and as single frames and "
        "reconstructs them under a 75 % mask against a mean-fill baseline, fetches the digest-pinned HLS Burn Scars tarball (2.6 GB, no "
        "credential) and extracts exactly the 44 pinned scenes and masks, validates them and assigns the model repository's roles "
        "(24 training, 8 validation, 12 test), scores the majority baseline on the held-out scenes at the patch level, trains a bounded "
        "linear probe on the frozen patch tokens, scores the same scenes again, exports the embeddings and the probe maps, exports the "
        "probe as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On a T4 the model time is under a minute after the downloads; the terratorch install and the tarball are "
        "the slowest steps."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled scenes as a zip holding `pairs.csv` (columns `id`, `image`, `label`) beside six-band 512 × 512 GeoTIFF chips "
        "(blue, green, red, narrow NIR, SWIR 1, SWIR 2 — surface reflectance in [0, 1] or × 10 000) and single-band label rasters "
        "(0 = negative class, 1 = positive class, −1 = no data); at least four chips with some positive pixels. Your chips are split by "
        "seed into training, validation and test sets and flow through the same contract — validation, majority baseline, probe "
        "training, held-out evaluation, probe maps, artifact export and reload parity. The expected schema, the ceilings and the privacy "
        "guidance are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "pipeline_class": "PrithviFeaturePipeline",
    "model_load": "PrithviFeaturePipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "prithvi-eo-2.0-300m",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "timm", "lightning", "tifffile"],
    "title": "Prithvi-EO-2.0-300M embeddings — DIMER E2E feature-extraction and linear-probe tutorial (standalone)",
    "badges": BADGES,
    "capability": "token and scene embeddings of six-band HLS reflectance stacks (1–4 dates) with the Prithvi-EO-2.0-300M masked autoencoder, masked-patch reconstruction against a mean-fill baseline, and a bounded linear probe on frozen patch tokens for labelled scenes, scored at the patch level against the majority baseline",
    "intro": (
        "Prithvi-EO-2.0 (Szwarcman et al., 2024) is NASA and IBM's foundation model for Harmonized Landsat Sentinel-2 imagery: a "
        "ViT-L masked autoencoder with 3-D patch embeddings over (time, height, width) cubes of 1 × 16 × 16, pretrained on 4.2 M "
        "multispectral samples by reconstructing 75 % masked patches. The checkpoint packaged here is the 300 M-parameter variant "
        "without temporal or location embeddings — the encoder (24 blocks of width 1024) and the reconstruction decoder (8 blocks of "
        "width 512) — the base that the fleet's flood and burn-scar rows were fine-tuned from.\n\n"
        "Two things about this row are handled in the open. **The upstream asset is a pickle** — a torch state dict. Section 3 downloads "
        "and digest-verifies it, statically lists every global the pickle would import (a state dict of tensors and nothing else), refuses "
        "anything outside that allow-list, unpickles it exactly once through torch's weights-only loader, and writes a safetensors file "
        "whose digest is pinned in the carried module; the model you run is built from the installed `terratorch` package and loads that "
        "file strictly — the `prithvi_mae.py` module the Hub repository ships is never imported. **A foundation model has no task head**, "
        "so the adaptation contract here is a **linear probe**: Section 6 trains a 1024 → 2 head on the frozen patch tokens of encoder "
        "block 12 (zero-based 11) for patch-level burn-scar labels derived from the HLS Burn Scars masks — the encoder is run once per "
        "scene and stores no gradient, the probe is 2,050 parameters — and the paired comparison is against the majority baseline, which "
        "is exactly what the untrained (zero) head predicts. The dataset ships as one 2.6 GB tarball, so Section 4 pins the tarball by "
        "size and digest, streams through it once and copies out exactly the 88 pinned members (each pinned again by size and digest, no "
        "`extractall`, no paths taken from the archive), and leaves the other 1,522 alone."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify a pickled "
        "checkpoint, read its static audit and see it converted into safetensors; embed a four-date HLS time series and read the "
        "CLS and mean-token embeddings; read the masked-reconstruction error against a mean-fill baseline; extract pinned members from a "
        "digest-verified tarball and validate real labelled multispectral scenes with an ignore class; derive patch-level labels from "
        "pixel masks; train a bounded linear probe on frozen tokens with explicit hyperparameters and validation-loss epoch selection; "
        "compare the probe with the majority baseline on the same held-out scenes; export embeddings and probe maps; and export a "
        "safetensors probe that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "fine-tuning of the encoder or the decoder, pixel-level segmentation (the probe predicts one class per 16 × 16 patch), the "
        "temporal/location-embedding (`TL`) variants, tiling of scenes larger than 1024 pixels, atmospheric correction, cloud masking, "
        "the GEO-bench scores of the paper, and any claim that a 44-scene sample stands in for an operational evaluation. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12): the ViT-L encoder runs in float32 and needs about 2.7 GB of GPU memory for a 512 × 512 scene; on CPU one scene takes about two seconds to embed and the default path would take several minutes. About 6 GB of disk is needed for the checkpoint, its conversion and the tarball; the `terratorch` install pulls torchgeo, lightning and their dependencies and takes several minutes.",
        "- **Knowledge:** what a multispectral surface-reflectance scene is (bands, scaling, no-data), what a masked autoencoder reconstructs, what a token embedding and a linear probe are, and how IoU, precision and recall are read against a majority baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; `terratorch` is installed from PyPI at a pinned version.",
        "- **Data contract:** an embedding record is `{{id, frames}}` — a (6, T, H, W) reflectance stack with 1–4 dates and sides that are multiples of 16 in [64, 1024] (or 1–4 GeoTIFF paths in chronological order); a labelled record is `{{id, image, label}}` — a (6, 512, 512) chip (or a GeoTIFF; 13-band Sentinel-2 L1C files are reduced to the six HLS-equivalent bands) with values in [0, 1] or × 10 000, no-data 0 or −9999, and a (512, 512) mask with 0 / 1 / −1. Validation is structural: nothing checks that the bands are the right six in the right order, that the reflectance is corrected, or that the label belongs to the scene.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — commercial imagery under licence or unreleased assessments are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches one pinned object — the 2.6 GB `hls_burn_scars.tar.gz` of the Hugging Face dataset `ibm-nasa-geospatial/hls_burn_scars` at an immutable revision — over HTTPS, digest-verified before any member is read; the dataset is CC BY 4.0 (NASA IMPACT / University of Alabama in Huntsville).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample scenes, the example time series, validation and roles\n\n"
                "The default labelled dataset is 44 HLS scenes of the HLS Burn Scars dataset — 24 from the training split, 8 from the "
                "validation split and 12 from the test split that the upstream authors publish beside the burn-scar fine-tune, drawn with "
                "a fixed seed from the scenes whose mask is at least 60 % valid and 3 % burn scar. `fetch_corpus` downloads the dataset "
                "tarball from the Hub at its immutable revision, refuses it on any size or SHA-256 mismatch, streams through it once and "
                "copies out exactly the 88 pinned members — each refused on its own size or digest mismatch and written under its base "
                "name, never at a path taken from the archive — then reads the six-band float32 scenes (already surface reflectance in "
                "[0, 1]) and the masks, which keep −1 for no data. `dataset_manifest` validates every split, checks that no scene appears "
                "twice and records a digest. The four Mexico tiles that ship in the model snapshot (HLS S30 T13REM, 2018, four dates, "
                "448 × 560 pixels) become one four-date embedding record and four single-date ones.\n\n"
                "Look for: 24 / 8 / 12 scenes with burn fractions around 0.12..0.21, the patch-level label fractions the probe will "
                "see (16 × 16 patches; a patch is positive when at least half of its labelled pixels are), a written sample pair "
                "(`outputs/{stem}_sample_chip.tif` + `_sample_label.tif`, the BYOD shape), the validated example stacks, and three "
                "refusal probes — a five-band scene, a mask with an unknown class, a five-date stack — each rejected before the model "
                "runs. The tarball takes about a minute to fetch and a minute to stream."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/hls-burn-scars')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    grids = [patch_labels(r['label']) for r in splits[name]]\n"
                "    labelled = sum(int((g >= 0).sum()) for g in grids)\n"
                "    print({{name: {{'burn_fraction_pixels': part['class_pixel_fraction']['burn scar'], 'patches_labelled': labelled, 'patch_positive_fraction': round(sum(int((g == 1).sum()) for g in grids) / labelled, 4), 'tiles': part['regions']}}}})\n"
                "print({{'first_test_scene': validate_inputs(test_records[0])}})\n"
                "sample_pair = write_sample_pair(test_records[0], 'outputs/{stem}_sample_chip.tif', 'outputs/{stem}_sample_label.tif')\n"
                "print({{'sample_pair': sample_pair, 'pairs_csv': str(write_dataset_csv(test_records, 'outputs/{stem}_sample_pairs.csv'))}})\n\n"
                "example_tiles = sorted((WEIGHTS_DIR / 'examples').glob('*.tif'))\n"
                "series_record = {{'id': 'mexico-T13REM-2018-series', 'frames': [str(p) for p in example_tiles], 'dates': [p.name.split('.')[3][:7] for p in example_tiles]}}\n"
                "frame_records = [{{'id': 'mexico-T13REM-' + p.name.split('.')[3][:7], 'frames': str(p)}} for p in example_tiles]\n"
                "stack_report = validate_stacks([series_record, *frame_records])\n"
                "print({{'example_stacks': stack_report['n_records'], 'shapes': stack_report['shapes'], 'dates': series_record['dates']}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'five-band scene': [{{**test_records[0], 'image': test_records[0]['image'][:5]}}, *test_records[1:4]],\n"
                "    'unknown label class': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 1, 7, test_records[0]['label'])}}, *test_records[1:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})\n"
                "try:\n"
                "    validate_stacks([{{'id': 'five dates', 'frames': np.stack([test_records[0]['image']] * 5, axis=1)}}])\n"
                "    print({{'probe': 'five-date stack', 'verdict': 'accepted'}})\n"
                "except (TypeError, ValueError) as exc:\n"
                "    print({{'probe': 'five-date stack', 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. The frozen model: embeddings, masked reconstruction and the majority baseline\n\n"
                "`pipe.embed` standardises each stack with the band statistics of the pinned configuration (HLS digital-number scale), "
                "runs the encoder with every patch visible and returns, per record, the CLS embedding and the mean of the patch tokens of "
                "the final normalised layer (1024 floats each; the token grid is T × H/16 × W/16 and can be kept with `keep_tokens=True`). "
                "`pipe.reconstruct` runs the pretraining task — a seeded random mask over 75 % of the patches, the decoder predicting "
                "them — and reports the MSE over the masked patches in standardised units beside a **mean-fill baseline** that predicts "
                "every masked patch as the per-band, per-date mean of the visible pixels. `pipe.evaluate` on labelled scenes scores the "
                "attached probe at the patch level; before adaptation the probe is a zero head, so the frozen number **is** the majority "
                "baseline (every patch not burned): accuracy equal to the unburned fraction, burn-scar IoU 0.\n\n"
                "Look for: cosine similarities near 1 between the CLS embeddings of the four dates (the CLS token varies little across "
                "a site; the mean token is more discriminative, in the build record 0.98–0.99), a reconstruction MSE below the mean-fill "
                "baseline on the series and on every single date (0.077 vs 0.129 for the series in the build record), and a written "
                "RGB reconstruction of the first date. These are sample-sanity numbers on one site and 12 scenes, not the benchmark."
            ),
            "code": (
                "import time\n\n"
                "import tifffile\n\n"
                "t0 = time.perf_counter()\n"
                "embeddings = pipe.embed([series_record, *frame_records])\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'embedding': embeddings['embedding'], 'dim': embeddings['dim']}})\n"
                "for entry in embeddings['embeddings']:\n"
                "    print({{'record': entry['id'], 'frames': entry['frames'], 'token_grid': entry['token_grid'], 'cls_norm': round(float(np.linalg.norm(entry['cls'])), 2), 'mean_norm': round(float(np.linalg.norm(entry['mean'])), 2)}})\n"
                "def cosine(vectors):\n"
                "    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)\n"
                "    return np.round(unit @ unit.T, 3)\n"
                "frame_cls = np.stack([e['cls'] for e in embeddings['embeddings'][1:]])\n"
                "frame_mean = np.stack([e['mean'] for e in embeddings['embeddings'][1:]])\n"
                "print({{'cosine_cls_between_dates': cosine(frame_cls).tolist()}})\n"
                "print({{'cosine_mean_between_dates': cosine(frame_mean).tolist()}})\n"
                "with open('outputs/{stem}_embeddings.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'model': embeddings['model'], 'embedding': embeddings['embedding'], 'dim': embeddings['dim'], 'records': [{{'id': e['id'], 'frames': e['frames'], 'token_grid': e['token_grid'], 'cls': e['cls'].tolist(), 'mean': e['mean'].tolist()}} for e in embeddings['embeddings']]}}, f)\n\n"
                "reconstruction = pipe.reconstruct([series_record, *frame_records], keep_images=True, seed=0)\n"
                "for entry in reconstruction['results']:\n"
                "    print({{'record': entry['id'], 'masked_fraction': entry['masked_fraction'], 'masked_mse': entry['masked_mse'], 'baseline_mean_fill_mse': entry['baseline_mean_fill_mse']}})\n"
                "first = reconstruction['results'][0]\n"
                "rgb = np.clip(first['reconstruction'][[2, 1, 0], 0] / 0.3, 0, 1)  # red, green, blue of the first date, reflectance stretched to 0.3\n"
                "tifffile.imwrite('outputs/{stem}_reconstruction_rgb_date0.tif', (np.moveaxis(rgb, 0, -1) * 255).astype(np.uint8), photometric='rgb')\n"
                "tifffile.imwrite('outputs/{stem}_reconstruction_mask_date0.tif', first['mask'][0])\n"
                "print({{'metric': reconstruction['metric'], 'written': ['outputs/{stem}_reconstruction_rgb_date0.tif', 'outputs/{stem}_reconstruction_mask_date0.tif']}})\n\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_not_burned_test': {{k: frozen_test['baseline_not_burned'][k] for k in ('iou', 'accuracy', 'f1')}}}})\n"
                "print({{'frozen_test_zero_head': {{k: frozen_test['model'][k] for k in ('iou', 'mean_iou', 'accuracy', 'f1')}}}})\n"
                "print({{'frozen_validation_zero_head': {{k: frozen_val['model'][k] for k in ('iou', 'f1')}}}})"
            ),
        },
        {
            "md": (
                "## 6. Bounded linear probe on frozen patch tokens\n\n"
                "`pipe.adapt` runs the encoder once per training and validation scene with every patch visible, takes the patch tokens "
                "of encoder block 12 (zero-based `FEATURE_LAYER = 11`, chosen on the validation split during the build: the last "
                "block's tokens serve reconstruction, the middle blocks' tokens separate burned from unburned patches better), "
                "standardises them with the training-set mean and standard deviation, and trains a 1024 → 2 linear head (2,050 "
                "parameters) with cross-entropy and Adam at a fixed learning rate over seeded mini-batches of tokens. The encoder stores "
                "no gradient and is never changed. Epoch 0 records the zero head — the majority baseline; the epoch with the lowest "
                "validation loss is kept.\n\n"
                "Watch the validation loss: in the build record it fell from 0.693 to about 0.20 over 30 epochs while the validation "
                "burn-scar F1 rose to about 0.76; the whole adaptation takes under ten seconds on a T4 once the 32 scenes are embedded. "
                "`FEATURE_LAYER = 23` uses the final normalised layer (F1 about 0.77 on the test scenes in the build record, against "
                "0.90 for block 12)."
            ),
            "code": (
                "EPOCHS = 30  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-2  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4096  # @param {{type:\"integer\"}}\n"
                "FEATURE_LAYER = 11  # @param {{type:\"integer\"}}\n"
                "CLASS_BALANCE = False  # @param {{type:\"boolean\"}}\n\n"
                "def report(entry):\n"
                "    if entry['epoch'] % 5 and 'note' not in entry:\n"
                "        return\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_burn_iou'] = entry['val']['iou']['burn scar']\n"
                "        row['val_f1'] = entry['val']['f1']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, feature_layer=FEATURE_LAYER, class_balance=CLASS_BALANCE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'encoder_parameters_frozen': adapt_result['n_total'] - adapt_result['n_trainable'], 'features': adapt_result['features'], 'train_patches': adapt_result['n_train_patches'], 'positive_fraction_train': adapt_result['positive_fraction_train'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'loss': adapt_result['loss'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison\n\n"
                "The test scenes were never used for training or epoch selection (they come from the model repository's test split). "
                "The probe is scored exactly as the zero head was in Section 5, at the patch level, and the table puts the majority "
                "baseline (= the frozen zero head) and the probe side by side. The cell asserts what the procedure guarantees — the kept "
                "epoch's validation loss is no higher than the zero head's, and re-scoring the validation scenes reproduces the kept "
                "epoch's burn-scar IoU within 0.01 — and that the probe's test burn-scar IoU is above the baseline's 0: a linear probe "
                "that could not beat *every patch unburned* would be a finding. In the build record the test burn-scar IoU went from 0 "
                "to 0.815 (F1 0.898, accuracy 0.954 against a majority baseline of 0.787), a sample-sanity observation on 12 scenes with "
                "no dispersion estimate; the burn-scar row's full fine-tune reaches 0.925 at the pixel level on the same scenes, which is "
                "the number to read this one against."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{}}\n"
                "for key in ('mean_iou', 'accuracy', 'precision', 'recall', 'f1'):\n"
                "    comparison[key] = {{'baseline_not_burned': frozen_test['baseline_not_burned'][key], 'frozen_zero_head': frozen_test['model'][key], 'probe': adapted_test['model'][key]}}\n"
                "comparison['burn_iou'] = {{'baseline_not_burned': frozen_test['baseline_not_burned']['iou']['burn scar'], 'frozen_zero_head': frozen_test['model']['iou']['burn scar'], 'probe': adapted_test['model']['iou']['burn scar']}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "best = adapt_result['history'][adapt_result['best_epoch']]\n"
                "print({{'validation_burn_iou': {{'zero_head': frozen_val['model']['iou']['burn scar'], 'probe': adapted_val['model']['iou']['burn scar']}}, 'validation_loss': {{'zero_head': adapt_result['history'][0]['val_loss'], 'kept_epoch': best['val_loss']}}}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'embeddings': {{'records': [e['id'] for e in embeddings['embeddings']], 'cosine_mean_between_dates': cosine(frame_mean).tolist()}},\n"
                "    'reconstruction': [{{k: v for k, v in r.items() if k not in ('reconstruction', 'mask')}} for r in reconstruction['results']],\n"
                "    'frozen': {{'test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'test': adapted_test, 'validation': adapted_val}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['model']['iou'][CLASS_NAMES[1]] - best['val']['iou'][CLASS_NAMES[1]]) < 1e-2\n"
                "assert adapted_test['model']['iou'][CLASS_NAMES[1]] > frozen_test['baseline_not_burned']['iou'][CLASS_NAMES[1]]\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 8. Probe maps, artifact export and fresh reload\n\n"
                "The probe maps the first four test scenes at the patch level (one class per 16 × 16 pixels; the argmax over the two "
                "probe scores, written as 32 × 32 GeoTIFFs beside the burn fraction the map implies and the label's), a sanity check on "
                "what a patch-level map looks like next to the pixel labels.\n\n"
                "`pipe.save_artifact` writes the probe — its weight and bias and the feature mean and standard deviation, about 17 KB — "
                "as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, the "
                "digest of the converted base file, the adaptation scope and feature layer, the tensor names, the file size and SHA-256, "
                "the training configuration and the epoch history (OUT8). `PrithviFeaturePipeline.from_artifact` re-verifies the base "
                "file, checks the artifact manifest, scope, layer and digest **before** deserialising, rebuilds the model and attaches "
                "the probe — a fresh object from files, not the in-memory model (VER2). The cell asserts the same held-out burn-scar IoU "
                "within 0.001 and score maps within 0.001 (VER4)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "probe_maps = pipe.predict(test_records[:4])\n"
                "for record, pred in zip(test_records[:4], probe_maps['predictions']):\n"
                "    tifffile.imwrite(f'outputs/{stem}_probe_map_' + record['source_id'] + '.tif', pred['mask'])\n"
                "    grid = patch_labels(record['label'])\n"
                "    print({{'scene': record['source_id'], 'burn_fraction_patches_label': round(float((grid == 1).sum() / max((grid >= 0).sum(), 1)), 3), 'burn_fraction_patches_probe': pred['class_fraction']['burn scar'], 'resolution': probe_maps['resolution']}})\n"
                "with open('outputs/{stem}_predictions.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'model': probe_maps['model'], 'classes': probe_maps['classes'], 'resolution': probe_maps['resolution'], 'decision_rule': probe_maps['decision_rule'], 'predictions': [{{'id': p['id'], 'class_fraction': p['class_fraction']}} for p in probe_maps['predictions']]}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable': artifact_manifest['adapter']['trainable'], 'feature_layer': artifact_manifest['adapter']['feature_layer'], 'tensors': artifact_manifest['tensors'], 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PrithviFeaturePipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "before = pipe.predict(test_records[:2])['predictions']\n"
                "after = reloaded.predict(test_records[:2])['predictions']\n"
                "parity = {{'positive_iou_diff': round(abs(reloaded_test['model']['iou'][CLASS_NAMES[1]] - adapted_test['model']['iou'][CLASS_NAMES[1]]), 6), 'metrics_identical': reloaded_test['model'] == adapted_test['model'], 'max_abs_score_diff': max(float(np.abs(a['scores'] - b['scores']).max()) for a, b in zip(before, after))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch'], 'reloaded_feature_layer': reloaded.feature_layer}})\n"
                "assert parity['positive_iou_diff'] < 1e-3 and parity['max_abs_score_diff'] < 1e-3\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': [e for e in MANIFEST['files'] if e['path'] == SOURCE_CKPT_NAME],\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'data_tarball': {{'name': TAR_NAME, 'sha256': TAR_SHA256, 'pinned_members': 2 * len(SAMPLE_RECORDS)}},\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'lightning': lightning.__version__, 'tifffile': tifffile.__version__, 'terratorch': importlib.metadata.version('terratorch')}},\n"
                "    'data_source': data_source,\n"
                "    'reconstruction': evaluation_report['reconstruction'],\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 12 held-out scenes from the burn-scar model repository's test split, a 2,050-parameter linear probe on the frozen patch "
        "tokens of encoder block 12 of Prithvi-EO-2.0-300M reaches a patch-level burn-scar IoU near 0.8 against a majority baseline of "
        "0, and the pretraining task reconstructs 75 % masked patches of an HLS time series with less error than filling them with "
        "the visible mean. That is the claim: the embeddings carry the burned/unburned distinction linearly, the adaptation contract "
        "runs end to end on real labelled multispectral scenes drawn from a digest-verified tarball, the pickle is audited and "
        "converted rather than served, and the artifact that carries the change is 17 KB and reloads with the same outputs. It is not a "
        "claim about the model's GEO-bench skill, nor that a patch-level probe replaces the pixel-level fine-tune of the burn-scar row "
        "(0.925 IoU on the same scenes).\n\n"
        "The numbers are sample-sanity evidence: one seeded run, 12 test scenes from a handful of HLS tiles, no dispersion estimate, "
        "patch-pooled metrics that let large burns dominate, and labels derived from a burned-area product with their own uncertainty. "
        "The feature layer was chosen on the validation split of this dataset; another task may prefer another layer. Nothing here "
        "measures the model outside the contiguous United States, outside 2018–2021, on scenes larger than 1024 pixels, or with the "
        "temporal and location embeddings of the `TL` variants.\n\n"
        "Three things to carry to real data. **The six bands and their scaling are the contract:** blue, green, red, narrow NIR, "
        "SWIR 1, SWIR 2 in that order, surface reflectance in [0, 1] or × 10 000; a different band order or an uncorrected product is "
        "embedded without complaint and silently wrong. **Split by site or tile, not by scene:** neighbouring scenes of one fire are "
        "near-duplicates, and a random split makes memorisation look like skill. **Read the baseline first:** on a scene with 5 % "
        "burn the not-burned baseline is 95 % accurate; only the burn-scar IoU, precision and recall say whether the probe did "
        "anything.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, "
        "can acquire and digest-verify a pickled upstream checkpoint, audit and convert it into safetensors without executing "
        "anything outside the audited allow-list, build the model from the installed package, embed and reconstruct multi-date HLS "
        "stacks, fetch a digest-pinned tarball and extract exactly the pinned labelled scenes, execute a bounded linear probe, "
        "evaluate against a baseline on held-out scenes, and emit the shown machine-readable artifacts — without the repository "
        "being reachable. It does **not** establish benchmark superiority, production fitness, or Earth-observation skill beyond "
        "the checks shown.\n\n"
        "**Optional experiments (they do not affect the default path):** set `FEATURE_LAYER = 23` or `5` and compare; set "
        "`CLASS_BALANCE = True` and watch precision and recall trade places; raise `EPOCHS` and watch the validation loss; or bring "
        "your own labelled scenes through BYOD and read the baseline before the probe number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M (revision `{MODEL_REVISION}`)\n"
        "- HLS Burn Scars dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars (CC BY 4.0)\n"
        "- Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications. arXiv:2412.02732: https://arxiv.org/abs/2412.02732\n"
        "- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R. (2022). Masked autoencoders are scalable vision learners. CVPR: https://arxiv.org/abs/2111.06377\n"
        "- TerraTorch: https://github.com/IBM/terratorch\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
