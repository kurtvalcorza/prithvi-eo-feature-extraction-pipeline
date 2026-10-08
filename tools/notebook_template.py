"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

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
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime builds an isolated environment from the hash-locked pins (torch, torchvision, "
        "terratorch and its stack, tifffile, numpy, safetensors, huggingface-hub); nothing is installed into the notebook's own Python, so "
        "no restart is needed and Run all completes in one pass. It then stages and digest-verifies the pinned Prithvi-EO-2.0-300M snapshot "
        "(1.34 GB: the checkpoint and the four upstream example tiles) from the Hub, statically audits the pickle against an "
        "allow-list, converts it once into safetensors with a pinned digest, builds the masked autoencoder from the installed "
        "`terratorch` package and loads it strictly, embeds the four example tiles as one time series and as single frames and "
        "reconstructs them under a 75 % mask against a mean-fill baseline, fetches the digest-pinned HLS Burn Scars tarball (2.6 GB, no "
        "credential) and extracts exactly the 44 pinned scenes and masks, validates them and assigns the model repository's roles "
        "(24 training, 8 validation, 12 test), scores the majority baseline on the held-out scenes at the patch level, trains a bounded "
        "linear probe on the frozen patch tokens, scores the same scenes again, exports the embeddings and the probe maps, exports the "
        "probe as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.2 §5). On a T4 the model time is under a minute after the downloads; building the isolated environment and the tarball are "
        "the slowest steps."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled scenes — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, through "
        "the Colab upload dialog — as a zip (or folder) holding `pairs.csv` (columns `id`, `image`, `label`) beside six-band 512 × 512 GeoTIFF chips "
        "(blue, green, red, narrow NIR, SWIR 1, SWIR 2 — surface reflectance in [0, 1] or × 10 000) and single-band label rasters "
        "(0 = negative class, 1 = positive class, −1 = no data); at least **seven** labelled chips with some positive pixels (the seeded "
        "split keeps a quarter for test and a fifth for validation and needs four for training; Section 4 prints the minimum). An optional "
        "`group` column (a fire, tile or site id) keeps every chip of one group in one role, so near-duplicate scenes of one fire cannot sit on both "
        "sides of the split. Your chips are split by "
        "seed into training, validation and test sets and flow through the same contract — validation, majority baseline, probe "
        "training, held-out evaluation, probe maps, artifact export and reload parity. The expected schema, the ceilings and the privacy "
        "guidance are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see what a geospatial "
                "foundation model's embeddings carry: how a masked autoencoder pretrained on Harmonized Landsat Sentinel-2 (HLS) imagery "
                "reconstructs hidden patches, and whether a two-number-per-feature linear probe on its frozen patch tokens can find burn scars. "
                "No prior experience with Prithvi, TerraTorch or remote sensing models is assumed; the terms are explained where they first "
                "matter and again in the **Glossary** at the end. A GPU runtime (T4 or better) is expected.\n\n"
                "**Input → Model → Output.**\n\n"
                "| | Embedding and reconstruction | Linear probe |\n"
                "|---|---|---|\n"
                "| Input | a six-band HLS reflectance stack, 1–4 dates, sides multiples of 16 | 512 × 512 six-band chips with 0 / 1 / −1 masks (24 training, 8 validation, 12 test in the sample) |\n"
                "| Model | the frozen Prithvi-EO-2.0-300M encoder (and its decoder for reconstruction), loaded from audited, converted safetensors | the same frozen encoder; a 1024 → 2 head on the tokens of encoder block 12 is the only thing trained |\n"
                "| Output | CLS and mean-token embeddings (1024 floats), masked-patch MSE beside a mean-fill baseline | patch-level burn-scar maps, IoU / F1 beside the majority baseline, a 17 KB safetensors probe that reloads with parity |\n\n"
                "**How to use this notebook.** Choose a GPU runtime (**Runtime → Change runtime type → T4 GPU**), then **Runtime → Run all**. "
                "Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 "
                "are **infrastructure** — the isolated environment, the carried package and the audited model snapshot — and their cells are "
                "collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only "
                "values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to "
                "**Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded "
                "run (the Kaggle Tesla T4 run of 25 September 2026 recorded in `docs/release-verification.md`). **Troubleshooting**, a "
                "**Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n"
                "**Roadmap:** 1–3 infrastructure → 4 the labelled scenes, the example time series, validation and refusals *(evaluation "
                "practice)* → 5 the frozen model: embeddings, masked reconstruction and the majority baseline *(core concept)* → 6 a bounded "
                "linear probe on frozen tokens *(core concept)* → 7 the held-out paired comparison *(evaluation practice)* → 8 probe maps, "
                "export and fresh reload *(engineering)* → interpretation, troubleshooting, glossary and your conclusion."
            ),
        ],
    },
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
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12): the ViT-L encoder runs in float32 and needs about 2.7 GB of GPU memory for a 512 × 512 scene; on CPU one scene takes about two seconds to embed and the default path would take several minutes. About 6 GB of disk is needed for the checkpoint, its conversion and the tarball; building the isolated environment (terratorch pulls torchgeo, lightning and their dependencies) takes several minutes the first time and is reused on a re-run.",
        "- **Knowledge:** what a multispectral surface-reflectance scene is (bands, scaling, no-data), what a masked autoencoder reconstructs, what a token embedding and a linear probe are, and how IoU, precision and recall are read against a majority baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; `terratorch` is installed from PyPI at a pinned version.",
        "- **Data contract:** an embedding record is `{id, frames}` — a (6, T, H, W) reflectance stack with 1–4 dates and sides that are multiples of 16 in [64, 1024] (or 1–4 GeoTIFF paths in chronological order); a labelled record is `{id, image, label}` — a (6, 512, 512) chip (or a GeoTIFF; 13-band Sentinel-2 L1C files are reduced to the six HLS-equivalent bands) with values in [0, 1] or × 10 000, no-data 0 or −9999, and a (512, 512) mask with 0 / 1 / −1. Validation is structural: nothing checks that the bands are the right six in the right order, that the reflectance is corrected, or that the label belongs to the scene.",
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
                "see (16 × 16 patches; a patch is positive when at least half of its labelled pixels are), a BYOD template folder "
                "(`outputs/{stem}_byod_template/` — `pairs.csv` naming the one written chip and its label, which the cell reloads "
                "through the BYOD loader to prove the shape; a real BYOD set needs at least seven rows), the validated example stacks, and three "
                "refusal probes — a five-band scene, a mask with an unknown class, a five-date stack — each rejected before the model "
                "runs. The tarball takes about a minute to fetch and a minute to stream.\n\n"
                "The roles come from the upstream splits, which are by scene, not by tile: two of the 12 test scenes (HLS tiles T10TFQ "
                "and T10TGS) share a tile with training scenes, and the cell prints that overlap (`test_tiles_also_in_training`). "
                "Keep it in mind when reading the test number — a mild optimism, since a tile seen in training is not fully new ground. "
                "Every re-run of this cell first removes this notebook's earlier exports from `outputs/`, so the files there always "
                "describe the current run (sample or BYOD).\n\n"
                "*Evaluation practice.* **Predict before running:** the masks keep −1 for no data. Should a patch that is mostly no-data count "
                "as *not burned* when the probe is scored?"
            ),
            "code": (
                "import json\n"
                "import os\n"
                "import shutil\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "# A .zip or a folder already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# FX-M2: start every run (sample or BYOD) from an empty export set, so no file in outputs/ describes an earlier run.\n"
                "for stale in sorted(Path('outputs').glob('{stem}_*')):\n"
                "    shutil.rmtree(stale) if stale.is_dir() else stale.unlink()\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding pairs.csv and the GeoTIFF files.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding pairs.csv and the GeoTIFF files.')\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/hls-burn-scars')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "if USE_BYOD:\n"
                "    byod_records = train_records + val_records + test_records\n"
                "    byod_grouped = all(r.get('group') for r in byod_records)\n"
                "    print({{'byod_chips': len(byod_records), 'minimum_chips': byod_minimum_records(), 'grouped_split': byod_grouped, 'groups': sorted({{r['group'] for r in byod_records}}) if byod_grouped else 'no group column: chips of one fire or tile may land in different roles'}})\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    grids = [patch_labels(r['label']) for r in splits[name]]\n"
                "    labelled = sum(int((g >= 0).sum()) for g in grids)\n"
                "    print({{name: {{'burn_fraction_pixels': part['class_pixel_fraction']['burn scar'], 'patches_labelled': labelled, 'patch_positive_fraction': round(sum(int((g == 1).sum()) for g in grids) / labelled, 4), 'tiles': part['regions']}}}})\n"
                "train_tiles = {{r['region'] for r in train_records if r.get('region')}}\n"
                "print({{'test_tiles_also_in_training': sorted({{r['region'] for r in test_records if r.get('region')}} & train_tiles), 'validation_tiles_also_in_training': sorted({{r['region'] for r in val_records if r.get('region')}} & train_tiles)}})\n"
                "print({{'first_test_scene': validate_inputs(test_records[0])}})\n"
                "# FX-m3: the template folder holds pairs.csv naming exactly the files written beside it, so zipping it gives a loadable BYOD upload.\n"
                "byod_template = Path('outputs/{stem}_byod_template')\n"
                "sample_pair = write_sample_pair(test_records[0], byod_template / 'sample_chip.tif', byod_template / 'sample_label.tif')\n"
                "pairs_csv = write_dataset_csv(test_records[:1], byod_template / 'pairs.csv', files={{test_records[0]['id']: ('sample_chip.tif', 'sample_label.tif')}})\n"
                "print({{'byod_template': byod_template.as_posix(), 'files': sorted(p.name for p in byod_template.iterdir()), 'rows_reloaded_by_the_byod_loader': len(load_byod_dataset(byod_template))}})\n\n"
                "example_tiles = sorted((WEIGHTS_DIR / 'examples').glob('*.tif'))\n"
                "series_record = {{'id': 'mexico-T13REM-2018-series', 'frames': [str(p) for p in example_tiles], 'dates': [p.name.split('.')[3][:7] for p in example_tiles]}}\n"
                "frame_records = [{{'id': 'mexico-T13REM-' + p.name.split('.')[3][:7], 'frames': str(p)}} for p in example_tiles]\n"
                "stack_report = validate_stacks([series_record, *frame_records])\n"
                "print({{'example_stacks': stack_report['n_records'], 'shapes': stack_report['shapes'], 'dates': series_record['dates']}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "# FX-m1: pad each probe to the dataset minimum from the other roles, so a small BYOD test split still fails on the intended rule.\n"
                "probe_fill = (test_records[1:] + train_records + val_records)[:MIN_RECORDS - 1]\n"
                "probes = {{\n"
                "    'five-band scene': [{{**test_records[0], 'image': test_records[0]['image'][:5]}}, *probe_fill],\n"
                "    'unknown label class': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 1, 7, test_records[0]['label'])}}, *probe_fill],\n"
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
                '**What to notice:** the three split sizes, the burn fractions, `patch_positive_fraction` beside the pixel fraction, `disjoint`, and the three refusals.\n\n<details><summary>Check '
                'your reasoning</summary>No. A no-data pixel is neither burned nor unburned, so it is ignored: a patch is labelled only when enough of its pixels are '
                'labelled, and the rest are excluded from training and scoring (`patches_labelled` counts the ones kept). Counting no-data as *not burned* would '
                "inflate the majority baseline's accuracy for free. The refusals show the other half of the contract: a five-band scene, an unknown label class and a "
                'five-date stack are rejected before any model call, with the rule in the message. Validation is structural only — nothing here can tell that six bands '
                'are in the wrong order. The overlap line should list T10TFQ and T10TGS for the sample: the upstream split is by scene, so the test number carries a little '
                'tile-level optimism.</details>'
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
                "baseline (every patch not burned): accuracy equal to the unburned fraction, burn-scar IoU 0. A figure shows the first "
                "date three ways: what the encoder saw (75 % of the patches hidden), which patches were hidden, and the reconstruction.\n\n"
                "Look for: cosine similarities near 1 between the CLS embeddings of the four dates (the CLS token varies little across "
                "a site; the mean token is more discriminative, in the build record 0.98–0.99), a reconstruction MSE below the mean-fill "
                "baseline on the series and on every single date (0.077 vs 0.129 for the series in the build record), and a written "
                "RGB reconstruction of the first date. These are sample-sanity numbers on one site and 12 scenes, not the benchmark. "
                "The frozen evaluation always scores the zero head: if you re-run this cell after Section 6, the trained probe is set "
                "aside for these two calls and re-attached afterwards, so the numbers labelled *frozen* stay the majority baseline.\n\n"
                "**Predict before running:** will the decoder's reconstruction of the hidden 75 % beat filling every hidden patch with the "
                "visible mean, and by roughly how much?"
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
                "print({{'metric': reconstruction['metric'], 'written': ['outputs/{stem}_reconstruction_rgb_date0.tif', 'outputs/{stem}_reconstruction_mask_date0.tif']}})\n"
                "# FX-m4: show the first date as the encoder saw it (hidden patches grey), the hidden patches, and the reconstruction.\n"
                "import matplotlib.pyplot as plt\n\n"
                "hidden = first['mask'][0].astype(bool)\n"
                "visible_rgb = np.moveaxis(rgb, 0, -1).copy()\n"
                "visible_rgb[hidden] = 0.5\n"
                "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
                "for ax, image, title in zip(axes, (visible_rgb, hidden, np.moveaxis(rgb, 0, -1)), ('visible 25 % (hidden patches grey)', 'hidden patches (75 %)', 'reconstruction: visible + decoder fill')):\n"
                "    ax.imshow(image, cmap='gray' if image.ndim == 2 else None, interpolation='nearest')\n"
                "    ax.set_title(title, fontsize=9)\n"
                "    ax.axis('off')\n"
                "fig.suptitle(first['id'] + ', first date, RGB stretched to reflectance 0.3', fontsize=10)\n"
                "plt.show()\n\n"
                "t0 = time.perf_counter()\n"
                "# SWP-F: the frozen numbers are the zero head. On a re-run after Section 6 the trained probe is set aside for these\n"
                "# two calls and re-attached afterwards, so 'frozen' never scores the adapted probe.\n"
                "_attached_probe = (pipe.probe, pipe.feature_stats, pipe.feature_layer, pipe.adapter)\n"
                "pipe.probe, pipe.feature_stats, pipe.adapter = None, None, None\n"
                "try:\n"
                "    frozen_test = pipe.evaluate(test_records)\n"
                "    frozen_val = pipe.evaluate(val_records)\n"
                "finally:\n"
                "    pipe.probe, pipe.feature_stats, pipe.feature_layer, pipe.adapter = _attached_probe\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_not_burned_test': {{k: frozen_test['baseline_not_burned'][k] for k in ('iou', 'accuracy', 'f1')}}}})\n"
                "print({{'frozen_test_zero_head': {{k: frozen_test['model'][k] for k in ('iou', 'mean_iou', 'accuracy', 'f1')}}}})\n"
                "print({{'frozen_validation_zero_head': {{k: frozen_val['model'][k] for k in ('iou', 'f1')}}}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the cosine tables (CLS against mean token), `masked_mse` beside `baseline_mean_fill_mse` for the series and each date, and the '
                "zero head's test numbers, which equal the majority baseline's.\n\n<details><summary>Check your reasoning</summary>Yes, clearly. In the recorded run the "
                'masked MSE on the four-date series was 0.0773 against 0.1294 for mean fill — about 40 % less error, and lower on every single date too. The encoder '
                'has learned spatial and spectral structure that a per-band mean cannot express. The frozen evaluation shows burn-scar IoU 0 and accuracy 0.7865: with '
                'a zero head every patch is *not burned*, which is exactly the majority baseline — the number any probe has to beat.</details>'
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
                "`FEATURE_LAYER = 23` uses the final normalised layer; comparing it with block 12 is the optional activity at the end "
                "(change the field here, then re-run Sections 6–8).\n\n"
                "**Predict before running:** the probe has 2,050 parameters and sees each scene through a frozen encoder. Will the "
                "validation loss keep falling for all 30 epochs, or flatten early?"
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
                '**What to notice:** epoch 0 (the zero head, validation loss 0.693 = ln 2), the trend of `val_loss` and `val_f1`, `best_epoch`, and the trainable count '
                'against the frozen encoder count.\n\n<details><summary>Check your reasoning</summary>It kept falling. In the recorded run the validation loss went from '
                '0.6931 to 0.1996 and the kept epoch was the last one, 30, with validation F1 0.7577. Kept epoch = last epoch says the budget, not over-fitting, set '
                'the stopping point; more epochs might help a little. Only 2,050 numbers changed — the 300 M encoder parameters are untouched, which is why the '
                'artifact is 17 KB.</details>'
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison\n\n"
                "The test scenes were never used for training or epoch selection (they come from the model repository's test split). "
                "The probe is scored exactly as the zero head was in Section 5, at the patch level, and the table puts the majority "
                "baseline (= the frozen zero head) and the probe side by side. The cell stops only on what the procedure guarantees — the "
                "kept epoch's validation loss is no higher than the zero head's (epoch 0 is a candidate), and re-scoring the validation "
                "scenes reproduces the kept epoch's burn-scar IoU within 0.01. Whether the probe beats the baseline is **recorded as a "
                "verdict** (`improved`, `no improvement` or `worse`) in the report rather than asserted, so a run on your own scenes still "
                "exports, reloads and writes `result.json` when the probe does not help — a linear probe that could not beat *every patch "
                "unburned* would be a finding to read, not a crash. In the build record the test burn-scar IoU went from 0 "
                "to 0.815 (F1 0.898, accuracy 0.954 against a majority baseline of 0.787), a sample-sanity observation on 12 scenes with "
                "no dispersion estimate; the burn-scar row's full fine-tune reaches 0.925 at the pixel level on the same scenes, which is "
                "the number to read this one against.\n\n"
                "*Evaluation practice.* **Predict before running:** the majority baseline is already about 79 % accurate on these patches. "
                "Which number will tell you more about the probe — accuracy or burn-scar IoU?"
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
                "probe_iou = adapted_test['model']['iou'][CLASS_NAMES[1]]\n"
                "baseline_iou = frozen_test['baseline_not_burned']['iou'][CLASS_NAMES[1]]\n"
                "# SWP-A: the quality comparison is a recorded verdict, not an assert, so a BYOD run still exports, reloads and writes result.json.\n"
                "probe_verdict = 'improved' if probe_iou > baseline_iou else ('no improvement' if probe_iou == baseline_iou else 'worse')\n"
                "comparison['verdict'] = {{'probe_vs_majority_baseline_burn_iou': probe_verdict}}\n"
                "print({{'verdict': comparison['verdict']}})\n"
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
                "# Contract integrity (not model quality): epoch 0 is a selection candidate, and re-scoring reproduces the kept epoch.\n"
                "if best['val_loss'] > adapt_result['history'][0]['val_loss']:\n"
                "    raise RuntimeError('contract: the kept epoch has a higher validation loss than the zero head, which epoch selection cannot produce')\n"
                "if abs(adapted_val['model']['iou'][CLASS_NAMES[1]] - best['val']['iou'][CLASS_NAMES[1]]) >= 1e-2:\n"
                "    raise RuntimeError('contract: re-scoring the validation scenes does not reproduce the kept epoch')\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the `burn_iou` row, `accuracy` beside the baseline's, precision against recall, and the `verdict` line.\n\n<details><summary>Check "
                'your reasoning</summary>Burn-scar IoU. In the recorded run accuracy moved from 0.7865 to 0.9539 — impressive-looking, but the baseline got 0.79 by '
                'predicting nothing. Burn-scar IoU went from 0 to 0.8149 (F1 0.8980, precision 0.8504, recall 0.9512): the probe finds almost every burned patch and '
                'over-calls some neighbours. The verdict was *improved*. Twelve scenes, one seed and no dispersion estimate make this a sample-sanity result; the '
                "burn-scar row's pixel-level fine-tune (IoU 0.925) is the stronger reference.</details>"
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
                "within 0.001 and score maps within 0.001 (VER4).\n\n"
                "**Decision rule and who owns the threshold.** The two numbers the probe gives each patch are the softmax of a linear "
                "head's scores. They are **not calibrated** probabilities, and the map is simply their argmax, with no threshold. "
                "Where to put a threshold, and how to trade missed burns against false alarms, is a deployment decision. Whoever "
                "deploys the probe owns it and must calibrate it on their own labelled data. Nothing in this notebook sets or "
                "validates a threshold.\n\n"
                "A figure shows each mapped scene's RGB, its pixel label (grey = no data) and the probe's 32 × 32 patch map side by side. "
                "The maps use the scene's `source_id` for the sample and the `id` column for BYOD chips."
            ),
            "code": (
                "import importlib.metadata\n"
                "import platform\n"
                "import shutil\n\n"
                "import re\n\n"
                "import matplotlib.pyplot as plt\n\n"
                "# FX-M2: BYOD records carry id / image / label (and an optional group); source_id exists for the sample alone.\n"
                "scene_name = lambda record: re.sub(r'[^A-Za-z0-9._-]+', '_', str(record.get('source_id', record['id'])))\n"
                "probe_maps = pipe.predict(test_records[:4])\n"
                "fig, axes = plt.subplots(len(probe_maps['predictions']), 3, figsize=(10, 3.4 * len(probe_maps['predictions'])), squeeze=False)\n"
                "for row, (record, pred) in enumerate(zip(test_records[:4], probe_maps['predictions'])):\n"
                "    tifffile.imwrite(f'outputs/{stem}_probe_map_' + scene_name(record) + '.tif', pred['mask'])\n"
                "    grid = patch_labels(record['label'])\n"
                "    scene_rgb = np.clip(np.moveaxis(np.asarray(record['image'])[[2, 1, 0]], 0, -1) / 0.3, 0, 1)\n"
                "    label_view = np.where(record['label'] < 0, 0.5, record['label']).astype(np.float32)\n"
                "    for ax, image, title in zip(axes[row], (scene_rgb, label_view, pred['mask']), (scene_name(record) + ' RGB', 'label (white = burn scar, grey = no data)', 'probe patch map (white = burn scar)')):\n"
                "        ax.imshow(image, cmap=None if image.ndim == 3 else 'gray', vmin=0, vmax=1, interpolation='nearest')\n"
                "        ax.set_title(title, fontsize=8)\n"
                "        ax.axis('off')\n"
                "    print({{'scene': scene_name(record), 'burn_fraction_patches_label': round(float((grid == 1).sum() / max((grid >= 0).sum(), 1)), 3), 'burn_fraction_patches_probe': pred['class_fraction']['burn scar'], 'resolution': probe_maps['resolution']}})\n"
                "plt.show()\n"
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
                "    'verdict': comparison['verdict'],\n"
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
        {
            "md": (
                "**What to notice:** the label against probe burn fractions for the four scenes, the artifact's size and `feature_layer`, and `reload_parity`.\n\n<details><summary>Check "
                "your reasoning</summary>Patch maps are coarse: one class per 16 × 16 pixels, so fire edges come out as staircases and the probe's burn fraction "
                "differs from the label's by a few points per scene. In the recorded run the reloaded probe reproduced the held-out metrics exactly "
                '(`positive_iou_diff` 0.0, `metrics_identical` True, `max_abs_score_diff` 0.0): the 16.7 KB safetensors file plus the pinned, re-verified base is the '
                'whole model.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**On the default sample** — 12 held-out scenes from the burn-scar model repository's test split — a 2,050-parameter linear probe on the frozen patch "
        "tokens of encoder block 12 of Prithvi-EO-2.0-300M reached a patch-level burn-scar IoU of 0.8149 in the recorded run, against a majority baseline of "
        "0, and the pretraining task reconstructs 75 % masked patches of an HLS time series with less error than filling them with "
        "the visible mean. That is the claim: the embeddings carry the burned/unburned distinction linearly, the adaptation contract "
        "runs end to end on real labelled multispectral scenes drawn from a digest-verified tarball, the pickle is audited and "
        "converted rather than served, and the artifact that carries the change is 17 KB and reloads with the same outputs. It is not a "
        "claim about the model's GEO-bench skill, nor that a patch-level probe replaces the pixel-level fine-tune of the burn-scar row "
        "(0.925 IoU on the same scenes). **If you ran your own scenes (BYOD),** these sample numbers do not describe them: read your "
        "Section 7 `verdict`, your burn-scar IoU against your own majority baseline, and your split (grouped or not) instead; "
        "the three rules below apply unchanged.\n\n"
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
        "## Optional activity: Predict → Change → Run → Observe → Explain\n\n"
        "None of this is part of the default **Run all**. Items 1–3 each change one form field in Section 6. **Scope of a re-run:** change the "
        "field, then run Sections 6, 7 and 8 in that order. `pipe.adapt` starts every run from a fresh zero head, so each run is a new "
        "experiment, not continued training. Section 5 need not be re-run; if you do re-run it, it still scores the zero head. Write "
        "your prediction down before you run.\n\n"
        "1. **Which layer?** *Predict:* will the tokens of the final layer (`FEATURE_LAYER = 23`) separate burned from unburned "
        "patches better or worse than block 12's? *Change* the field, *run* 6–8, *observe* the validation F1 per epoch and the "
        "Section 7 `burn_iou` row, and *explain* the difference in terms of what the last layer was trained to do (reconstruct pixels).\n"
        "2. **Class balance.** *Predict:* with `CLASS_BALANCE = True` (burned patches weighted up), which moves more: precision or "
        "recall, and in which direction? *Observe* the Section 7 `precision` and `recall` rows and *explain* the trade.\n"
        "3. **Epochs.** *Predict:* with `EPOCHS = 60`, will the kept epoch again be the last one? *Observe* `best_epoch` and the "
        "validation loss.\n"
        "4. **Your own scenes.** Set `USE_BYOD = True` with `BYOD_PATH` (add a `group` column so one fire stays in one role) and re-run "
        "from Section 4. Read the majority baseline before the probe number.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 3 is slow or stops during the pickle audit or '
        'conversion** — the audit refuses any global outside the allow-list and names it; that means the downloaded checkpoint is not the pinned one. Delete '
        "the snapshot folder and run Section 3 again.\n- **Section 4 stops on the tarball's size or SHA-256** — the 2.6 GB download was cut short or altered. "
        'Run Section 4 again: a truncated download is fetched again, and scenes already extracted and verified are reused. If the digest still fails, delete `weights/hls-burn-scars/` and run it once more.\n- **CUDA out of memory in Section 5 or 6** — another notebook is holding the GPU, or the '
        'runtime is smaller than a T4. Restart the session and choose **Run all**; on CPU the default path still completes, only slower.\n- **BYOD: "BYOD '
        'datasets must be a directory or a .zip holding pairs.csv" or a band/shape refusal** — the message names the rule; chips must be six-band 512 × 512 '
        'GeoTIFFs in the documented band order with masks of 0 / 1 / −1.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working '
        'directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip (or folder) in the '
        'runtime and set `BYOD_PATH` to its path.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the '
        "cell again.\n\n## Glossary\n\n- **HLS (Harmonized Landsat Sentinel-2):** NASA's surface-reflectance product that puts Landsat 8/9 and Sentinel-2 on one 30 "
        'm grid; the six bands here are blue, green, red, narrow NIR, SWIR 1 and SWIR 2.\n- **Surface reflectance:** the fraction of sunlight a surface reflects '
        'in a band, after atmospheric correction; stored in [0, 1] or × 10 000.\n- **Masked autoencoder (MAE):** a model pretrained by hiding most input patches '
        'and reconstructing them; its encoder becomes a general feature extractor.\n- **Patch / token:** the encoder cuts each date into 16 × 16 pixel patches; '
        'each becomes one 1024-number token.\n- **CLS embedding / mean token:** two scene summaries — a learned summary token, and the average of the patch tokens.\n- '
        '**Mean-fill baseline:** predicting every hidden patch as the per-band, per-date mean of the visible pixels; the reconstruction has to beat it.\n- '
        '**Linear probe:** a single linear layer trained on frozen features; if it works, the information was already in the features.\n- **Feature layer:** the '
        'encoder block whose tokens feed the probe (block 12 of 24 here, zero-based 11).\n- **Majority baseline:** predicting the most common class everywhere '
        '(*not burned*); the zero head predicts exactly this.\n- **IoU, precision, recall, F1:** overlap of predicted and true burned patches; the share of '
        'predicted burns that are real; the share of real burns found; their harmonic mean.\n- **Pickle audit / safetensors:** the upstream checkpoint is a '
        'pickle that could run code when loaded; it is statically checked against an allow-list and converted once to safetensors, a format that stores only tensors.\n- '
        '**Isolated environment:** the separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\nComplete '
        "these in your own words; the recorded run's values are in the **Check your reasoning** answers above.\n\n- On held-out scenes, the probe's burn-scar IoU "
        "was ___ against the majority baseline's ___ (verdict: ___).\n- Masked reconstruction beat mean fill by ___ on the four-date series, which tells me the "
        'frozen encoder ___.\n- The number I would not trust on its own is ___, because ___.\n- Before using this probe on my own scenes I would check the band '
        'order and scaling, split by ___, and read ___ first.\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M (revision `{MODEL_REVISION}`)\n"
        "- HLS Burn Scars dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars (CC BY 4.0)\n"
        "- Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications. arXiv:2412.02732: https://arxiv.org/abs/2412.02732\n"
        "- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R. (2022). Masked autoencoders are scalable vision learners. CVPR: https://arxiv.org/abs/2111.06377\n"
        "- TerraTorch: https://github.com/IBM/terratorch\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
