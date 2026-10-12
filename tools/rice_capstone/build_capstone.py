# ruff: noqa: E501  (learner-facing markdown is kept as one line per paragraph)
"""Generate tutorials/DIMER_Philippines_Rice_Mapping_Capstone.ipynb deterministically (E2E, WORKSHOP, standalone).

    python tools/rice_capstone/build_capstone.py          # write the notebook
    python tools/rice_capstone/build_capstone.py --check  # exit 1 if the committed notebook differs

The notebook carries, byte-identical and hash-verified: this directory's `capstone.py` (the stage runner), the
repository's `pipeline.py` verbatim as `prithvi_pipeline.py` (audit, conversion, strict load), the hash lock
`tutorials/requirements-rice-capstone.lock.txt`, the model snapshot manifest and a dataset pin manifest. It is written
output-free. Do not edit the notebook by hand; edit this file or the carried sources and regenerate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Philippines_Rice_Mapping_Capstone.ipynb"
LOCK = REPO / "tutorials" / "requirements-rice-capstone.lock.txt"
PIPELINE = REPO / "src" / "prithvi_eo_feature_extraction_pipeline" / "pipeline.py"
WEIGHT_MANIFEST = REPO / "weights" / "prithvi-eo-2.0-300m" / "dimer-base-manifest.json"
GENERATOR = "tools/rice_capstone/build_capstone.py"
UV_WHEEL = {  # identical to the EO workshop's pinned uv (tools/eo_workshop/build_workshop.py)
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/"
    "uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}
MANAGED_PYTHON = "3.12.12"
DATA_MANIFEST = {
    "slug": "kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching",
    "version": 1,  # versions 2 and 3 have identical data but no Kaggle download archive
    "url": "https://www.kaggle.com/datasets/kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching",
    "license": "CC BY 4.0 (component terms in RIGHTS.md continue to apply)",
    "access": "public; anonymous download, no credential",
    "index_sha256": "e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d",
    "package_sha256": {
        "data-train.zip": "74458ba9f3322705b3c0aed62e5d4a63590804233882a0e7ba5719cfc9596a52",
        "data-validation.zip": "4af5c646c9888bce39b2c8a2a2a0e709d8dc01b1609c67ad0c212553038beaa7",
        "development-example.zip": "3d4e8de06ba0d8fd7450d2a9bcc6da07dbd23c5a8ae289104a6fc8584155a823",
        "INDEX.jsonl": "e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d",
        "RIGHTS.md": "8ed21cab21cc5b5771545f9be0fb32e15e487e661a90a57ab68ee94f0fb94b56",
    },
    "chips": {"train": 32, "validation": 8},
    "keys_used": ["S2", "S2_valid", "label"],
    "keys_never_used": ["label_raw", "AlphaEarth", "AE_valid", "granule_count"],
    "label_lineage": {  # issue #22 M2; facts from the dataset build recipe (see the spec, section 3)
        "target": "annual paddy-rice occurrence according to the source map (not field-verified rice, rice area or number of seasons)",
        "source": "Chen et al., 20 m resolution maps of paddy rice cropping intensity and planting dates in Monsoon Asia from 2018 to 2021, doi:10.5281/zenodo.15124936, CC BY 4.0",
        "layer": "annual PADDYINTENSITY<year> of the same year as the chip-year; four-year composite and planting dates not used",
        "code_map": {"1": 1, "2": 1, "3": 1, "0": 0, "other": "build fails"},
        "non_rice_includes": "every source 0 in the chip, including water and sea (no per-pixel land mask)",
        "resampling": "nearest neighbour from the 1/5400 degree EPSG:4326 source onto the 20 m EPSG:32651 chip grid (corners on multiples of 2560 m), with a one-pixel halo",
        "ignore_255": "every pixel whose 3 x 3 neighbourhood holds both rice and non-rice (one pixel on each side of each class edge, diagonals included), plus pixels without source coverage",
        "scored_pixels": "label != 255 only",
        "recipe_sha256": "7adf68fe9d23622a1cf008f71cacd7c5006ed1377e34a5fc8c17c046953dc63e",
        "rebuild_kit": "recipe-kit.zip sha256 976530ffa324f0d701051d757214a5dca977621d5a3e752b4f11bac0dbf7f032 in Kaggle kurtvalcorza/rice-philippines-s2-alphaearth-v1, version 2",
    },
}
PIECE = 1000  # at most this many characters per carried string piece, so no cell line is unreadably long


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def carried_files() -> dict[str, str]:
    files = {
        "capstone.py": _read(HERE / "capstone.py"),
        "prithvi_pipeline.py": _read(PIPELINE),
        "requirements.txt": _read(LOCK),
        "weights/prithvi-eo-2.0-300m/dimer-base-manifest.json": _read(WEIGHT_MANIFEST),
        "data_manifest.json": json.dumps(DATA_MANIFEST, indent=2, sort_keys=True) + "\n",
    }
    hashes = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in files.items()}
    files["source.json"] = json.dumps({
        "repository": "kurtvalcorza/prithvi-eo-feature-extraction-pipeline",
        "notebook": "tutorials/" + NOTEBOOK.name, "generator": GENERATOR, "carried_sha256": hashes,
        "pipeline_source": "src/prithvi_eo_feature_extraction_pipeline/pipeline.py (verbatim)",
    }, indent=2, sort_keys=True) + "\n"
    return files


def _literal(text: str) -> str:
    pieces: list[str] = []
    for line in text.splitlines(keepends=True):
        while len(line) > PIECE:
            pieces.append(line[:PIECE])
            line = line[PIECE:]
        pieces.append(line)
    return "(\n" + "".join(f"        {piece!r}\n" for piece in pieces) + "    )"


def carrier_cell(files: dict[str, str]) -> str:
    hashes = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in files.items()}
    body = ["# @title Infrastructure: write and verify the embedded source, lock and manifests", "CARRIED_FILES = {"]
    body += [f"    {name!r}: {_literal(text)}," for name, text in files.items()]
    body += ["}", "CARRIED_HASHES = {"]
    body += [f"    {name!r}: {digest!r}," for name, digest in hashes.items()]
    body += ["}", "for name, text in CARRIED_FILES.items():",
             "    data = text.encode('utf-8')",
             "    if hashlib.sha256(data).hexdigest() != CARRIED_HASHES[name]:",
             "        raise RuntimeError(f'{name}: embedded text does not match its pinned SHA-256; regenerate the notebook')",
             "    target = ROOT / name",
             "    target.parent.mkdir(parents=True, exist_ok=True)",
             "    target.write_bytes(data)",
             "print(f'{len(CARRIED_FILES)} carried files written and verified under {ROOT}')"]
    return "\n".join(body)


# --------------------------------------------------------------------------------------------------
# learner-facing and infrastructure cells
# --------------------------------------------------------------------------------------------------

TITLE = r"""# Rice From Space: What Does a Satellite Foundation Model Add?

**DIMER Philippines Rice Mapping Capstone** · **Profile:** `E2E` · **Mode:** `WORKSHOP` · **Kind:** `CAPSTONE` · **Notebook standard:** DIMER Notebook Specification `2.4` · **Release status:** Candidate · **Model:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` @ `9eb1b1102806` · **Data:** Kaggle `kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching`, version 1

> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,
> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party
> model weights and datasets under their own licenses (see the model card) and uses compute
> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are
> responsible for having the right to use it. Results here are tutorial evidence, not
> benchmarks. Everything is provided "as is", without warranty of any kind. To the maximum
> extent permitted by law, the maintainers and the DIMER platform accept no liability for
> its use. Its rice maps are not official crop statistics, area estimates or agricultural advice.

**Driving question:** Given 32 labelled Sentinel-2 chips, does a head trained on frozen Prithvi-EO-2.0 features map Philippine rice better than a hand-built spectral-temporal baseline, and what can 8 validation chips actually establish? B and C are whole pipelines that differ in more than the pretrained features, so the comparison asks which pipeline agrees better with a published rice map, not what pretraining alone contributes.

Rice paddies change through the year: fields are flooded, the crop greens up, and then it is harvested. Optical satellites see that cycle as a pattern over time, not in a single image [1]. In this notebook you will compare three ways of turning a year of Sentinel-2 observations [2] into a rice map, and practise stating only what a small, clustered evaluation can support. It is an optional capstone after the *Rice from space* course lessons [L3 (Sentinel-2 LightGBM)](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3-lightgbm-baseline) [11] and [L3B (AlphaEarth LightGBM)](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3b-alphaearth-baseline) [12] on Kaggle; it links to them rather than re-teaching feature engineering.

**Who this notebook is for.** Learners who can open a hosted notebook, run cells in order and read short Python. No prior remote-sensing or deep-learning experience is assumed: each geospatial or evaluation term is explained at first use, and the glossary below collects them.

**Prerequisites and runtime.** A fresh Google Colab session with a **T4 GPU** (Runtime → Change runtime type → T4 GPU) and at least 12 GiB free disk. No account, token, upload, DIMER service or Git clone is needed. The notebook downloads the public teaching dataset (about 350 MB), the 1.33 GB model checkpoint and a hash-locked Python 3.12.12 environment (Linux x86_64 only). **Expected runtime: about 10 minutes** for Run all on the default path, measured as 589 s on a fresh Colab Tesla T4 for the previous blob `33037307` (commit `d82963c`, Colab CLI sequential execution, 2026-10-11, including the 75 s environment setup); about 11 minutes (633 s) in the same environment with the fine-tune and bring-your-own-data sections switched on. Each run measures and records its own runtime and GPU memory.

**Input → System → Output.** A year of six-band Sentinel-2 surface reflectance for a 2.56 km chip (24 calendar bins, 20 m pixels) → one of three systems: **A** call everything non-rice, **B** a LightGBM model on hand-built spectral-temporal features, or **C** a small head on frozen Prithvi-EO-2.0 features [3] → a 0/1 rice map scored against a published rice map. The trained head is exported, reloaded in a fresh process and can score your own chips.

### Learning objectives

By the end of this notebook you will be able to:

1. **Explain** Sentinel-2 bands, 20 m pixels, cloud gaps and temporal compositing, and why rice is a temporal signal.
2. **Trace** how the input a pretrained model expects is built — four quarterly frames, standardised, gaps filled — without resizing, and **identify** what was filled.
3. **Run** the training of a small head on frozen features and **compare** it with two baselines under a frozen protocol.
4. **Interpret** pooled precision, recall, F1 and IoU for a minority class, and **explain** why accuracy hides failure.
5. **Run** one controlled experiment (one frame versus four) and **explain** the result.
6. **Write** a bounded conclusion that states its evidence, its uncertainty and what it cannot claim.

### How to use this notebook

- **Run all** from a fresh T4 runtime (Runtime → Run all). Every cell runs in order without editing; nothing asks for input, and no manual restart is needed. The optional fine-tuning and bring-your-own-data sections are switched off by default.
- **Infrastructure** cells are collapsed and titled *Infrastructure*. They make the run reproducible. You may run them without studying their implementation; they are not learning objectives.
- Sections are tagged **[Concept]**, **[Evaluation practice]** or **[Engineering]**.
- Before each result you will be asked to **predict**. Write your prediction down first. After each result, **What to notice** describes normal output, and a collapsed **Check your reasoning** box holds a sample answer: open it only after answering.
- Completion notes are optional personal learning aids, **not required submissions**.

### Roadmap

| Section | What you do | Kind |
|---|---|---|
| 1 | Prepare a reproducible, isolated runtime | Engineering |
| 2 | Inspect the chips: bands, gaps, the rice season, the labels | Concept |
| 3 | Predict, declare the protocol, and run baselines A and B | Evaluation practice |
| 4 | Trace how Prithvi's input is built and extract frozen features | Concept |
| 5 | Train head C on the fit chips; choose epoch and threshold on development | Concept |
| 6 | Change one thing: one frame versus four | Evaluation practice |
| 7 | Optional extension: fine-tune the encoder (off by default) | Evaluation practice |
| 8 | Score A, B and C once on the validation chips | Evaluation practice |
| 9 | Interpret: pooled and per-chip results, error maps, the 320 m patch effect | Evaluation practice |
| 10 | Export, reload in a fresh process and verify | Engineering |
| 11 | Read the run's answer and conclude with evidence | Evaluation practice |
| 12 | Optional: bring your own chips | Engineering |

**Evidence boundary.** 40 chip-years: 32 for training, 8 for validation, all validation chips from two regions (Central and Eastern Visayas). This is a teaching comparison, not a national accuracy estimate.

**Release status and evidence.** The status, **Candidate**, is a maintainer decision and is separate from how much execution evidence exists. The previous blob `1ed294f0` was run by the maintainer in a browser on 2026-10-12: Run all on the defaults, Run all with the fine-tune checkbox ticked, bring-your-own-data through the upload dialog, and a malformed upload that was refused with one clean message; earlier blobs also passed Colab CLI runs on a T4. This revision (issue #22: claims, label lineage, BYOD verification, spatial audit, observation support, GeoTIFF placement checks) changes the notebook, so those runs are not evidence for it until it is run again; `docs/release-verification.md` in the host repository records each run and what remains open."""

PROBLEM = r"""### Problem statement

**Who has the problem.** Agencies, researchers and analysts who monitor rice in the Philippines and want a yearly map of where paddy rice is grown, made from satellite imagery rather than from field surveys alone.

**The task.** Turn a year of Sentinel-2 observations of a 2.56 km chip into a 20 m map that marks each pixel as paddy rice or not, where the reference is a published annual paddy-rice map (Section 2 gives the exact label definition). Scores cover the retained labelled pixels (`label ≠ 255`), not every exported pixel; class-edge pixels are excluded, which removes part of the small-field and edge-delineation problem from the score.

**What a useful outcome looks like.** Keep three questions apart. (1) **Signal:** on the 8 validation chips, which are scored once, a pooled rice F1 clearly above the all-non-rice floor (A, F1 0) shows that a system finds some reference-rice signal. (2) **Comparator:** the frozen-Prithvi pipeline (C) is ahead only if it reaches a higher pooled rice F1 than the hand-built spectral-temporal pipeline (B) on the same pixels; because the two pipelines also differ in time steps, gap handling, pixel sampling and output architecture, that does not by itself isolate what pretraining contributes. (3) **Practical usefulness:** meeting operational requirements such as agreed accuracy targets, area estimates and field validation. This capstone addresses the first two questions, not the third. Section 10 states this run's answer from its own validation table.

**Scope boundary.** This notebook does not estimate rice area, cropping intensity or planting dates, does not forecast early in the season, and makes no national or operational accuracy claim.

### Data context

The sample is the public Kaggle dataset *Philippines Rice S2 AlphaEarth v1 Teaching*, version 1 [9] (CC BY 4.0), compiled by Kurt Valcorza. It is a 40-chip teaching subset of a larger benchmark of Philippine chips from 2019–2021: 32 training chips and 8 validation chips, each 128 × 128 pixels at 20 m (about 350 MB in total). Each chip holds 24 calendar-bin median composites of six Sentinel-2 surface-reflectance bands (modified Copernicus Sentinel data [2]), a validity mask, AlphaEarth annual embeddings [8] (not used here), and a label derived from a published annual paddy-rice cropping-intensity map [4] (same year as the chip): 0 non-rice, 1 rice, 255 ignore. Section 2 gives the exact derivation.

**How representative it is.** The chips were chosen by a seeded geographic sampling rule that favours rice-containing chips, not drawn as a probability sample, and all 8 validation chips come from two neighbouring regions (PH-07 Central Visayas, PH-08 Eastern Visayas). Rice is a minority of the labelled pixels. Labels are map products, not field observations. Whether Prithvi-EO-2.0's pretraining imagery covered these places is unknown. These gaps limit the results to tutorial evidence about these chips."""

GLOSSARY = r"""<details>
<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>

| Term | Meaning in this notebook |
|---|---|
| **Pixel** | One grid cell; here 20 m × 20 m on the ground. A chip is 128 × 128 pixels = 2.56 km square. |
| **Band** | One measured wavelength range. We use six Sentinel-2 bands: blue B2, green B3, red B4, narrow near-infrared B8A, shortwave-infrared B11 and B12 [2]. |
| **Surface reflectance** | The fraction of sunlight a surface reflects after atmospheric correction. The chips store it directly (digital number × 0.0001). |
| **Calendar bin / compositing** | The year is cut into 24 equal bins; each bin holds the median of the cloud-free observations that fell in it. Empty bins are missing, not zero. |
| **Valid mask** | `S2_valid` is 1 where a bin has a cloud-free observation. Missing values are NaN and are never read as reflectance. |
| **NDVI** | (NIR − red) / (NIR + red): high for green vegetation, low for water and bare soil. |
| **Label** | 0 non-rice, 1 paddy rice, 255 ignore (one pixel on each side of every rice/non-rice edge, about 40 m across). Ignored pixels are never scored. |
| **Foundation model** | A large model pretrained on broad data and then adapted; Prithvi-EO-2.0 is one for Earth observation [3]. |
| **Frozen features / head** | The pretrained encoder's internal outputs, used without changing its weights; the *head* is the small trainable layer on top. |
| **Patch / token** | Prithvi cuts each frame into 16 × 16 pixel patches (320 m here) and describes each with a 1024-number *token*. |
| **Pixel spacing / resolved detail** | Pixel spacing is the size of a map's grid cells (20 m for every output here). Resolved detail is the smallest pattern the map can actually separate; C predicts from 320 m tokens, so its detail is coarser than its 20 m spacing. |
| **Fit / development / validation** | Fit chips train models; development chips make every choice; validation chips are scored once, at the end. |
| **Threshold** | The cut-off that turns a rice score in [0, 1] into a rice/non-rice decision; choosing it is model selection. |
| **TP / FP / FN / TN** | Rice found / non-rice called rice / rice missed / non-rice correctly left. |
| **Precision / recall / F1 / IoU** | TP/(TP+FP) / TP/(TP+FN) / 2TP/(2TP+FP+FN) / TP/(TP+FP+FN). |
| **Pooled** | Counts summed over all scored pixels of all chips before computing a ratio. |
| **Epoch** | One pass of training over all fit chips; more epochs fit the fit chips more closely, which can stop helping on new chips. |
| **Loss (cross-entropy)** | The number training lowers: small when the head gives high rice scores to rice pixels and low ones to non-rice. |
| **Early stopping** | Keeping the epoch (or boosting round) where the development score was best instead of the last one. |
| **LightGBM / gradient boosting** | A fast method that adds many small decision trees, each correcting the previous ones; baseline B uses it on per-pixel features. |
| **Transformer encoder** | Prithvi's network: layers (blocks) that let every patch token exchange information with every other before the next block. |

</details>"""

SEC1 = r"""## 1. Prepare a reproducible environment · [Engineering]

> **Infrastructure.** The next three code cells are collapsed. You may run them without studying their implementation; the learning activities start in Section 2.

**Input:** a fresh supported GPU runtime. **System:** hash-verified embedded files and a hash-locked dependency set installed into an isolated environment. **Output:** an isolated process for each stage and a record of the exact source, data and model identities.

Supported runtime: Google Colab (Linux x86_64) with a T4 GPU; the lock is built for manylinux x86_64 and the first cell stops on any other platform. Colab's own Python (3.13) and packages are left untouched: the model stack runs in a separate CPython 3.12.12 environment, so nothing is installed into the notebook kernel and no restart is needed. The cells download public data anonymously and stop on any failed integrity check; do not remove pins to get past a failure. Each run writes to a new directory."""

CELL_RUNTIME = r"""# @title Infrastructure: check the runtime, GPU and disk; create a fresh run directory
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

SESSION_START = time.perf_counter()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Supported runtime: Google Colab (Linux x86_64) with a T4 GPU. The hash lock is built for '
                       'manylinux x86_64; other platforms are not supported.')
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
except FileNotFoundError as exc:
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU, then Runtime > Run all.') from exc
if gpu.returncode:
    raise RuntimeError('Select a fresh T4 GPU runtime; CUDA was not detected.')
print('GPU:', gpu.stdout.strip())
ROOT = Path.cwd() / 'outputs' / 'philippines_rice' / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True)
if shutil.disk_usage(ROOT).free < 12 * 1024**3:
    raise RuntimeError('At least 12 GiB free disk is required for the environment, weights, chips and exports.')
print('Run directory:', ROOT)"""

INSTALL_MD = r"""The next cell installs the isolated runtime and defines small display helpers. Each stage then runs as a separate process of that environment, so GPU memory is released between stages. A failed stage stops the notebook; it is never recorded as a skipped success.

**Expected result:** the Python and PyTorch versions of the isolated environment and `True` for CUDA. Installation messages are normal."""

CELL_INSTALL = r"""# @title Infrastructure: install the locked runtime and define the stage runner
import io

UV_URL = '@@UV_URL@@'
UV_BYTES = @@UV_BYTES@@
UV_SHA256 = '@@UV_SHA256@@'
for attempt in range(3):
    try:
        with urllib.request.urlopen(UV_URL, timeout=90) as response:
            wheel = response.read(UV_BYTES + 1)
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)
if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
    raise RuntimeError('uv wheel size/hash mismatch: the pinned download changed; stop and report it')
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
           PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', KAGGLEHUB_CACHE=str(ROOT / 'cache' / 'kagglehub'))
for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN',
             'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
    ENV.pop(name, None)  # anonymous downloads only; the kernel's paths must not leak into the environment
# Colab exports MPLBACKEND=module://matplotlib_inline...; the isolated environment has no matplotlib_inline, so every
# stage writes figures to files with the Agg backend and this kernel displays them.
ENV['MPLBACKEND'] = 'Agg'
subprocess.run([str(UV), 'venv', '--quiet', '--managed-python', '--python', '@@PYTHON@@', str(ROOT / 'env')], env=ENV, check=True)
PYTHON = ROOT / 'env/bin/python'
subprocess.run([str(UV), 'pip', 'install', '--quiet', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:',
                '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / 'requirements.txt')], env=ENV, check=True)
subprocess.run([str(PYTHON), '-c', 'import sys, torch, numpy; print("Python", sys.version.split()[0], "| torch", torch.__version__,'
                ' "| numpy", numpy.__version__, "| CUDA", torch.cuda.is_available()); assert torch.cuda.is_available()'],
               env=ENV, check=True)
BOOTSTRAP_SECONDS = time.perf_counter() - SESSION_START
print(f'Bootstrap took {BOOTSTRAP_SECONDS:.0f} s')


REFUSED_EXIT = 3  # capstone.py's exit code for an input it refused with a message (BYOD); not a crash


def run_stage(stage, *options, refusal_ok=False):
    # Runs one stage in the isolated environment and streams its output once. Returns True on success; with
    # refusal_ok, returns False when the stage refused its input (the reason is already printed above).
    log = ROOT / (stage + '.log')
    print('Running', stage, '- one process per stage; log:', log, flush=True)
    omitted = 0
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'capstone.py'), '--root', str(ROOT), '--stage', stage, *options],
                                   env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 2000:
                print(line.rstrip(), flush=True)
            elif line.strip():
                omitted += 1  # an over-long line would freeze the output area; it stays in the log
        process.wait()
    if process.returncode == REFUSED_EXIT and refusal_ok:
        return False
    if process.returncode:
        if omitted:
            print(f'({omitted} over-long line(s) not shown; the full output is in {log})')
        raise RuntimeError(f'{stage} failed with exit {process.returncode}. Read the diagnostics above.')
    return True


import csv
from IPython.display import FileLink, Image, Markdown, display


def _cell(value):
    if value is None or value == '':
        return 'undefined'
    if isinstance(value, bool) or value in ('True', 'False'):
        return str(value).lower()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).replace('|', '/').replace('\n', ' ')
    return f'{number:.4f}' if ('.' in str(value) or 'e' in str(value)) else str(value)


def display_rows(rows, columns):
    lines = ['| ' + ' | '.join(title for _key, title in columns) + ' |', '|' + '---|' * len(columns)]
    lines += ['| ' + ' | '.join(_cell(row.get(key)) for key, _title in columns) + ' |' for row in rows]
    display(Markdown('\n'.join(lines)))


def show_table(name, columns=None, limit=40):
    with (ROOT / 'outputs' / name).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError('Expected a non-empty evidence table: ' + name)
    keys = columns or [k for k in rows[0] if k not in ('undefined',)]
    display_rows(rows[:limit], [(k, k) for k in keys])
    undefined = [f"{r.get('system') or r.get('variant') or r.get('chip')}: {r['undefined']}" for r in rows if r.get('undefined')]
    for note in undefined[:limit]:
        print('undefined ->', note)
    if len(rows) > limit:
        print(f'Showing {limit} of {len(rows)} rows; the complete CSV is in results.zip.')


def show_figures(pattern):
    paths = sorted((ROOT / 'outputs' / 'figures').glob(pattern))
    if not paths:
        raise RuntimeError('Expected figures are missing: ' + pattern)
    for path in paths:
        display(Markdown(f'`{path.name}`'))
        display(Image(filename=str(path)))


def show_source(*names):
    # Prints top-level functions of the carried capstone.py, so the core steps can be read in place.
    import ast
    text = (ROOT / 'capstone.py').read_text(encoding='utf-8')
    found = {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
             if isinstance(node, ast.FunctionDef)}
    for name in names:
        print(found[name], end='\n\n')


def show_record(name):
    value = json.loads((ROOT / 'outputs' / name).read_text(encoding='utf-8'))
    if name == 'prepare.json':
        print('Layout:', value['layout'], '| chips:', value['chips'], '| roles:', value['roles'])
        print('Development regions:', value['development_regions'], '| fit regions:', value['fit_regions'])
        display_rows(value['table'], [('chip', 'Chip'), ('role', 'Role'), ('region', 'Region'), ('rice_pixels', 'Rice px'),
                                      ('nonrice_pixels', 'Non-rice px'), ('ignored_pixels', 'Ignored px'),
                                      ('selected_bins', 'Frames (bins)'), ('min_selected_valid_fraction', 'Min valid'),
                                      ('filled_fraction', 'Filled')])
        display_rows(value['refusal_probes'], [('probe', 'Refusal probe'), ('refused', 'Refused'), ('message', 'Message')])
    elif name == 'spatial_audit.json':
        print(f"Chips: {value['chips']} | unique sites: {value['unique_sites']} | sites seen in more than one year: "
              f"{value['sites_with_repeated_years']} | sites in more than one role: {value['sites_in_more_than_one_role'] or 'none'}")
        print('Footprints shared across roles:', value['cross_role_overlapping_footprints'] or 'none',
              '| touching across roles:', value['cross_role_touching_footprints'] or 'none')
        display_rows([{'pair': k, 'metres': v, 'chips': value['nearest_cross_role_chips'][k]}
                      for k, v in value['min_cross_role_distance_m'].items()],
                     [('pair', 'Roles'), ('metres', 'Minimum distance (m)'), ('chips', 'Closest chips')])
    elif name == 'selection.json':
        for key in ('B', 'C', 'C-FT'):
            if key in value:
                item = value[key]
                print(f"{key}: {item['system']} | threshold {item['threshold']} | development F1 {item['dev_f1']:.4f}"
                      + (f" | epoch {item['selected_epoch']}" if 'selected_epoch' in item else ''))
        if 'activity' in value:
            print('Activity:', value['activity'])
        last = value.get('C', value.get('B'))
        display_rows(last['sweep'], [('threshold', 'Threshold'), ('f1', 'Dev F1'), ('iou', 'Dev IoU'),
                                     ('precision', 'Dev precision'), ('recall', 'Dev recall')])
    else:
        rows = []
        for key, item in value.items():
            if isinstance(item, dict):
                rows += [{'check': f'{key}: {k}', 'result': v} for k, v in item.items() if not isinstance(v, (dict, list))]
            elif not isinstance(item, list):
                rows.append({'check': key, 'result': item})
        display_rows(rows, [('check', 'Record'), ('result', 'Value')])"""

SEC2 = r"""## 2. Inspect the chips · [Concept]

The teaching dataset holds 40 chip-years from 2019–2021: 32 in training regions and 8 in validation regions (PH-07 Central Visayas and PH-08 Eastern Visayas). Each chip is 128 × 128 pixels at 20 m in UTM zone 51N (EPSG:32651).

- **Bands.** Six Sentinel-2 bands in the order the model expects: blue B2, green B3, red B4, narrow near-infrared B8A, and shortwave-infrared B11 and B12 [2]. Visible light shows colour; near-infrared is bright over healthy leaves; shortwave infrared drops over water and wet soil.
- **Compositing and gaps.** The year is split into 24 equal calendar bins. Each bin is the median of that bin's cloud-free observations; where clouds hid every observation, the bin is missing (NaN, `S2_valid` = 0). The Philippines is cloudy, so gaps are normal.
- **Why rice is temporal.** A paddy is flooded before transplanting (dark in shortwave infrared, low NDVI), then greens up quickly (NDVI rises), then is harvested (NDVI falls). One image can confuse young rice with water or grass; the sequence is distinctive [1].
- **Labels.** The target is **annual paddy-rice occurrence according to a published map**: labels are map products, not field observations. How each label was made:
  - *Source:* Chen et al.'s *20 m resolution maps of paddy rice cropping intensity* [4], using for each chip-year the **annual layer of the same year** (`PADDYINTENSITY2019`, `2020` or `2021`); the four-year composite and the planting-date layers are not used.
  - *Classes:* source codes 1, 2 and 3 (single, double and triple paddy rice) become **1 rice**, so the number of seasons is discarded; code 0 (no paddy rice) becomes **0 non-rice**, which includes water and sea because no land mask is applied inside a chip. Any other code would have stopped the dataset build.
  - *Grid:* the source grid (1/5400°, about 20 m, geographic coordinates) is resampled by **nearest neighbour** onto each chip's 20 m UTM grid.
  - *Ignored edges:* **255** marks every pixel whose 3 × 3 neighbourhood holds both rice and non-rice — one 20 m pixel on **each** side of every class edge, diagonals included, so about 40 m across an edge (computed with a one-pixel halo beyond the chip). Scores use only pixels with `label ≠ 255`, so excluding these edge pixels removes part of the small-field and field-edge problem from the score.
  - *Pinned recipe:* every chip's index entry records label-recipe SHA-256 `7adf68fe…`, which the prepare stage checks; the public rebuild kit is cited under *Data, model, attribution and licences*.

**Input:** the public dataset, verified against the pinned `INDEX.jsonl` digest and every chip's SHA-256. **System:** schema checks, frame selection and three deliberate refusal probes. **Output:** a per-chip table, composites, an NDVI series and valid-observation counts by bin. The development example shipped with the dataset is a training sample from the parent study, so it is never read.

**Predict before running:** in which months do you expect the most cloud gaps? Will a rice pixel's NDVI curve have one peak, two, or none?"""

CELL_PREPARE = r"""run_stage('prepare')
show_record('prepare.json')
show_figures('overview_*')"""

AFTER_PREPARE = r"""**What to notice:** the table lists every chip with its role, the four bins chosen as Prithvi's frames, the lowest valid fraction among them, and the share of model input that had to be filled. The three refusal probes show that a malformed chip is stopped with a message naming the broken rule, before any model runs. The NDVI figure plots one rice pixel and one non-rice pixel from one chip: compare their curves, but treat them as single examples, not a representative phenology analysis — either may depart from the textbook pattern. In the valid-fraction bar chart, find the bins where clouds removed most observations.

**Checkpoint:** name one month range with frequent gaps, one feature of the rice NDVI curve, and one reason the label might disagree with what you see in the composite.

<details>
<summary>Check your reasoning (open after answering)</summary>

Gaps cluster in the wet season, when clouds are persistent; that is exactly when some rice is growing, so missing data and the signal you need overlap. Many Philippine rice areas are cropped twice a year, so a rice curve can show two green-up peaks separated by low values at harvest and flooding — but the pixel you see may show one peak, none, or a jagged line. Plausible reasons: missing bins break the series in the cloudy season; the field may be cropped once that year; the pixel may mix a field with a levee or road; or the map label may be wrong for that pixel. The non-rice pixel need not be flat either: other crops, grass and wetlands also change through the year. Describe what your plot shows and give a reason before generalising from one pixel. The label can disagree because it comes from a published map with its own errors, because a 20 m pixel can mix a field edge with a road or levee (hence the ignored edge band), and because the composite you see is one bin, not the whole year.

</details>"""

SEC3 = r"""## 3. Predict, declare the protocol and run the baselines · [Evaluation practice]

**Predict first:** where do you expect a pretrained model and a hand-built baseline to disagree — field edges, cloudy chips, small fields, wet but non-rice land? Write one guess.

**Roles.** The 32 training chips are split by **whole region** into **fit** (trains models) and **development** (makes every choice) with a declared, seeded rule: regions are ordered by a SHA-256 hash and added to development until it holds about 25 % of chips with at least two rice chips. Grouping by whole region reduces the risk that neighbouring chips land in different roles, because nearby pixels and chips are not independent [5]; it does not by itself guarantee distance, since chips on either side of a regional boundary can still be close. So the prepare stage measures it: the first table below (`spatial_audit.json`) lists the unique sites, any site seen in several years or in more than one role, any footprint shared or touching across roles, and the minimum ground distance between chips of different roles. The 8 **validation** chips are scored once, after every choice is frozen. The development example is never used.

**Frozen rules (written to `selection.json`).**
- A threshold is chosen on development from 0.10 to 0.90 in steps of 0.05 by the highest pooled development rice F1; ties go to the threshold closest to 0.5, then the lower one. A pixel is rice when its score is **≥** the threshold.
- Head C's training epoch is chosen on development; the activity in Section 6 runs on development only.
- Validation is scored once for A, B and C, and the result is kept even if a baseline wins.

**Systems.**
- **A — all non-rice:** predicts non-rice everywhere. Rice is a minority class, so A can score high accuracy while finding nothing.
- **B — spectral-temporal LightGBM:** the course study's M1 recipe — 70 features per pixel (per-band mean, spread, minimum, maximum and 10th/50th/90th percentiles over the observed year; quarterly band means; quarterly valid fractions) — on a class-stratified sample of at most 65,536 fit pixels, one fixed configuration (31 leaves), early stopping on development. Lesson [L3](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3-lightgbm-baseline) [11] teaches this recipe.

**Metrics.** Counts are pooled over every pixel with `label ≠ 255`. Rice F1 is primary; IoU, precision and recall are reported beside it, and accuracy only to show why it misleads. For an imbalanced class, precision-based measures stay informative where accuracy does not [6]. A ratio with a zero denominator is reported as **undefined** with its reason, never as 0 or 1.

**Expected result:** the spatial audit, then B's threshold sweep on development and a two-row development table."""

CELL_BASELINE = r"""show_record('spatial_audit.json')  # measured by the prepare stage: how far apart the roles really are
run_stage('baseline')
show_record('selection.json')
show_table('development_metrics.csv', ['system', 'threshold', 'f1', 'iou', 'precision', 'recall', 'accuracy', 'scored_pixels'])"""

AFTER_BASELINE = r"""**What to notice:** in the spatial audit of the pinned dataset, every chip is a different site seen in one year, no footprint is shared or touches across roles, and the closest fit and development chips are about 29 km apart (every validation chip is at least about 190 km from every training chip; distances computed from the pinned index). That is measured separation for these chips, not a guarantee of independence: pixels within a chip, and chips within one role, remain clustered. A's accuracy equals the share of non-rice pixels, while its recall and F1 are 0 and its precision is undefined (it never predicts rice). Read B's sweep around the selected threshold: a broad plateau means the choice is not fragile.

<details>
<summary>Check your reasoning (open after answering)</summary>

If about nine in ten development pixels are non-rice, A scores about 0.9 accuracy without finding a single paddy — which is why the comparison is read in F1 and IoU. B's threshold is a real choice made on development only; its development F1 is therefore optimistic for B and is not a held-out score. The threshold is now frozen.

</details>"""

SEC4 = r"""## 4. Trace how Prithvi's input is built and extract frozen features · [Concept]

**Input:** each chip's 24 bins. **Model:** Prithvi-EO-2.0-300M, a vision transformer pretrained by masked reconstruction of multispectral, multitemporal imagery [3], loaded from an immutable revision of its model card [10]. **Output:** for every chip, an 8 × 8 grid of 1024-number features from encoder block 12.

How the input is built, and what each step costs:

1. **Four frames.** The model was pretrained on four time steps. We take the clearest bin of each calendar quarter — highest valid fraction, earliest bin on ties — and record the chosen bins and their valid fractions (Section 2 table).
2. **Normalisation.** Reflectance × 10,000, then subtract and divide by the model's pinned band means and standard deviations. This is the repository's audited preprocessing, not a new recipe.
3. **Gaps.** After standardisation every missing value is set to 0, which is the pretraining mean. The filled share is recorded per chip and per frame. Pixels with no observation in any chosen frame stay in the scoring, as in the course study. Note the asymmetry with B: B's features include each quarter's observed fraction, so B knows where data were missing, while C receives mean-filled values and **no validity channel**, so its head cannot tell an observed average pixel from a filled one. Section 9 reports a secondary diagnostic by the number of observed frames.
4. **No resizing.** Prithvi was pretrained on 30 m Harmonized Landsat–Sentinel (HLS) data; these chips are 20 m Sentinel-2 surface reflectance. We keep the native 20 m grid and do not resample or harmonise to HLS, so the model sees finer pixels than it was trained on, and the processing level is close to, but not the same as, HLS. Both are stated domain shifts. Resampling or harmonising would be a different design choice with its own trade-offs; it is not tested here.
5. **Patches.** Each 16 × 16 pixel patch — **320 m** at 20 m — becomes one token, so the features are spaced 320 m apart even though the input pixels are 20 m. A 128-pixel chip gives 8 × 8 patches per frame. Block 12 of 24 is fixed in advance (the repository's validated `FEATURE_LAYER`); there is no layer sweep. Tokens are averaged over the four frames.

Steps 1–3 are the functions `select_frames` and `prithvi_input` in `capstone.py`, which the second Infrastructure cell (*write and verify the embedded source*) writes to the run directory printed in Section 1. The next cell prints those two functions. Reading them is optional; the steps above describe what they do.

The upstream checkpoint is a pickle. Its imported globals are audited statically against a pinned digest, it is unpickled once with torch's `weights_only` loader and converted to safetensors, and the model is loaded strictly from the verified file. No downloaded model code is executed. Features are computed once in float32 with deterministic GPU settings and cached, so later stages train on the cache.

**Predict before running:** how many patch tokens does the encoder return for one chip with four frames? (Hint: one extra summary token is added.)"""

CELL_SOURCE_INPUT = r"""# Optional reading: steps 1-3 above (frame choice, normalisation and gap filling), from the carried capstone.py
show_source('select_frames', 'prithvi_input')"""

CELL_FEATURES = r"""run_stage('features')
show_record('model.json')
show_record('features.json')"""

AFTER_FEATURES = r"""**What to notice:** the model record shows the pinned revision, the audit and conversion (or that a verified file was reused), the parameter count and a strict load. The feature record shows grid shapes of 1024 × 8 × 8, the time taken and the peak GPU memory measured in this run.

<details>
<summary>Check your reasoning (open after answering)</summary>

Four frames × 8 × 8 patches = 256 patch tokens, plus one class token = 257; the stage refuses to continue if the count differs. Each token summarises a 320 m square, so the features cannot resolve a 40 m field boundary by themselves: whatever head we train will start from a coarse grid.

</details>"""

SEC5 = r"""## 5. Train head C on the fit chips · [Concept]

**System C:** the frozen block-12 features → a 1 × 1 convolution from 1024 to 256 channels → GELU → a 1 × 1 convolution to 2 classes → bilinear upsampling from 8 × 8 to 128 × 128 pixels. Loss: cross-entropy with label 255 ignored, no class weights. Fixed in advance: AdamW, learning rate 0.001, weight decay 0.01, batch 8, at most 30 epochs, seed 0, no augmentation. The epoch with the highest development rice F1 at 0.5 is kept (earliest on ties), then C's threshold is chosen on development with the same rule as B. Training runs on the cached features, so it takes seconds. The head is `_head_module` and the training loop `train_head` in the same `capstone.py`; the next cell prints the head.

**What this head can resolve.** It predicts one pair of logits per 320 m patch — an 8 × 8 grid — and interpolates them bilinearly to 128 × 128; it has no high-resolution skip connection and no learned decoder below the patch size. So C's map has 20 m **pixel spacing** but its **resolved detail** is set by the 320 m token grid: a 20 m output grid does not establish 20 m field delineation. This is a property of this simple head, not of every Prithvi-based segmentation design; decoders with learned upsampling or skip connections are a different design not tested here.

**Predict before running:** will the development loss keep falling for 30 epochs? Will C's development F1 beat B's?"""

CELL_SOURCE_HEAD = r"""# Optional reading: head C (two 1 x 1 convolutions on the 8 x 8 token grid, then bilinear upsampling)
show_source('_head_module')"""

CELL_TRAIN = r"""run_stage('train-head')
show_figures('head_training_curves*')
show_record('selection.json')
show_table('development_metrics.csv', ['system', 'threshold', 'f1', 'iou', 'precision', 'recall', 'accuracy'])"""

AFTER_TRAIN = r"""**What to notice:** fit loss usually keeps falling while development loss flattens or rises; the dashed line marks the kept epoch. The development table now has three rows. All three were selected on these development chips, so this table is for choosing, not for claiming.

<details>
<summary>Check your reasoning (open after answering)</summary>

With 24 fit chips the head can memorise the fit set; the development curve is what tells you when that starts. If C's development F1 is below B's, that is a legitimate result: B sees all 24 bins at every pixel, while C sees four frames through 320 m patches. If C wins on development, the margin may still shrink on validation, because development informed C's epoch and threshold.

</details>"""

SEC6 = r"""## 6. Change one thing: one frame versus four · [Evaluation practice]

**Predict → Change one thing → Run → Observe → Explain.** This is a **fixed-recipe, development-only comparison**. Keep the frozen Prithvi weights and the head's recipe (architecture, optimiser, seed), reuse the epoch count and threshold that were selected for the four-frame head C, and train one head per input with that recipe. Change only the frames the encoder sees: **T = 1** — only one quarter's clearest frame, tried for each of the four quarters — versus the canonical **T = 4**. The T = 1 heads are not tuned separately, so this is not an equally tuned comparison of one-frame and four-frame systems. Everything is scored on development chips only; validation is untouched.

**Predict before running:** which single quarter will score highest? Will it fall below T = 4, match it or exceed it — and what could explain each outcome?

This activity does **not** change the canonical system: validation uses T = 4, as declared before the activity."""

CELL_ACTIVITY = r"""run_stage('activity')
show_table('activity.csv', ['variant', 'epochs', 'threshold', 'f1', 'iou', 'precision', 'recall'])
show_figures('activity_*')"""

AFTER_ACTIVITY = r"""**Observe and explain:** compare each single quarter with T = 4. The T = 4 row reproduces C's development result exactly (same recipe, seed and epochs), which shows the procedure is unchanged; each T = 1 row is a head trained with the same recipe on one-frame features, not C itself coping with fewer frames. Read the table within its limits: the epoch count and threshold were selected for T = 4 on these same development chips and the T = 1 heads borrow them, so the comparison is tilted towards T = 4, and these development chips alone establish neither the size nor the direction of a general difference.

<details>
<summary>Check your reasoning (open after answering)</summary>

If T = 4 scored higher, the supported sentence is: under a fixed training recipe and the epoch count and threshold selected for the four-frame system, the four-frame input achieved higher development F1 than the single-quarter alternatives. This does not isolate whether the advantage comes from seasonal progression (flooding, green-up, harvest), from improved observation coverage (four frames fill each other's cloud gaps), or from settings better suited to the four-frame system. If a single quarter matched or exceeded T = 4, that is a legitimate result too: one well-timed image may carry most of the signal on these development chips, or the borrowed settings may happen to suit it. Separating these explanations would need each variant tuned under the same search budget, or coverage-matched and temporal-order controls; this notebook does not run them.

</details>"""

SEC7 = r"""## 7. Optional extension: fine-tune the encoder (C-FT) · [Evaluation practice]

Switched off by default. When `RUN_FINETUNE` is ticked, a separate system **C-FT** unfreezes the patch embedding, the CLS token and encoder blocks 1–12 (the blocks that feed block 12; blocks 13–24 and the decoder stay frozen) and trains them with a fresh head under the same fit/development protocol: AdamW (encoder 0.00001, head 0.001), batch 4, 10 epochs fixed in advance, fp16 mixed precision, epoch and threshold chosen on development. Its validation predictions are computed without labels at the selected epoch and stay sealed; C-FT is scored in Section 8 only because it was selected **before** validation scoring. If you tick it after Section 8 has run, the stage refuses: a model chosen after seeing validation results would be exploratory. The run records its peak GPU memory with the full AdamW state. For scale (not re-run here): the recorded Colab CLI runs with the fine-tune switched on, of the earlier blobs `c1c966bb` (commit `d6956ab`) and `33037307` (commit `d82963c`) on a Tesla T4 on 2026-10-11, each measured a 4.404 GiB peak for this stage (see `docs/release-verification.md` in the host repository). The canonical comparison and the exported artifact remain the frozen-encoder system C.

To use it: tick the box, then Runtime → Run all on a fresh runtime.

**Expected result / What to notice:** a fine-tune curve with the fit loss and the development rice F1 at threshold 0.5 per epoch (development loss is not computed for C-FT) and a record naming the selected epoch, threshold and peak GPU memory; in Section 8 a fourth validation row, C-FT. Fine-tuning far more parameters on 24 fit chips can overfit, so C-FT may score below C; that is a result, not an error. Some PyTorch operations it uses are non-deterministic, so its numbers can vary slightly between runs."""

CELL_FINETUNE = r"""RUN_FINETUNE = False  # @param {type:"boolean"}
if RUN_FINETUNE:
    run_stage('finetune')
    show_table('finetune_curve.csv')
    show_record('finetune.json')
else:
    print('C-FT not run (RUN_FINETUNE = False); the canonical comparison is A, B and C.')"""

SEC8 = r"""## 8. Score A, B and C once on the validation chips · [Evaluation practice]

**Question tested:** with every choice frozen on development, how do the all-non-rice rule, B at its development threshold and C at its development threshold compare on the 8 validation chips from PH-07 and PH-08?

The stage first records the digest of `selection.json`, then scores. It refuses a second scoring in the same run directory, and the export stage refuses if the selection changed afterwards.

**Predict before running:** will the ranking on validation match development? Which validation chip will be hardest?"""

CELL_EVALUATE = r"""run_stage('evaluate')
show_table('metrics.csv', ['system', 'threshold', 'f1', 'iou', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'fn', 'tn'])
show_table('per_chip_metrics.csv', ['chip', 'region', 'system', 'f1', 'iou', 'precision', 'recall', 'reference_rice_pixels'])"""

SEC9 = r"""## 9. Interpret the result · [Evaluation practice]

**Pooled first.** The first table is the comparison: one pooled F1, IoU, precision and recall per system over the same labelled pixels. **Per chip second**, for diagnosis only. Two validation chips contain no reference rice: their recall is undefined, and their F1 is undefined if a system also predicts no rice there (or 0 if it predicts any). Undefined values are printed with their reason, never averaged in as numbers.

**What the evidence can bear.** Eight chips from two neighbouring regions, many of them close together. Pixels and chips are spatially dependent, so treating them as independent samples would overstate confidence [5], and spatially naive validation can make mapping models look far better than they are [7]. We therefore compute no significance tests and no confidence intervals, and make no national or operational claim. Labels are published map products, so disagreement is not proof of model error. Whether Prithvi's pretraining imagery included these places is unknown.

**For reference only (not re-run here):** on the same 8 validation chips, the course lessons report teaching-scale F1 of **0.760** for Sentinel-2 LightGBM ([Lesson 3](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3-lightgbm-baseline) [11], Kaggle version 4, notebook code SHA-256 `c30514a4…`) and **0.928** for AlphaEarth LightGBM ([Lesson 3B](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3b-alphaearth-baseline) [12], Kaggle version 3, notebook code SHA-256 `e98f6a57…`). Those are lesson results with their own caveats — in particular, they chose their threshold on these same validation chips, so they are development scores, not held-out ones. AlphaEarth annual embeddings [8] combine several sensors and had crop-related supervision; they are not used by A, B or C.

**Observation support (secondary diagnostic).** The next cell first splits B's and C's pooled metrics by how many of C's four selected frames observed each pixel (0 to 4). These groups partition the same scored pixels as the table above; the primary comparison is unchanged and no difficult pixel is removed from it. Read it as a diagnostic: if C does much worse where frames were filled, part of the B–C gap may come from gap handling rather than the features themselves; small groups give unstable numbers. Each validation chip's `<chip>_observed_frames.tif` export holds this count.

The next cell then shows the error maps for every validation chip (TP, FP, FN, TN; ignored pixels grey), where B and C disagree, and the 320 m patch effect on the chip with the most rice."""

CELL_FIGURES = r"""show_table('observation_support.csv', ['system', 'observed_frames', 'scored_pixels', 'reference_rice_pixels', 'f1', 'precision', 'recall'])
show_figures('validation_*')"""

AFTER_FIGURES = r"""**Interpret:** pick one chip where B and C disagree. Is the disagreement along field edges, in whole fields, or near cloud-filled areas? In the patch figure, can you see C's 16-pixel blocks? Weigh model error, label error and the 320 m grid as competing explanations.

<details>
<summary>Check your reasoning (open after answering)</summary>

Read the maps rather than assuming a pattern. Two things to look for: C decides from 320 m patches that are then smoothed, so check whether its misses are whole fields (low recall) or edges; B decides per 20 m pixel without spatial context, so check whether its errors are speckled. Compare the pooled F1 gap with the per-chip table: if the gap comes from one or two chips, a different chip selection could change it; if most chips agree, it is more robust, though eight clustered chips from two regions still cannot establish a national ranking. The defensible sentence names the chips, the systems and the numbers, then says what the sample cannot establish. If B beats C, report it: the hand-built baseline is a real competitor, and a foundation model is not required to win.

</details>"""

SEC10 = r"""## 10. Export, reload in a fresh process and verify · [Engineering]

**Exported artifact:** `head_adapter/` — `adapter.safetensors` (the head's four tensors only; no base weights, no training data) and `manifest.json` (format, base model revision and converted-weights digest, feature block, frame rule, normalisation, threshold, training configuration, selection basis, file sizes and SHA-256). A consumer needs the base model at the recorded revision plus this folder.

**Fresh reload:** a new process verifies the artifact's digests, rebuilds the base model strictly from verified files, recomputes the validation chips' features from the HDF5 files, and compares the reloaded head's scores with the evaluation's scores (tolerance 0.0001) and decisions (must be identical). It reopens the prediction GeoTIFFs and recomputes the metrics from them, and checks that each GeoTIFF sits exactly on its chip: EPSG:32651, 20 m north-up pixels, the chip's upper-left corner as origin and bounds one chip (2.56 km) wide.

**Prediction files, per validation chip:** `<chip>_C.tif` is the scored map (0 non-rice, 1 rice, 255 where the reference label is ignored, so it reproduces the evaluation); `<chip>_C_all_pixels.tif` gives C's decision for every pixel; `<chip>_C_score.tif` holds C's rice score in [0, 1] as float32, so you can apply a different threshold; `<chip>_observed_frames.tif` holds, per pixel, how many of C's four selected frames had a valid observation (0 = C's input there was entirely filled). The reload checks the first three against the evaluation's scores and the fourth against the chips. Passing shows the saved computation is reproducible from files; it does not show the model or the labels are right.

**Expected result:** a verification table whose checks are all true, a download link for `results.zip` (records, tables, figures, GeoTIFFs, the adapter, attribution and `checksums.json`; no source imagery or base weights), and then **this run's answer** to the driving question — the validation metrics, whether C scored above B, and the evidence level — followed by a conclusion scaffold."""

CELL_EXPORT = r"""run_stage('export')
run_stage('reload')
run_stage('report')
show_record('verification.json')
display(FileLink(str(ROOT / 'outputs' / 'results.zip')))
display(Markdown((ROOT / 'outputs' / 'conclusion.md').read_text(encoding='utf-8')))"""

SEC11 = r"""## 11. Conclude with evidence · [Evaluation practice]

The output above ends with **This run's answer**, written from the validation table the run just produced: it states whichever result occurred, with its evidence level. Read it, then complete the conclusion scaffold that follows it in your own words. A strong conclusion states the task and pixel population, reports every system on the same pixels — including a baseline that may have won — names the frame activity's result, uses the error maps to say *where* systems differ, and states the limits: eight validation chips from two regions, spatial clustering, map-derived labels, pipelines that differ in more than pretraining, the 20 m versus 30 m mismatch, the 320 m patch grid and unknown pretraining overlap. It also keeps three claims apart — finding some rice signal, beating the comparator under the declared metric, and being useful in practice — and makes only the first two.

<details>
<summary>Check your reasoning: what makes a conclusion strong? (open after writing yours)</summary>

Strong (written for a run in which B scored higher; swap the systems if yours did not): "On the retained labelled pixels of eight validation chips in Central and Eastern Visayas, the spectral-temporal LightGBM pipeline achieved higher pooled rice F1 than the frozen-Prithvi pipeline, *y* versus *x*. [Name the precision/recall pattern, for example: the Prithvi pipeline had higher precision but substantially lower recall.] These scores measure agreement with a published reference map, not independently established field accuracy. Because the pipelines differ in temporal information, missing-data treatment, spatial prediction architecture and training procedure, this comparison does not isolate the contribution of foundation-model pretraining. The development experiment favoured four frames under the prescribed settings [if yours did], but does not independently establish that phenological reasoning caused that advantage. The error maps show [where C and B differ: whole fields, edges or cloud-filled areas]. With eight clustered chips, this does not establish which approach is better nationally or whether either map is fit for operational use; next we would score a larger, spatially separated test set." Weak: "Prithvi is better at mapping rice" from one pooled number.

</details>

**Transfer:** Lessons [L3](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3-lightgbm-baseline) [11] and [L3B](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3b-alphaearth-baseline) [12] show the feature-engineering and embedding routes in depth; Section 12 lets you score your own chips with the exported head and, with enough labelled sites, adapt a new head under the same protocol."""

SEC12 = r"""## 12. Optional: bring your own chips · [Engineering]

Switched off by default, so Run all never waits for an upload. After Run all has finished:

1. Tick `USE_BYOD`.
2. Either set `BYOD_PATH` to a folder or `.zip` already in this runtime, or leave it empty to get an upload dialog (Colab).
3. Run this cell (the code cell below). You do not need to Run all again: it uses the head exported earlier in this session. If the runtime has restarted since then, Run all first.

**Required format — this dataset's own schema (checked before any model runs):**

| Key | Contract |
|---|---|
| `S2` | float32 `[24, 6, 128, 128]`: 24 equal calendar bins × bands B2, B3, B4, B8A, B11, B12 in that order; surface reflectance (digital number × 0.0001); NaN allowed where `S2_valid` = 0 |
| `S2_valid` | 0/1 `[24, 128, 128]` |
| `label` | uint8 `[128, 128]`: 0 non-rice, 1 rice, 255 ignore — needed for scoring and adaptation; optional for prediction only |
| attributes (optional) | Location, either this dataset's `affine` (JSON `[a, b, c, d, e, f]`) with `crs` = `EPSG:32651`, used only for a 20 m north-up grid, or plain `x_min`, `y_max` (EPSG:32651 metres, upper-left corner); a chip without a usable location is still scored, its GeoTIFF has no location and the notebook says so. Site: `group_id` or `group`. Without either, the site is the file name without a trailing `_YYYY`; that grouping is **unverified** (a renamed file becomes a new site), so the summary lists those chips and the cell warns |

One `.h5` file per chip-year with a unique file name, at most 200 files and 4 GiB in total; a `.zip` with an absolute or `..` member is refused. Values above 2.0 mean the file stores reflectance × 10,000; it is refused, not rescaled. Other keys (AlphaEarth, `label_raw`) are ignored.

**What happens:** every file is validated with the same rules as the sample data (a failure names the file and the broken rule); every chip is scored with the exported head at its frozen threshold (GeoTIFFs and, if labelled, per-chip metrics). With at least 6 labelled chips from at least three sites, the notebook also adapts a new head under the same protocol: sites are split into fit, development and test by a declared hash rule (adaptation is not run if two labelled chips have identical content, or if georeferenced chips that overlap on the ground carry different site ids, because either would put one place in two roles; the summary's `grouping` and `spatial_audit` records show the site sources and the minimum distances between roles), the epoch and threshold are chosen on development, the test sites are scored against the all-non-rice baseline, and the adapted head is exported. Its saved weights are then read back and re-applied to the cached features of the test chips: the scores must agree within 0.0001 and every rice/non-rice decision must be identical (**head-serialization parity**). If that check fails, the cell stops with an error, the adapted head is discarded and no `byod_results.zip` is written. This check covers the saved head only; unlike Section 10, it does not rebuild the base model or recompute features in a fresh process.

**Expected result / What to notice:** a per-chip table (bins, filled share, predicted rice share, whether the GeoTIFF is georeferenced and, for labelled chips, the scores) and a summary; with enough labelled sites, the adapted head's test result beside the all-non-rice baseline and the exported canonical head on the same test sites, so you can see whether adapting helped; and a download link for `byod_results.zip`. A refused upload prints one `Refused:` message naming each file and the broken rule, shows no results and keeps none from earlier attempts; fix the file and run this cell again (no Run all needed).

**Privacy:** your files stay in this runtime; nothing is sent to any other service. Do not upload confidential, restricted, licensed or personal data unless you are authorised to process it in Google Colab."""

CELL_BYOD = r"""USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
if USE_BYOD:
    sources = [Path(BYOD_PATH)] if BYOD_PATH else []
    if not sources:
        from google.colab import files  # only on Colab, only after the sample path has completed
        sources = [Path(name).resolve() for name in files.upload()]
    # The stage empties its input folder and its outputs first, refuses unsafe archives and duplicate file names,
    # validates every chip, then scores, adapts, exports and bundles byod_results.zip.
    if run_stage('byod', '--input', *[str(source) for source in sources], refusal_ok=True):
        show_table('byod/byod_canonical_head.csv')
        summary = json.loads((ROOT / 'outputs' / 'byod' / 'byod_summary.json').read_text())
        adaptation = summary.get('adaptation')
        if isinstance(adaptation, dict):  # the test-site comparison first, so a long record never hides it
            display_rows([{'system': name, **adaptation[key]} for name, key in (
                ('all non-rice', 'test_all_non_rice'), ('exported canonical head', 'test_canonical_head'),
                ('adapted head', 'test_adapted_head'))],
                [('system', 'Test sites'), ('f1', 'F1'), ('iou', 'IoU'), ('precision', 'Precision'), ('recall', 'Recall'),
                 ('scored_pixels', 'Scored px')])
        else:
            print('Adaptation:', adaptation)
        print(json.dumps(summary, indent=1)[:6000])  # the full record is byod/byod_summary.json in byod_results.zip
        display(FileLink(str(ROOT / 'outputs' / 'byod_results.zip')))
    else:  # a refused upload is a recoverable outcome, not a crash: nothing was scored and no older result is shown
        print('Nothing was scored, and no BYOD results from earlier attempts are kept. Fix the file(s) named above, '
              'then run this cell again (no Run all needed).')
else:
    print('BYOD not run (USE_BYOD = False).')"""

TROUBLE = r"""## Troubleshooting

| Observation | Appropriate response |
|---|---|
| "Select a fresh T4 GPU runtime" or CUDA not detected | Runtime → Change runtime type → T4 GPU, then Runtime → Run all. |
| "Supported runtime: Google Colab (Linux x86_64)" | The lock is built for manylinux x86_64; use Colab (or another Linux x86_64 GPU runtime). |
| Download, size or SHA-256 mismatch (uv, packages, model, INDEX, chips) | Re-run once for a transient network error. A persistent mismatch means a pinned source changed: stop and report it; do not remove the check. |
| Kaggle download fails | The dataset is public and needs no account; retry once. If Kaggle is unreachable, the notebook cannot run. |
| Out of disk | Start a fresh runtime with at least 12 GiB free. |
| GPU out of memory | Use a fresh T4 runtime; do not run the optional fine-tune with a larger batch. Never record a skipped stage as a success. |
| "run the earlier stages in order" | A stage's inputs are missing: Runtime → Run all from the top. |
| "validation was already scored" / fine-tune refuses | Scoring happens once per run directory. Start a fresh runtime to change a development choice. |
| B beats C, or C beats B | Preserve the result and examine the error maps; either outcome is valid. |
| Reload parity fails | Preserve `verification.json` and the logs; the release cannot be qualified until the mismatch is explained. |
| BYOD adaptation fails verification (head-serialization parity) | Not a problem with your files and not a `Refused:` message: the adapted head did not reproduce its own scores after saving. Nothing was bundled. Keep `byod.log` and `outputs/byod/byod_verification_failure.json` and report it. |
| BYOD file refused (`Refused:` in the BYOD cell) | Read the message: it names the file and the key, shape, dtype, value range or label rule that failed. Nothing was scored. Fix the file and run the BYOD cell again; you do not need to Run all again. |"""

LICENCES = r"""## Data, model, attribution and licences

**Data:** `kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching`, version 1 (public, CC BY 4.0; later versions change only the card and rights wording, with identical data files), verified by `INDEX.jsonl` SHA-256 `e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d` and every chip's SHA-256. Component attribution, from the dataset's rights file:

- **Rice labels:** Chen et al., *20 m resolution maps of paddy rice cropping intensity and planting dates in Monsoon Asia from 2018 to 2021* [4], https://doi.org/10.5281/zenodo.15124936, CC BY 4.0 — the annual `PADDYINTENSITY<year>` layer of each chip's year (not the four-year composite), codes 1/2/3 → rice and 0 → non-rice, nearest-neighbour resampling to the 20 m EPSG:32651 chip grid, and a 255 band one pixel wide on each side of every class edge. Label recipe SHA-256 `7adf68fe9d23622a1cf008f71cacd7c5006ed1377e34a5fc8c17c046953dc63e` (checked against every INDEX row); public rebuild kit `recipe-kit.zip` SHA-256 `976530ffa324f0d701051d757214a5dca977621d5a3e752b4f11bac0dbf7f032` in the parent Kaggle dataset `kurtvalcorza/rice-philippines-s2-alphaearth-v1` (version 2; it holds `build_samples.py` and `dataset_contract.py`). The teaching dataset itself does not carry the kit. The Chen rasters' whole-file integrity was not verified at build time (Philippine crops were range-read from the authors' mirror), and the build re-derived the labels with the same code rather than an independent implementation.
- **Optical inputs:** Contains modified Copernicus Sentinel data 2019–2021 (cloud/quality masking, six-band selection, 20 m resampling, 24 annual median bins), under the Copernicus Sentinel data terms.
- **AlphaEarth Foundations Satellite Embedding** (Google and Google DeepMind) [8], CC BY 4.0 — present in the dataset files; not read or shown by this notebook.
- **Regional boundaries:** geoBoundaries PHL ADM1 (2020 vintage; NAMRIA/PSA/OCHA), CC BY 3.0 IGO.
- **Dataset compilation:** Kurt Valcorza, CC BY 4.0. Component terms continue to apply.

**Model:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` at revision `9eb1b1102806593963daa333bcc491b1c6f8562f` (checkpoint SHA-256 `faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7`), Apache-2.0 [3], [10]. The carried repository code is Apache-2.0. The exported head contains no base weights and no training data.

**AI Assistance Disclosure:** Generative AI assisted this notebook's code and technical writing under maintainer direction, including checking scholarly references through the Scite literature service. The maintainer remains responsible for reviewing the implementation, validating results and making release decisions. AI assistance is not independent verification, provider endorsement or release approval.

## References

[1] X. Xiao et al., "Mapping paddy rice agriculture in southern China using multi-temporal MODIS images," *Remote Sens. Environ.*, vol. 95, no. 4, pp. 480–492, 2005, doi: 10.1016/j.rse.2004.12.009. [Online]. Available: https://doi.org/10.1016/j.rse.2004.12.009

[2] M. Drusch et al., "Sentinel-2: ESA's optical high-resolution mission for GMES operational services," *Remote Sens. Environ.*, vol. 120, pp. 25–36, 2012, doi: 10.1016/j.rse.2011.11.026. [Online]. Available: https://doi.org/10.1016/j.rse.2011.11.026

[3] D. Szwarcman et al., "Prithvi-EO-2.0: A versatile multitemporal foundation model for Earth observation applications," *IEEE Trans. Geosci. Remote Sens.*, vol. 64, pp. 1–20, 2026, doi: 10.1109/TGRS.2025.3642610. [Online]. Available: https://doi.org/10.1109/TGRS.2025.3642610

[4] Y. Chen, S. Liang, W. Li, H. Ma, H. Fang, and J. Xu, "20 m resolution maps of paddy rice cropping intensity and planting dates in Monsoon Asia from 2018 to 2021," Zenodo, 2025, data set, doi: 10.5281/zenodo.15124936. [Online]. Available: https://doi.org/10.5281/zenodo.15124936

[5] D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017, doi: 10.1111/ecog.02881. [Online]. Available: https://doi.org/10.1111/ecog.02881

[6] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432. [Online]. Available: https://doi.org/10.1371/journal.pone.0118432

[7] P. Ploton et al., "Spatial validation reveals poor predictive performance of large-scale ecological mapping models," *Nature Commun.*, vol. 11, no. 1, 2020, doi: 10.1038/s41467-020-18321-y. [Online]. Available: https://doi.org/10.1038/s41467-020-18321-y

[8] C. F. Brown et al., "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data," 2025, arXiv:2507.22291, doi: 10.48550/arXiv.2507.22291. [Online]. Available: https://doi.org/10.48550/arXiv.2507.22291

[9] K. Valcorza, "Philippines Rice S2 AlphaEarth v1 Teaching," Kaggle, data set, version 1, Oct. 2026. [Online]. Available: https://www.kaggle.com/datasets/kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching/versions/1

[10] IBM and NASA, "Prithvi-EO-2.0-300M," Hugging Face model card, revision 9eb1b1102806593963daa333bcc491b1c6f8562f. [Online]. Available: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/tree/9eb1b1102806593963daa333bcc491b1c6f8562f

[11] K. Valcorza, "Rice from space — Lesson 3: LightGBM baseline," Kaggle notebook, version 4, Oct. 2026. [Online]. Available: https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3-lightgbm-baseline

[12] K. Valcorza, "Rice from space — Lesson 3B: AlphaEarth baseline," Kaggle notebook, version 3, Oct. 2026. [Online]. Available: https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3b-alphaearth-baseline

**Data and software sources:** [Teaching dataset on Kaggle, version 1](https://www.kaggle.com/datasets/kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching/versions/1) · [Prithvi-EO-2.0-300M model card at the pinned revision](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/tree/9eb1b1102806593963daa333bcc491b1c6f8562f) · [Host repository](https://github.com/kurtvalcorza/prithvi-eo-feature-extraction-pipeline) · [Lesson 3](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3-lightgbm-baseline) and [Lesson 3B](https://www.kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3b-alphaearth-baseline) of *Rice from space* · DIMER Notebook Specification 2.4 (maintained in the DIMER fleet specifications) · [Reference capstone](https://github.com/kurtvalcorza/prithvi-flood-segmentation-pipeline/blob/main/tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb)"""

INFRA = {"cellView": "form", "jupyter": {"source_hidden": True}}
CELL_ID = "_cell_id"  # a cell added after release keeps every earlier cell's id: it carries its own id, not an index


def cells() -> list[tuple[str, str, dict]]:
    install = (CELL_INSTALL.replace("@@UV_URL@@", UV_WHEEL["url"]).replace("@@UV_BYTES@@", str(UV_WHEEL["bytes"]))
               .replace("@@UV_SHA256@@", UV_WHEEL["sha256"]).replace("@@PYTHON@@", MANAGED_PYTHON))
    return [
        ("markdown", TITLE, {}), ("markdown", PROBLEM, {CELL_ID: "rice-problem"}), ("markdown", GLOSSARY, {}),
        ("markdown", SEC1, {}),
        ("code", CELL_RUNTIME, INFRA),
        ("code", carrier_cell(carried_files()), {**INFRA, "dimer": {"embedded_sources": True}}),
        ("markdown", INSTALL_MD, {}), ("code", install, INFRA),
        ("markdown", SEC2, {}), ("code", CELL_PREPARE, {}), ("markdown", AFTER_PREPARE, {}),
        ("markdown", SEC3, {}), ("code", CELL_BASELINE, {}), ("markdown", AFTER_BASELINE, {}),
        ("markdown", SEC4, {}), ("code", CELL_SOURCE_INPUT, {CELL_ID: "rice-source-input"}), ("code", CELL_FEATURES, {}), ("markdown", AFTER_FEATURES, {}),
        ("markdown", SEC5, {}), ("code", CELL_SOURCE_HEAD, {CELL_ID: "rice-source-head"}), ("code", CELL_TRAIN, {}), ("markdown", AFTER_TRAIN, {}),
        ("markdown", SEC6, {}), ("code", CELL_ACTIVITY, {}), ("markdown", AFTER_ACTIVITY, {}),
        ("markdown", SEC7, {}), ("code", CELL_FINETUNE, {}),
        ("markdown", SEC8, {}), ("code", CELL_EVALUATE, {}),
        ("markdown", SEC9, {}), ("code", CELL_FIGURES, {}), ("markdown", AFTER_FIGURES, {}),
        ("markdown", SEC10, {}), ("code", CELL_EXPORT, {}),
        ("markdown", SEC11, {}), ("markdown", SEC12, {}), ("code", CELL_BYOD, {}),
        ("markdown", TROUBLE, {}), ("markdown", LICENCES, {}),
    ]


def build() -> str:
    files = carried_files()
    nb_cells = []
    index = 0
    for kind, source, metadata in cells():
        metadata = dict(metadata)
        cell_id = metadata.pop(CELL_ID, None)
        if cell_id is None:
            cell_id, index = f"rice-{index:02d}", index + 1
        cell = {"cell_type": kind, "id": cell_id, "metadata": metadata, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU",
            "colab": {"name": NOTEBOOK.name, "gpuType": "T4", "provenance": []},
            "dimer": {
                "notebook_spec": "2.4", "notebook_profile": "E2E", "notebook_mode": "WORKSHOP", "notebook_kind": "CAPSTONE",
                "standalone": True,
                "requires_dimer_worker": False, "release_status": "Candidate",
                "reference_exemplar": "DIMER_Philippines_Flood_Mapping_Capstone (NOTEBOOK_SPEC 25.15)",
                "owner_decisions": {"D1": "E2E with BYOD of HDF5 chips in the dataset schema",
                                    "D2": "frozen-encoder head canonical; opt-in fine-tune extension",
                                    "D3": "fixed block 12 (FEATURE_LAYER = 11)"},
                "generated_from": {
                    "repository": "kurtvalcorza/prithvi-eo-feature-extraction-pipeline", "generator": GENERATOR,
                    "files": {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in files.items()},
                    "post_generation_revisions": [],
                },
            },
        },
        "cells": nb_cells,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed notebook is not up to date")
    args = parser.parse_args(argv)
    text = build()
    if args.check:
        current = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
        if current != text:
            print(f"STALE: {NOTEBOOK} differs from the generator output; run {GENERATOR}", file=sys.stderr)
            return 1
        print(f"OK: {NOTEBOOK.name} is up to date ({len(text.encode('utf-8'))} bytes)")
        return 0
    NOTEBOOK.write_bytes(text.encode("utf-8"))
    print(f"wrote {NOTEBOOK} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
