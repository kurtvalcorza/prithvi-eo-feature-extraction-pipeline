# ruff: noqa: E501
"""Generate the DIMER AI for Earth Observation and Climate Applications workshop notebook.

Reads the workshop draft (`draft_source.json`, notebook JSON, outputs stripped: cell ids and the cells kept unchanged) and the
pinned test-chip records exported from the prithvi-* pipeline repositories (`*_records.json`), and writes
`tutorials/DIMER_AI_for_Earth_Observation_and_Climate_Applications_Workshop.ipynb`. `--check` exits non-zero when the
committed notebook differs from what the generator produces. Do not edit the notebook by hand.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK_NAME = "DIMER_AI_for_Earth_Observation_and_Climate_Applications_Workshop.ipynb"
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("src", nargs="?", type=Path, default=HERE / "draft_source.json")
_parser.add_argument("out", nargs="?", type=Path, default=REPO / "tutorials" / NOTEBOOK_NAME)
_parser.add_argument("--check", action="store_true")
_args = _parser.parse_args()
SRC, OUT = _args.src, _args.out
orig = json.loads(SRC.read_text(encoding="utf-8"))
orig_cells = {c["id"]: c for c in orig["cells"]}


def records(name):
    return [r for r in json.loads((HERE / f"{name}_records.json").read_text()) if r[1] == "test"]


def fmt_records(rows):
    return "[\n" + "".join(f"        {tuple(r)!r},\n" for r in rows) + "    ]"


cells = []

CROP_LOADER = r'''def load_crop_model():
    """Audited checkpoint -> weights-only unpickle (three data-only NumPy metadata globals bound to inert stand-ins)
    -> drop the training-only auxiliary head -> strict load -> SafeTensors (pinned digest) -> fresh strict load."""
    ckpt, audit = fetch_checkpoint("crop")
    bindings = [
        (_meta_scalar, "numpy.core.multiarray.scalar"),
        (_meta_dtype, "numpy.dtype"),
        (_meta_encode, "_codecs.encode"),
        _MetaValue,
    ]
    with torch.serialization.safe_globals(bindings):
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    kept = {k: v for k, v in payload["state_dict"].items() if not k.startswith("auxiliary_head.")}
    del payload
    model = PrithviCropSegmenter()
    model.load_state_dict(kept, strict=True)
    del kept
    fresh, converted = convert_and_reload("crop", model, PrithviCropSegmenter)
    del model
    gc.collect()
    return fresh, ckpt, converted, audit


'''


CROP_NORMALISE = r'''def crop_normalise(scene_dn):
    """(18, 224, 224) HLS digital numbers returned by validate_crop_scene -> standardised (1, 6, 3, 224, 224). The
    units are declared as digital numbers, so already-converted data is never rescaled."""
    x = validate_crop_scene(scene_dn, name="crop model input", units="dn", log=False).reshape(1,18,224,224)
    mean = np.asarray(CROP_MEANS * 3, dtype=np.float32)[None,:,None,None]
    std = np.asarray(CROP_STDS * 3, dtype=np.float32)[None,:,None,None]
    flat = (x - mean) / std
    # Reproduce the historical training layout exactly.
    return np.ascontiguousarray(flat.reshape(1,6,3,224,224))

'''

CROP_PREDICT = r'''def predict_crop(model, scene_dn):
    """(18, 224, 224) validated HLS digital numbers -> class map (224, 224) uint8, softmax scores (13, 224, 224) and
    class fractions. The class scores are checked (shape, finite values, probabilities, label domain) before any map
    is formed: an invalid output raises InvalidModelOutput."""
    x = torch.from_numpy(crop_normalise(scene_dn)).to(DEVICE)
    with torch.inference_mode(), torch.autocast(
        device_type=DEVICE.split(":")[0],
        dtype=torch.float16,
        enabled=DEVICE.startswith("cuda")
    ):
        logits = model(x)
    logits = logits.float()
    n_classes = len(CROP_CLASS_NAMES)
    check_class_scores(logits, name="crop model output", batch=1, num_classes=n_classes, spatial=(224, 224))
    prob = torch.softmax(logits, dim=1)
    check_probabilities(prob, name="crop model output")
    scores = prob[0].cpu().numpy()
    mask = scores.argmax(axis=0).astype(np.uint8)
    check_class_map(mask, name="crop model output", num_classes=n_classes)
    fractions = {CROP_CLASS_NAMES[i]: float((mask == i).mean()) for i in range(n_classes)}
    return mask, scores, fractions'''


def md(cid, text):
    cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": text.strip("\n")})


def code(cid, text):
    cells.append(
        {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}
    )


def keep(cid):
    c = dict(orig_cells[cid])
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    cells.append(c)


# ------------------------------------------------------------------------------------------------------------------
md("762f85f8", r"""
# DIMER Workshop: AI for Earth Observation and Climate Applications

**Profile:** `MULTI-CAPABILITY`
**Mode:** `WORKSHOP`
**Notebook specification:** `2.1`
**Status:** Candidate workshop carrier (clean-runtime execution evidence pending)
**Recommended runtime:** NVIDIA Tesla T4 or equivalent

This standalone workshop demonstrates how Earth-observation foundation models and task-specialized models can be used to **represent satellite observations**, **map floods**, **map burn scars**, and **classify crops/land cover through time**.

The canonical path executes locally in this notebook runtime. It does **not** clone a Git repository, import or download DIMER repository source, call a DIMER worker or API, require a user token or login, or require a file upload.

### Run all

The default path:

1. installs the pinned runtime into an isolated Python environment and runs the rest of the notebook there (see §1), so a single **Run all** completes without a restart;
2. downloads the four pinned model checkpoints, three unseen demonstration scenes, and small **labelled** test samples for flood, burn-scar and crop mapping, each verified by SHA-256;
3. validates every input against its data contract before any model runs;
4. runs four live DIMER model capabilities locally, one model on the GPU at a time;
5. **evaluates** flood, burn-scar and crop maps against reference labels and against naive baselines;
6. runs each task model on a new scene that no evaluation step used;
7. writes machine-readable predictions, metrics and provenance under `outputs/`, and creates a report bundle.

### Bring Your Own Data

After the automatic sample workflow, an optional BYOD section accepts compatible GeoTIFF input using the same local validation and inference semantics. BYOD is disabled by default and never interrupts **Run all**.

### Learning goals

By the end of this notebook you should be able to:

- distinguish RGB imagery from multispectral and multi-temporal Earth-observation data;
- explain what an EO foundation-model embedding represents;
- inspect masked reconstruction as a self-supervised learning objective;
- run local flood, burn-scar, and crop/land-cover mapping;
- evaluate a segmentation map with IoU, F1, precision and recall against a naive baseline, and explain why pixel accuracy alone misleads;
- interpret pixel maps without confusing them with calibrated probabilities or operational products;
- identify sensor, geographic, seasonal, and spatial-resolution sources of domain shift;
- export machine-readable predictions and reproducibility metadata; and
- repeat an applicable inference workflow on your own compatible scene.

### Important scope boundary

This is an **Earth-observation and climate-impact applications** workshop. It does not perform meteorological forecasting or climate simulation. Flood segmentation does not forecast when floods will occur; burn-scar segmentation does not predict wildfire ignition; crop mapping does not predict yield. Earth-system forecasting with **Aurora** belongs in a separate workshop.

All results produced here are **tutorial/sanity evidence** measured on small samples, not benchmark or operational-validation evidence.

**AI Use Disclosure:** Generative AI assisted with this notebook’s code and instructional content under maintainer direction. The maintainer remains responsible for review, validation, and release decisions. AI-generated material may contain errors; validation claims are limited to documented runs and configurations. AI use does not imply independent verification, provider endorsement, or release approval.
""")

md("ec4ea156", r"""
## 0. Prerequisites and resource envelope

Use a fresh GPU runtime. The reference release target is **Google Colab with a Tesla T4**. A GPU is required for the supported path; CPU execution works for the smaller stages but is slow and unsupported.

The default path downloads (sizes are the pinned byte counts of the upstream files):

- 1.33 GB — Prithvi-EO-2.0-300M foundation checkpoint;
- 1.28 GB — Prithvi flood-mapping checkpoint;
- 1.30 GB — Prithvi burn-scar checkpoint;
- 1.68 GB — Prithvi-EO-1.0 crop-classification checkpoint;
- about 27 MB — 12 labelled Sen1Floods11 test chips, fetched one object at a time;
- 2.65 GB — the HLS Burn Scars archive, **streamed**: only the 12 pinned test chips and masks (about 80 MB) are kept, the archive itself is never written to disk;
- 1.18 GB — the multi-temporal crop validation archive, streamed the same way (12 pinned chips and masks kept); and
- about 19 MB — the four foundation scenes and three unseen demonstration scenes.

Expect roughly 9–10 GB of network transfer and about 8 GB of disk. Streaming the two archives takes several minutes. Only one large model is resident on the GPU at a time. These are estimates from the pinned file sizes, not measurements of a particular runtime.

Why the archives are streamed in full: the labelled burn-scar and crop chips are only published inside those two archives, and the pinned test members are spread through them (a probe on 2026-09-25 found the last crop member in the final 2 MB of its archive), so they cannot be fetched individually. Every kept member is checked against its pinned size and SHA-256, and the whole archive is checked against its pinned digest once the stream ends.
""")

md("aa757eb6", r"""
## 1. Install and verify the pinned runtime (isolated environment)

Hosted runtimes (Colab, Kaggle) import some packages, such as NumPy, before the first cell runs. The pinned stack needs newer versions (TerraTorch 1.2.13 requires `numpy>=2.2`; PyTorch 2.14 requires `cuda-bindings>=13`), and Python cannot swap a module that is already loaded, so installing the pins into the notebook's own Python would leave mixed versions or need a restart.

This notebook therefore leaves the kernel's packages untouched:

1. the first cell installs the exact pins into a separate environment (`dimer_eo_env/`, created with `uv` from the kernel's own Python);
2. the second cell starts one Python process in that environment and routes **every later code cell** to it. Printed output, tables and figures come back to the notebook as usual, variables persist from cell to cell, and an error stops **Run all** exactly as it would in the kernel;
3. the third cell, the first to run in the isolated environment, records the versions it actually imported.

These first two cells are marked `# dimer: kernel cell` and are the only cells that run in the kernel. If you re-run a single cell later, it still runs in the isolated environment with the variables created so far. To start over, restart the session and choose **Run all**.

The environment variable `DIMER_NOTEBOOK_CI_PREINSTALLED=1` lets an executor that has already installed exactly these pins run every cell in its own kernel instead. It never selects data or models.
""")

code("a349a890", r'''
# @title Install the tested notebook runtime into an isolated environment
# dimer: kernel cell (runs in the notebook kernel, not in the isolated environment)
import os
import shutil
import subprocess
import sys
from pathlib import Path

PINS = [
    "torch==2.14.0",
    "torchvision==0.29.0",
    "terratorch==1.2.13",
    "tifffile==2026.9.15",
    "numpy==2.5.3",
    "safetensors==0.8.0",
    "huggingface-hub==1.32.0",
    "matplotlib>=3.9,<3.11",
    "pandas>=2.2,<3.0",
]
UV_PIN = "uv==0.8.17"

SKIP_INSTALL = os.environ.get("DIMER_NOTEBOOK_CI_PREINSTALLED") == "1"
EO_ENV = Path(os.environ.get("DIMER_EO_ENV", "dimer_eo_env")).resolve()
EO_PYTHON = EO_ENV / "bin" / "python"

if SKIP_INSTALL:
    print("DIMER_NOTEBOOK_CI_PREINSTALLED=1: the pins are already installed; the notebook runs in this kernel.")
else:
    uv = shutil.which("uv")
    if uv is None:
        # uv is a self-contained binary; installing it does not touch any package this kernel has imported.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", UV_PIN], check=True)
        from uv import find_uv_bin

        uv = find_uv_bin()
    if not EO_PYTHON.is_file():
        subprocess.run([uv, "venv", "--quiet", "--python", sys.executable, str(EO_ENV)], check=True)
    subprocess.run([uv, "pip", "install", "--quiet", "--python", str(EO_PYTHON), *PINS], check=True)
    print({"isolated_environment": str(EO_ENV), "python": str(EO_PYTHON)})
''')

code("5d1c7e0a", r'''
# @title Route the remaining cells to the isolated environment
# dimer: kernel cell (runs in the notebook kernel, not in the isolated environment)
import signal
from multiprocessing.connection import Connection

# The worker runs in the isolated environment. It executes each routed cell in one persistent namespace and sends
# back printed text, displayed objects and matplotlib figures, so every cell behaves as it would in the kernel.
_WORKER_SOURCE = r"""
import ast, base64, builtins, io, linecache, os, signal, sys, traceback, types
from multiprocessing.connection import Connection

_send = Connection(int(sys.argv[1]), readable=False)
_recv = Connection(int(sys.argv[2]), writable=False)


class _Stream(io.TextIOBase):
    def __init__(self, name):
        self._name = name

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def isatty(self):
        return False

    def write(self, text):
        if text:
            _send.send(("stream", self._name, str(text)))
        return len(text)


sys.stdout, sys.stderr = _Stream("stdout"), _Stream("stderr")


def _figure_bundle(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"), "text/plain": repr(fig)}


def _flush_figures():
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for number in plt.get_fignums():
        _send.send(("display", _figure_bundle(plt.figure(number))))
    plt.close("all")


def _mimebundle(obj):
    if hasattr(obj, "savefig"):
        return _figure_bundle(obj)
    data = {"text/plain": repr(obj)}
    for method, mime in (("_repr_html_", "text/html"), ("_repr_markdown_", "text/markdown"), ("_repr_png_", "image/png")):
        render = getattr(obj, method, None)
        if callable(render):
            try:
                value = render()
            except Exception:
                value = None
            if isinstance(value, bytes):
                value = base64.b64encode(value).decode("ascii")
            if value is not None:
                data[mime] = value
    return data


def display(*objects, **kwargs):
    for obj in objects:
        _send.send(("display", _mimebundle(obj)))


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot

    matplotlib.pyplot.show = lambda *args, **kwargs: _flush_figures()
except ImportError:
    pass

if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":
    # google.colab only exists in the kernel; forward the BYOD upload dialog to it.
    def _upload():
        _send.send(("upload",))
        reply = _recv.recv()
        if reply[1] is None:
            raise RuntimeError("The notebook kernel could not open the upload dialog.")
        return reply[1]

    try:
        import google
    except ImportError:
        google = types.ModuleType("google")
        google.__path__ = []
        sys.modules["google"] = google
    _colab = types.ModuleType("google.colab")
    _files = types.ModuleType("google.colab.files")
    _files.upload = _upload
    _colab.files = _files
    google.colab = _colab
    sys.modules["google.colab"] = _colab
    sys.modules["google.colab.files"] = _files

_main = types.ModuleType("__main__")
_main.__dict__.update(__builtins__=builtins, display=display)
sys.modules["__main__"] = _main
_count = 0
while True:
    # An interrupt only lands inside a running cell; between cells it is ignored.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        message = _recv.recv()
    except EOFError:
        break
    if message[0] != "run":
        continue
    _count += 1
    filename = f"<isolated cell {_count}>"
    source = message[1]
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        tree = ast.parse(source, filename)
        tail = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, filename, "exec"), _main.__dict__)
        if tail is not None:
            value = eval(compile(tail, filename, "eval"), _main.__dict__)
            if value is not None:
                display(value)
        _flush_figures()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _send.send(("done",))
    except BaseException as exc:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        frames = exc.__traceback__.tb_next if exc.__traceback__ is not None else None
        _send.send(("error", "".join(traceback.format_exception(type(exc), exc, frames)), f"{type(exc).__name__}: {exc}"))
"""


class IsolatedCellError(RuntimeError):
    """A routed cell raised inside the isolated environment; its traceback is printed above."""


class IsolatedRuntime:
    """One persistent worker process in the isolated environment, fed one cell at a time."""

    def __init__(self, python, display=None):
        to_kernel_r, to_kernel_w = os.pipe()
        to_worker_r, to_worker_w = os.pipe()
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
        env["DIMER_KERNEL_IS_COLAB"] = "1" if "google.colab" in sys.modules else "0"
        self.proc = subprocess.Popen(
            [str(python), "-c", _WORKER_SOURCE, str(to_kernel_w), str(to_worker_r)],
            pass_fds=(to_kernel_w, to_worker_r),
            env=env,
            start_new_session=True,  # interrupts reach the worker only through run(), exactly once
        )
        os.close(to_kernel_w)
        os.close(to_worker_r)
        self._recv = Connection(to_kernel_r, writable=False)
        self._send = Connection(to_worker_w, readable=False)
        if display is None:
            from IPython.display import display
        self._display = display

    def _exited(self):
        return RuntimeError(
            f"The isolated environment's Python process exited (code {self.proc.wait()}); a crash of this kind is "
            "usually running out of memory. Restart the session and choose Run all again."
        )

    def run(self, source):
        try:
            self._send.send(("run", source))
        except OSError:
            raise self._exited() from None
        while True:
            try:
                message = self._recv.recv()
            except EOFError:
                raise self._exited() from None
            except KeyboardInterrupt:
                self.proc.send_signal(signal.SIGINT)
                continue
            kind = message[0]
            if kind == "stream":
                (sys.stdout if message[1] == "stdout" else sys.stderr).write(message[2])
            elif kind == "display":
                self._display(message[1], raw=True)
            elif kind == "upload":
                self._send.send(("upload", self._colab_upload()))
            elif kind == "error":
                sys.stderr.write(message[1])
                raise IsolatedCellError(message[2]) from None
            elif kind == "done":
                return

    @staticmethod
    def _colab_upload():
        try:
            from google.colab import files
        except ImportError:
            return None
        return files.upload()

    def close(self):
        self._send.close()
        self.proc.wait(timeout=30)


def _is_user_cell():
    # ipykernel transforms a cell before executing it; its caller knows whether this is a silent frontend request.
    frame = sys._getframe(2)
    while frame is not None:
        local = frame.f_locals
        if "silent" in local and "store_history" in local:
            return bool(local["store_history"]) and not bool(local["silent"])
        frame = frame.f_back
    return True


def _route_to_isolated_runtime(lines):
    source = "".join(lines)
    if not source.strip() or "# dimer: kernel cell" in source or not _is_user_cell():
        return lines
    return [f"_DIMER_EO_RUNTIME.run({source!r})\n"]


if SKIP_INSTALL:
    print("Routing disabled: the notebook runs in this kernel.")
else:
    _ip = get_ipython()
    _ip.input_transformers_cleanup[:] = [
        t for t in _ip.input_transformers_cleanup if getattr(t, "__name__", "") != "_route_to_isolated_runtime"
    ]
    if isinstance(globals().get("_DIMER_EO_RUNTIME"), IsolatedRuntime):
        _DIMER_EO_RUNTIME.close()
    _DIMER_EO_RUNTIME = IsolatedRuntime(EO_PYTHON)
    _ip.input_transformers_cleanup.append(_route_to_isolated_runtime)
    print(f"Every later code cell now runs in {EO_PYTHON} (pid {_DIMER_EO_RUNTIME.proc.pid}).")
''')

code("8e2b4f61", r'''
# @title Verify the tested runtime
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from huggingface_hub import hf_hub_download

RUNTIME = {
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "execution": "in-kernel (pins preinstalled)" if os.environ.get("DIMER_NOTEBOOK_CI_PREINSTALLED") == "1" else "isolated environment",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_gib": (
        round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None
    ),
    "numpy": np.__version__,
    "tifffile": importlib.metadata.version("tifffile"),
    "terratorch": importlib.metadata.version("terratorch"),
    "safetensors": importlib.metadata.version("safetensors"),
    "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    "matplotlib": matplotlib.__version__,
    "pandas": pd.__version__,
}
print(RUNTIME)

if not torch.cuda.is_available():
    print("WARNING: no CUDA device. The supported workshop runtime is a T4-class GPU; CPU execution is slow.")
''')

md("e94081be", r"""
## 2. Workshop controls

Keep the defaults on your first pass.

- `RUN_RECONSTRUCTION`, `RUN_FLOOD_MAPPING`, `RUN_BURNSCAR_MAPPING` and `RUN_CROP_MAPPING` skip a capability's checkpoint download and inference. They do **not** reduce data acquisition: §5 still downloads and validates every evaluation set, including the two streamed archives, so these switches are not a low-download mode.
- `USE_BYOD`, `BYOD_CAPABILITY`, `BYOD_PATH` and `BYOD_UNITS` configure the optional §12 inference on your own file. With `BYOD_UNITS = "auto"` the notebook decides once, from the valid (non-no-data) pixels, whether the file holds reflectance or HLS digital numbers, and stops with an error when the values are ambiguous; choose `"reflectance"` or `"dn"` to declare the units instead.
- `OUTPUT_DIR` receives this run's files. Each session gets a `RUN_ID`; the report bundle contains only the files this run wrote. Files left by earlier runs are listed in the manifest but never deleted or packaged.
""")

code("ba798aab", r'''
# @title Workshop controls
RUN_RECONSTRUCTION = True   # @param {type:"boolean"}
RUN_FLOOD_MAPPING = True    # @param {type:"boolean"}
RUN_BURNSCAR_MAPPING = True # @param {type:"boolean"}
RUN_CROP_MAPPING = True     # @param {type:"boolean"}

USE_BYOD = False            # @param {type:"boolean"}
BYOD_CAPABILITY = "burnscar" # @param ["embedding", "flood", "burnscar", "crop"]
BYOD_PATH = ""              # @param {type:"string"}
BYOD_UNITS = "auto"         # @param ["auto", "reflectance", "dn"]
OUTPUT_DIR = "outputs"      # @param {type:"string"}

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE.startswith("cuda")

np.random.seed(SEED)
torch.manual_seed(SEED)

# Execution precision per capability (recorded in the provenance export).
PRECISION = {
    "foundation": "float32",
    "flood": "float16 autocast" if USE_AMP else "float32",
    "burnscar": "float16 autocast" if USE_AMP else "float32",
    "crop": "float16 autocast" if USE_AMP else "float32",
}

import datetime
import uuid
from pathlib import Path

# One identity per session. Every file this run writes is registered under it (§4), and only those files are packaged.
# Re-running this cell (for example to switch on BYOD) keeps the session's RUN_ID.
RUN_ID = globals().get("RUN_ID") or (
    datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
)

OUTPUT_ROOT = Path(OUTPUT_DIR)
for sub in ("figures", "predictions", "embeddings", "metrics", "provenance"):
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
CACHE_ROOT = Path("workshop_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

print({"run_id": RUN_ID, "device": DEVICE, "precision": PRECISION, "output_root": str(OUTPUT_ROOT.resolve()), "seed": SEED})
''')

md("ebf78f3b", r"""
## 3. Immutable model, sample and evaluation identities

Every model is pinned to the upstream revision already reconciled with the live DIMER fleet, and every checkpoint is checked by byte size and SHA-256 before it is opened.

All four checkpoints are legacy PyTorch pickles. For each one the notebook statically audits the pickle globals (nothing is executed), loads it through PyTorch's `weights_only=True` unpickler, loads the weights **strictly** into an architecture built in this notebook, converts the result to SafeTensors, checks the converted file against the digest of the live DIMER profile, and finally loads a fresh model from that SafeTensors file. This is the same conversion the DIMER pipelines perform, so the workshop runs byte-identical weights. The crop checkpoint additionally binds three data-only NumPy metadata globals to inert stand-ins.

The labelled evaluation chips are pinned by size and SHA-256 per file, and the two archives they come from are pinned by size and SHA-256 as a whole. The selections are the test roles the DIMER pipeline repositories already use.
""")

flood_recs, burn_recs, crop_recs = records("flood-segmentation"), records("burnscar-segmentation"), records("crop-classification")
assert len(flood_recs) == len(burn_recs) == len(crop_recs) == 12

code("acff2ba5", r'''
TORCH_STATE_DICT_GLOBALS = frozenset(
    {"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage", "torch.LongStorage"}
)

MODEL_SPECS = {
    "foundation": {
        "model_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "revision": "9eb1b1102806593963daa333bcc491b1c6f8562f",
        "checkpoint": "Prithvi_EO_V2_300M.pt",
        "checkpoint_bytes": 1_326_658_396,
        "checkpoint_sha256": "faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7",
        "converted_bytes": 1_326_543_456,
        "converted_sha256": "15b8ed6dacf8b3ca02c542bb8eaffc92c7a7addc5351b3d967873ad3f73a24b0",
        "state_tensors": 398,
        "allowed_globals": frozenset({"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage"}),
        "license": "Apache-2.0",
    },
    "flood": {
        "model_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "revision": "91ce9d38086a80b078a192b374df758b8855b732",
        "checkpoint": "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        "checkpoint_bytes": 1_276_843_350,
        "checkpoint_sha256": "76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270",
        "converted_bytes": 1_276_749_320,
        "converted_sha256": "65e4377f96c651dff586bf6ef08c9b8d6c6b59c4dc05e5dfbfe320b264ac47fd",
        "state_tensors": 381,
        "allowed_globals": TORCH_STATE_DICT_GLOBALS,
        "license": "Apache-2.0",
    },
    "burnscar": {
        "model_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        "revision": "a3f2c410e45b8ac7417976614528a872f024d831",
        "checkpoint": "Prithvi_EO_V2_300M_BurnScars.pt",
        "checkpoint_bytes": 1_297_798_380,
        "checkpoint_sha256": "0c5f9334be9a75c9006387ab8f3dc05a55ea7fb5ef7956717316be57c62954d3",
        "converted_bytes": 1_297_682_024,
        "converted_sha256": "4209c5a013f90dfe372abb4ade3865e7a283d655ef17b1c7de95d36ecc033c3f",
        "state_tensors": 355,
        "allowed_globals": TORCH_STATE_DICT_GLOBALS,
        "license": "Apache-2.0",
    },
    "crop": {
        "model_id": "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification",
        "revision": "b53a88b8da673800b67c34a98a527b77076e7035",
        "checkpoint": "multi_temporal_crop_classification_Prithvi_100M.pth",
        "checkpoint_bytes": 1_680_468_041,
        "checkpoint_sha256": "37ed41637eccccec65ca2031324e2c03a4f168e1ea0ea71ad180910589fa018c",
        "converted_bytes": 537_722_508,
        "converted_sha256": "d1df8044700a0d1e00b11b1fbac66e0495cf6647e1632a858c003eaf4b5ce36d",
        "state_tensors": 98,
        "allowed_globals": TORCH_STATE_DICT_GLOBALS | {"numpy.core.multiarray.scalar", "numpy.dtype", "_codecs.encode"},
        "license": "Apache-2.0",
    },
}

# Unlabelled upstream scenes: the foundation time series, plus one NEW scene per task model that no evaluation
# step uses (each is absent from the pinned evaluation sets below). (path, bytes, sha256)
SCENES = {
    "foundation": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "repo_type": "model",
        "revision": "9eb1b1102806593963daa333bcc491b1c6f8562f", "license": "Apache-2.0 (model repository examples)",
        "files": [
            ("examples/Mexico_HLS.S30.T13REM.2018026T173609.v2.0_cropped.tif", 3_014_272, "e34c1e8f6b69092bbf16f87da1a0c2337e8e53f28d172d8076e2efab292b795d"),
            ("examples/Mexico_HLS.S30.T13REM.2018106T172859.v2.0_cropped.tif", 3_014_272, "a4b24a34d83d25cac7dbcb7742db3f5b1e4849e5773172c1f0fc43c541bcd3fd"),
            ("examples/Mexico_HLS.S30.T13REM.2018201T172901.v2.0_cropped.tif", 3_014_272, "fce050cc821ebec2974e85cfe702c0f093d74caf12196adb7ee88c8a30773d4f"),
            ("examples/Mexico_HLS.S30.T13REM.2018266T173029.v2.0_cropped.tif", 3_014_272, "f7f8c67c32027cd663f48226a5932c6c8119a55fb6e80a02636dea57f4733963"),
        ],
    },
    "flood": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11", "repo_type": "model",
        "revision": "91ce9d38086a80b078a192b374df758b8855b732", "license": "Apache-2.0 (model repository examples)",
        "files": [("examples/India_900498_S2Hand.tif", 2_151_620, "ee898621b387a731503a01960397599f209b45c268d4e088689928c704dbe968")],
    },
    "burnscar": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars", "repo_type": "model",
        "revision": "a3f2c410e45b8ac7417976614528a872f024d831", "license": "Apache-2.0 (model repository examples)",
        "files": [("examples/subsetted_512x512_HLS.S30.T10SFF.2018190.v1.4_merged.tif", 6_295_168, "b491445bcca5d23a534765ac9f8b24f4cb0c9a75a7254c969456d65a982207a5")],
    },
    "crop": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification-demo", "repo_type": "space",
        "revision": "5798a8e73368e2da0443d6e6333faca47663e6d6", "license": "Apache-2.0 (demonstration Space)",
        "files": [("chip_102_345_merged.tif", 1_808_174, "94c99f8d76093a21587c5a8757fcdae7bb054b2586b51fc570aef5b17fc3a591")],
    },
}

# Labelled evaluation samples: the test roles of the DIMER pipeline repositories' pinned sample sets.
EVAL_SETS = {
    "flood": {
        "dataset": "Sen1Floods11 v1.1 hand-labelled Sentinel-2 chips (official test split)",
        "base_url": "https://storage.googleapis.com/sen1floods11/v1.1/",
        "license": "CC BY 4.0 (Cloud to Street; Bonafilia et al., CVPRW 2020)",
        "label_semantics": "0 = no water, 1 = water, -1 = no data / cloud (ignored)",
        # (chip, role, image bytes, image sha256, label bytes, label sha256)
        "records": @@FLOOD@@,
    },
    "burnscar": {
        "dataset": "HLS Burn Scars (ibm-nasa-geospatial/hls_burn_scars)",
        "archive_url": "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/resolve/1864285e25010d346a842e4f068b1a1d4248ed6d/hls_burn_scars.tar.gz",
        "revision": "1864285e25010d346a842e4f068b1a1d4248ed6d",
        "archive_bytes": 2_645_552_531,
        "archive_sha256": "4e6f99a75cb2c500547b20662a15cbd531dc421376f815e91846ea542798e8e6",
        "license": "CC BY 4.0 (NASA IMPACT / University of Alabama in Huntsville)",
        "label_semantics": "0 = not burned, 1 = burn scar, -1 = no data (ignored)",
        # (scene, role, image member, bytes, sha256, mask member, bytes, sha256)
        "records": @@BURN@@,
    },
    "crop": {
        "dataset": "Multi-temporal crop classification (ibm-nasa-geospatial/multi-temporal-crop-classification), validation archive",
        "archive_url": "https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification/resolve/f285bb27c8f623a0fb6a44a6fd953c3ad34007d6/validation_chips.tgz",
        "revision": "f285bb27c8f623a0fb6a44a6fd953c3ad34007d6",
        "archive_bytes": 1_179_542_384,
        "archive_sha256": "d6e616cc008858a1935a8937e0cdf852d574754273b0655fb696bd29aebd2fd3",
        "license": "CC BY 4.0 (NASA IMPACT / IBM)",
        "label_semantics": "file values 0 = no data, 1..13 = class; loaded as 0..12 with no data = -1 (ignored)",
        "records": @@CROP@@,
    },
}

BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
S2_L1C_BAND_INDICES = (1, 2, 3, 8, 11, 12)  # Sentinel-2 L1C 13-band order -> the six Prithvi bands
IGNORE_INDEX = -1

print(pd.DataFrame([
    {
        "capability": k,
        "model_id": v["model_id"],
        "revision": v["revision"][:12],
        "license": v["license"],
        "new_scene": Path(SCENES[k]["files"][0][0]).name if k != "foundation" else "—",
        "eval_chips": len(EVAL_SETS[k]["records"]) if k in EVAL_SETS else "—",
    }
    for k, v in MODEL_SPECS.items()
]).to_string(index=False))
'''.replace("@@FLOOD@@", fmt_records(flood_recs)).replace("@@BURN@@", fmt_records(burn_recs)).replace("@@CROP@@", fmt_records(crop_recs)))

md("0b26fd6f", "## 4. Shared validation, integrity, acquisition, metric, and export helpers")

code("7fffe217", r'''
import gc
import hashlib
import http.client
import io
import json
import os
import pickletools
import shutil
import tarfile
import time
import urllib.request
import warnings
import zipfile
from collections.abc import Mapping

PREPROCESSING_LOG = []  # every band selection or unit conversion applied to an input (reported and exported)
UNIT_DECISIONS = {}     # input -> the unit decision made ONCE for it (declared, or inferred from valid pixels)
ARCHIVE_STATUS = {}     # evaluation archive -> whole-archive verification status of the cached members in use
PRODUCED_FILES = {}     # resolved path -> role ("canonical", "activity", "byod"): the files THIS run wrote


def note(name, change):
    PREPROCESSING_LOG.append({"input": name, "change": change})


def register_output(path, role="canonical"):
    """Record that this run wrote `path`. The report bundle packages registered files only."""
    PRODUCED_FILES[str(Path(path).resolve())] = role
    return Path(path)


def save_npz(path, role="canonical", **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return register_output(path, role)


def save_figure(path, role="canonical", **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **kwargs)
    return register_output(path, role)


def report_preprocessing(since=0):
    """Print the changes applied to inputs since log position `since`, grouped by kind of change."""
    rows = PREPROCESSING_LOG[since:]
    if not rows:
        print("preprocessing: no band selection or unit conversion was needed")
        return
    for change, group in pd.DataFrame(rows).groupby("change")["input"]:
        names = list(group)
        print(f"preprocessing: {change} — {len(names)} input(s): {', '.join(names[:3])}{' …' if len(names) > 3 else ''}")


# ---- integrity ---------------------------------------------------------------------------------------------------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, expected_bytes, expected_sha256, label):
    path = Path(path)
    size, digest = path.stat().st_size, sha256_file(path)
    if size != expected_bytes or digest != expected_sha256:
        raise ValueError(
            f"{label}: {size} bytes / sha256 {digest[:16]}… does not match the pinned "
            f"{expected_bytes} bytes / {expected_sha256[:16]}…. Delete the cached file and rerun; "
            "do not bypass the integrity check."
        )
    return digest


# ---- acquisition -------------------------------------------------------------------------------------------------
def download_hub_file(repo_id, filename, revision, repo_type="model"):
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, repo_type=repo_type))


def fetch_url(url, retries=4, timeout=120):
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dimer-eo-workshop/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


class _ResumableReader:
    """File-like HTTP body reader that hashes and counts every byte. If the connection drops or the body ends
    before the pinned length, it reopens the URL with a Range request at the current offset (up to `retries`
    times), so a multi-gigabyte stream survives a flaky network without starting over."""

    def __init__(self, url, total, retries=20):
        self.url, self.total, self.retries = url, total, retries
        self.sha, self.count, self.resumes = hashlib.sha256(), 0, 0
        self.response = None
        self._open()

    def _open(self):
        headers = {"User-Agent": "dimer-eo-workshop/1.0"}
        if self.count:
            headers["Range"] = f"bytes={self.count}-"
        self.response = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=300)
        if self.count and getattr(self.response, "status", None) != 206:
            raise ValueError(f"{self.url}: the server ignored the Range request, so the stream cannot resume")

    def read(self, size=-1):
        while True:
            error = None
            try:
                data = self.response.read(size)
            except (OSError, http.client.HTTPException) as exc:
                data, error = b"", exc
            if data or self.count >= self.total:
                self.sha.update(data)
                self.count += len(data)
                return data
            if self.retries <= 0:
                raise ValueError(f"{self.url}: connection lost at byte {self.count} of {self.total}; retries exhausted") from error
            self.retries -= 1
            self.resumes += 1
            self.response.close()
            time.sleep(3)
            self._open()

    def close(self):
        if self.response is not None:
            self.response.close()


def stream_pinned_members(url, archive_bytes, archive_sha256, wanted, cache_dir, *, label=None):
    """Stream a .tar.gz once, keep only the pinned members (each verified by size and SHA-256), and verify the whole
    archive's size and SHA-256 when the stream ends. Members are staged, and published under `cache_dir` (by basename)
    only after the whole-archive check has passed; a receipt of that check is written beside them. The cache is reused
    only when every member matches its pin AND the receipt records the same verified archive, so a failed archive
    check is never turned into apparent success on a retry. Member paths are never used as filesystem paths, so the
    archive cannot write outside `cache_dir`."""
    label = label or url
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {member: cache_dir / Path(member).name for member in wanted}
    pinned = {m: wanted[m][1] for m in sorted(wanted)}
    receipt_path = cache_dir / "archive_verification.json"
    receipt = None
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except ValueError:
            receipt = None
    members_ok = all(p.is_file() and p.stat().st_size == wanted[m][0] and sha256_file(p) == wanted[m][1] for m, p in out.items())
    if (
        members_ok and isinstance(receipt, dict) and receipt.get("whole_archive_verified") is True
        and receipt.get("archive_bytes") == archive_bytes and receipt.get("archive_sha256") == archive_sha256
        and receipt.get("members") == pinned
    ):
        ARCHIVE_STATUS[label] = {**receipt, "source": "cache: verification receipt matched and every member re-hashed"}
        return out
    receipt_path.unlink(missing_ok=True)  # this attempt supersedes any earlier status
    staging = cache_dir / ".staging"
    shutil.rmtree(staging, ignore_errors=True)  # the notebook's own staging directory inside its cache, nothing else
    staging.mkdir()
    ARCHIVE_STATUS[label] = {"url": url, "whole_archive_verified": False, "source": "stream started"}
    started = time.time()
    reader = _ResumableReader(url, archive_bytes)
    try:
        found = set()
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name not in wanted:
                    continue
                if not member.isfile():
                    raise ValueError(f"{name}: pinned member is not a regular file in the archive")
                data = archive.extractfile(member).read()
                size, digest = wanted[name]
                if len(data) != size or sha256_bytes(data) != digest:
                    raise ValueError(f"{name}: member {len(data)} bytes / {sha256_bytes(data)[:16]}… != pinned {size} / {digest[:16]}…")
                (staging / out[name].name).write_bytes(data)
                found.add(name)
        while reader.read(1 << 20):  # drain the remainder so the whole-archive digest can be checked
            pass
    finally:
        reader.close()
    if reader.count != archive_bytes or reader.sha.hexdigest() != archive_sha256:
        ARCHIVE_STATUS[label]["source"] = "whole-archive check FAILED; staged members were not published"
        raise ValueError(f"{url}: streamed {reader.count} bytes / {reader.sha.hexdigest()[:16]}…, pinned {archive_bytes} / {archive_sha256[:16]}…")
    missing = sorted(set(wanted) - found)
    if missing:
        ARCHIVE_STATUS[label]["source"] = "pinned members missing; staged members were not published"
        raise ValueError(f"archive is missing pinned members: {missing[:3]}…")
    for name in found:
        os.replace(staging / out[name].name, out[name])
    shutil.rmtree(staging, ignore_errors=True)
    receipt = {
        "url": url, "archive_bytes": archive_bytes, "archive_sha256": archive_sha256, "whole_archive_verified": True,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "members": pinned,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    ARCHIVE_STATUS[label] = {**receipt, "source": "streamed and verified in this run"}
    print(f"streamed {reader.count / 2**30:.2f} GiB in {time.time() - started:.0f} s ({reader.resumes} resumed connection(s)); kept {len(found)} verified members")
    return out

# ---- rasters and validation --------------------------------------------------------------------------------------
def load_tiff(path):
    """Pixel array as (bands, H, W) or (H, W); pixel-interleaved files are moved to band-first. No georeferencing."""
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        planar = tf.pages[0].planarconfig
    if arr.ndim == 3 and planar is not None and int(planar) == 1 and arr.shape[-1] <= 32:
        arr = np.moveaxis(arr, -1, 0)
    return np.asarray(arr)


NODATA = -9999
REFLECTANCE_CEILING = 1.5  # foundation / crop: a valid maximum at or below this is reflectance in [0, 1]
DN_FLOOR = 100.0           # ... at or above this, HLS digital numbers; a maximum in between is ambiguous and rejected


def to_digital_numbers(x, *, name, units="auto", log=True):
    """Return HLS digital numbers (reflectance × 10 000), keeping -9999 no-data pixels at -9999.

    The units are decided ONCE, from valid pixels only (not -9999): `units="reflectance"` or `"dn"` declares them;
    `"auto"` reads a valid maximum ≤ 1.5 as reflectance and ≥ 100 as digital numbers, and rejects anything in between
    as ambiguous. Data already returned by a validator is passed with `units="dn"`, so it is never scaled twice."""
    x = np.asarray(x, dtype=np.float32)
    valid = x != NODATA
    if not valid.any():
        raise ValueError(f"{name}: every pixel is no-data ({NODATA}); there is nothing to process")
    lo, hi = float(x[valid].min()), float(x[valid].max())
    if units == "auto":
        if hi <= REFLECTANCE_CEILING:
            source = "reflectance"
        elif hi >= DN_FLOOR:
            source = "dn"
        else:
            raise ValueError(
                f"{name}: valid values {lo:.4g}..{hi:.4g} are ambiguous between reflectance and HLS digital numbers; "
                "declare the units (BYOD_UNITS = 'reflectance' or 'dn')"
            )
        decision = "inferred from valid pixels"
    elif units in ("reflectance", "dn"):
        source, decision = units, "declared"
    else:
        raise ValueError(f"{name}: units must be 'auto', 'reflectance' or 'dn'; got {units!r}")
    if source == "reflectance":
        x = np.where(valid, x * 10_000.0, x).astype(np.float32)
    if log:
        if source == "reflectance":
            note(name, "reflectance converted to HLS digital numbers (× 10 000); no-data pixels kept")
        UNIT_DECISIONS[name] = {
            "source_units": source, "decision": decision, "valid_min": lo, "valid_max": hi,
            "nodata_pixels": int((~valid).sum()), "model_units": "HLS digital numbers",
        }
    return x


def _check_dn_range(x, *, name):
    valid = x != NODATA
    lo, hi = float(x[valid].min()), float(x[valid].max())
    if lo < -2_000 or hi > 20_000:
        raise ValueError(f"{name}: digital numbers {lo:.0f}..{hi:.0f} outside the plausible HLS range; check units and band order")


def validate_six_band_scene(array, *, name="scene", exact_size=None, multiple_of=None):
    x = np.asarray(array)
    if x.ndim != 3 or x.shape[0] != 6:
        raise ValueError(f"{name}: expected six spectral bands in HLS order (6, H, W); received shape {x.shape}")
    if exact_size is not None and x.shape[-2:] != (exact_size, exact_size):
        raise ValueError(f"{name}: expected {exact_size}×{exact_size} pixels, got {x.shape[-2]}×{x.shape[-1]}; the workshop does not resize inputs")
    if multiple_of is not None and (x.shape[-2] % multiple_of or x.shape[-1] % multiple_of):
        raise ValueError(f"{name}: height and width must be multiples of {multiple_of}, got {x.shape[-2:]}; the workshop does not crop or pad inputs")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: contains non-finite values")
    return np.ascontiguousarray(x.astype(np.float32))


def validate_foundation_scene(array, *, name="scene", units="auto", log=True):
    """Foundation input contract: six HLS-order bands, height and width multiples of 16, finite. Returns HLS digital
    numbers with -9999 no-data kept (the foundation normaliser maps it to the upstream fill value)."""
    x = validate_six_band_scene(array, name=name, multiple_of=16)
    x = to_digital_numbers(x, name=name, units=units, log=log)
    _check_dn_range(x, name=name)
    return np.ascontiguousarray(x)


def validate_segmentation_chip(raw, *, name, units="auto"):
    """Flood / burn-scar input contract: six HLS-order bands (or a 13-band Sentinel-2 L1C stack, reduced to them),
    exactly 512×512, finite; no-data (0, -9999) -> 0; reflectance × 10 000 -> reflectance ("auto" reads a maximum
    above 2 as × 10 000, as the pipelines' REFLECTANCE_MAX). Returns (6, 512, 512) reflectance. Idempotent."""
    x = np.asarray(raw)
    if x.ndim == 3 and x.shape[0] == 13:
        x = x[list(S2_L1C_BAND_INDICES)]
        note(name, f"13-band Sentinel-2 L1C stack reduced to bands {list(S2_L1C_BAND_INDICES)} (B2, B3, B4, B8A, B11, B12)")
    x = validate_six_band_scene(x, name=name, exact_size=512)
    nodata = (x == 0) | (x == NODATA)
    if nodata.all():
        raise ValueError(f"{name}: every pixel is no-data (0 or {NODATA}); there is nothing to map")
    if nodata.any():
        x = np.where(nodata, 0.0, x).astype(np.float32)
    if units == "auto":
        scale = False
        if float(x.max()) > 2.0:  # the plausible reflectance ceiling, as the pipelines' REFLECTANCE_MAX
            scale = True
        decision = "inferred from valid pixels"
    elif units in ("reflectance", "dn"):
        scale, decision = units == "dn", "declared"
    else:
        raise ValueError(f"{name}: units must be 'auto', 'reflectance' or 'dn'; got {units!r}")
    UNIT_DECISIONS[name] = {
        "source_units": "dn" if scale else "reflectance", "decision": decision,
        "nodata_pixels": int(nodata.sum()), "model_units": "reflectance",
    }
    if scale:
        x = x * 1e-4
        note(name, "values above 2 read as reflectance × 10 000 and scaled by 1e-4" if units == "auto" else "declared HLS digital numbers scaled by 1e-4")
    if float(x.min()) < -0.5 or float(x.max()) > 2.0:
        raise ValueError(f"{name}: reflectance {float(x.min()):.3f}..{float(x.max()):.3f} is outside the plausible range after scaling; check units and band order")
    return np.ascontiguousarray(x)


def validate_crop_scene(array, *, name="crop scene", units="auto", log=True):
    """Crop input contract: 18 date-major bands (3 dates × 6 HLS bands), 224×224, finite, no -9999 no-data. Returns
    HLS digital numbers. Units are decided once (`to_digital_numbers`); data already returned by this function is
    passed again with units="dn", so it is never rescaled."""
    x = np.asarray(array)
    if x.ndim != 3 or x.shape[0] != 18:
        raise ValueError(f"{name}: crop mapping requires exactly three acquisition dates × six bands = 18 bands; got shape {x.shape}")
    if x.shape[-2:] != (224, 224):
        raise ValueError(f"{name}: expected 224×224 pixels, got {x.shape[-2]}×{x.shape[-1]}; the workshop does not resize inputs")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: contains non-finite values")
    x = to_digital_numbers(x, name=name, units=units, log=log)
    n_nodata = int((x == NODATA).sum())
    if n_nodata:
        raise ValueError(
            f"{name}: {n_nodata} values are the {NODATA} no-data fill; the crop checkpoint has no documented no-data "
            "handling, so fill or crop those pixels out before inference"
        )
    _check_dn_range(x, name=name)
    return np.ascontiguousarray(x)


def require_integer_labels(mask, *, name):
    """Labels must be finite whole numbers; fractional values are rejected instead of truncated."""
    m = np.asarray(mask)
    if m.dtype.kind == "f":
        if not np.isfinite(m).all():
            raise ValueError(f"{name}: label contains non-finite values")
        fractional = m != np.round(m)
        if fractional.any():
            raise ValueError(f"{name}: label values must be whole numbers; found {np.unique(m[fractional])[:5].tolist()}")
    elif m.dtype.kind not in "iub":
        raise ValueError(f"{name}: unsupported label dtype {m.dtype}")
    return m.astype(np.int64)


def validate_label(mask, *, name, shape, num_classes):
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.shape != shape:
        raise ValueError(f"{name}: label dimensions {m.shape} do not match the image dimensions {shape}")
    m = require_integer_labels(m, name=name)
    values = set(np.unique(m).tolist())
    allowed = set(range(num_classes)) | {IGNORE_INDEX}
    if not values <= allowed:
        raise ValueError(f"{name}: label values {sorted(values - allowed)} outside {sorted(allowed)}")
    return np.ascontiguousarray(m)


# ---- model outputs -----------------------------------------------------------------------------------------------
class InvalidModelOutput(RuntimeError):
    """A model returned output that cannot be turned into a valid map, metric, embedding or export."""


def _host_array(values):
    return values.detach().float().cpu().numpy() if hasattr(values, "detach") else np.asarray(values, dtype=np.float32)


def check_class_scores(logits, *, name, batch, num_classes, spatial):
    """Output contract before softmax/argmax: finite class scores of shape (batch, num_classes, H, W)."""
    shape, expected = tuple(int(n) for n in logits.shape), (int(batch), int(num_classes), *(int(n) for n in spatial))
    if shape != expected:
        raise InvalidModelOutput(f"{name}: expected class scores of shape {expected}, got {shape}")
    bad = int((~np.isfinite(_host_array(logits))).sum())
    if bad:
        raise InvalidModelOutput(
            f"{name}: {bad} of {int(np.prod(shape))} class scores are NaN or infinite, so no map, metric or export is "
            "produced from them. Rerun the capability in a fresh runtime; if it persists, report the input and runtime."
        )


def check_probabilities(prob, *, name, atol=1e-3):
    p = _host_array(prob)
    if not np.isfinite(p).all() or float(np.abs(p.sum(axis=1) - 1.0).max()) > atol:
        raise InvalidModelOutput(f"{name}: class probabilities are non-finite or do not sum to 1")


def check_class_map(mask, *, name, num_classes):
    m = np.asarray(mask)
    if m.size and (int(m.min()) < 0 or int(m.max()) >= num_classes):
        raise InvalidModelOutput(f"{name}: class map values {int(m.min())}..{int(m.max())} outside 0..{num_classes - 1}")


def check_embedding(vector, *, name, dim):
    v = _host_array(vector)
    if v.shape != (dim,) or not np.isfinite(v).all() or not float(np.linalg.norm(v)) > 0.0:
        raise InvalidModelOutput(f"{name}: expected {dim} finite values with a non-zero norm; got shape {v.shape}")

# ---- visualization -----------------------------------------------------------------------------------------------
def rgb_limits(chws, rgb_indices=(2, 1, 0), p_low=2, p_high=98):
    """Per-channel display limits shared by several (bands, H, W) images, from valid pixels (finite, not -9999)."""
    limits = []
    for band in rgb_indices:
        values = np.concatenate([np.asarray(c, dtype=np.float32)[band].ravel() for c in chws])
        values = values[np.isfinite(values) & (values != NODATA)]
        lo, hi = (float(v) for v in np.percentile(values, [p_low, p_high])) if values.size else (0.0, 1.0)
        limits.append((lo, hi if hi > lo else lo + 1.0))
    return limits


def percentile_rgb(chw, rgb_indices=(2, 1, 0), p_low=2, p_high=98, limits=None):
    """Percentile-stretched display image; pass `limits` (from `rgb_limits`) to put several images on one stretch."""
    x = np.asarray(chw, dtype=np.float32)[list(rgb_indices)]
    out = np.zeros_like(x)
    for i in range(3):
        band = x[i]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        lo, hi = np.percentile(band[finite], [p_low, p_high]) if limits is None else limits[i]
        if hi <= lo:
            hi = lo + 1.0
        out[i] = np.clip((band - lo) / (hi - lo), 0, 1)
    return np.moveaxis(out, 0, -1)

def false_color(chw, indices=(3, 2, 1), p_low=2, p_high=98):
    return percentile_rgb(chw, indices, p_low, p_high)


def disagreement_map(pred, label):
    """0 = agree, 1 = false positive, 2 = false negative, 3 = unlabelled (binary maps)."""
    out = np.zeros(label.shape, dtype=np.uint8)
    out[(pred == 1) & (label == 0)] = 1
    out[(pred == 0) & (label == 1)] = 2
    out[label == IGNORE_INDEX] = 3
    return out


DISAGREE_CMAP = matplotlib.colors.ListedColormap(["#f2f2f2", "#d7301f", "#2b8cbe", "#969696"])


# ---- metrics -----------------------------------------------------------------------------------------------------
def confusion(preds, labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels, strict=True):
        p, l = np.asarray(p).reshape(-1).astype(np.int64), np.asarray(l).reshape(-1).astype(np.int64)
        keep = l != IGNORE_INDEX
        matrix += np.bincount(l[keep] * num_classes + p[keep], minlength=num_classes**2).reshape(num_classes, num_classes)
    return matrix


def binary_metrics(preds, labels, positive_name):
    """Pixel-pooled metrics over labelled pixels (ignore index excluded); class 1 is the positive class."""
    m = confusion(preds, labels, 2).astype(np.float64)
    total = m.sum()
    tp, fp, fn = m[1, 1], m[0, 1], m[1, 0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "positive_class": positive_name,
        "positive_iou": round(float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0, 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "pixel_accuracy": round(float(np.trace(m) / total), 4),
        "predicted_positive_fraction": round(float(m[:, 1].sum() / total), 4),
        "reference_positive_fraction": round(float(m[1].sum() / total), 4),
        "labelled_pixels": int(total),
        "confusion_rows_truth_cols_pred": m.astype(int).tolist(),
    }


def multiclass_metrics(preds, labels, class_names):
    m = confusion(preds, labels, len(class_names)).astype(np.float64)
    total = m.sum()
    tp, truth, pred = np.diag(m), m.sum(axis=1), m.sum(axis=0)
    union = truth + pred - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / np.where(union > 0, union, 1), np.nan)
        recall = np.where(truth > 0, tp / np.where(truth > 0, truth, 1), np.nan)
    rnd = lambda v: round(float(v), 4) if np.isfinite(v) else None
    return {
        "mean_iou": rnd(np.nanmean(iou)),
        "mean_class_accuracy": rnd(np.nanmean(recall)),
        "pixel_accuracy": rnd(np.trace(m) / total),
        "per_class_iou": {n: rnd(v) for n, v in zip(class_names, iou)},
        "per_class_recall": {n: rnd(v) for n, v in zip(class_names, recall)},
        "reference_class_fraction": {n: rnd(v / total) for n, v in zip(class_names, truth)},
        "classes_scored": int((union > 0).sum()),
        "labelled_pixels": int(total),
        "convention": "means over classes present in the labels or predictions of the scored set (nanmean)",
    }


# ---- models and memory -------------------------------------------------------------------------------------------
def audit_pickle_globals(path, allowed):
    """Static pickle GLOBAL/STACK_GLOBAL inventory of a plain pickle or a torch zip archive. Executes no pickle code."""
    data = Path(path).read_bytes()
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            streams = [zf.read(n) for n in zf.namelist() if n.endswith(".pkl")]
    else:
        streams = [data]
    found = {}
    for payload in streams:
        stack = []
        for op, arg, _ in pickletools.genops(io.BytesIO(payload)):
            if op.name == "GLOBAL":
                key = arg.replace("\n", " ").replace(" ", ".", 1)
                found[key] = found.get(key, 0) + 1
            elif op.name == "STACK_GLOBAL":
                key = f"{stack[-2]}.{stack[-1]}"
                found[key] = found.get(key, 0) + 1
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
                stack.append(arg)
            elif op.name not in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
                stack.append(None)
    violations = sorted(k for k in found if k not in allowed)
    if violations:
        raise ValueError(
            f"{Path(path).name}: pickle globals outside the allow-list: {violations}. "
            "Stop: do not extend the allow-list without reviewing the changed asset."
        )
    return {"globals": sorted(found), "violations": violations}


def fetch_checkpoint(capability):
    spec = MODEL_SPECS[capability]
    ckpt = download_hub_file(spec["model_id"], spec["checkpoint"], spec["revision"])
    verify_file(ckpt, spec["checkpoint_bytes"], spec["checkpoint_sha256"], f"{capability} checkpoint")
    audit = audit_pickle_globals(ckpt, spec["allowed_globals"])
    return ckpt, audit


def convert_and_reload(capability, model, build_fn):
    """Save `model`'s strictly loaded state dict as SafeTensors, verify it against the live DIMER profile's digest,
    and return a fresh model loaded strictly from that file (on DEVICE, eval mode, gradients off)."""
    from safetensors.torch import load_file, save_file

    spec = MODEL_SPECS[capability]
    canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
    if len(canonical) != spec["state_tensors"]:
        raise ValueError(f"{capability}: {len(canonical)} state tensors, expected {spec['state_tensors']}")
    path = CACHE_ROOT / capability / f"{capability}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(canonical, str(path), metadata={"format": "pt"})
    del canonical
    verify_file(path, spec["converted_bytes"], spec["converted_sha256"], f"{capability} converted SafeTensors")
    fresh = build_fn()
    fresh.load_state_dict(load_file(str(path)), strict=True)
    fresh.to(DEVICE).eval()
    for p in fresh.parameters():
        p.requires_grad_(False)
    return fresh, path


def free_accelerator():
    """Collect garbage and return cached GPU blocks. Call it AFTER deleting every reference to the model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return round(torch.cuda.memory_allocated() / 2**20, 1)
    return None


def save_json(path, payload, role="canonical"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return register_output(path, role)
''')

md("65e95022", r"""
## 5. Acquire and validate the workshop inputs

Everything the default path needs is downloaded here, before any model is loaded, and validated against its data contract. Model checkpoints are downloaded immediately before their capability runs.

- **Foundation scenes** — four dates of one HLS location from the Prithvi-EO-2.0 model repository.
- **Labelled evaluation chips** — 12 test chips with reference masks per task, pinned by the DIMER pipeline repositories. The flood chips come from the official Sen1Floods11 test split; the burn-scar chips from the test role of the burn-scar model repository's splits; the crop chips from spatially blocked test cells of the upstream **validation** archive (see the crop section for what that implies).
- **New scenes** — one unlabelled scene per task model from the upstream example files, chosen because it appears in none of the evaluation sets. It is used only for the new-scene inference step.
""")

code("732ca490", r'''
SCENE_PATHS, SAMPLE_MANIFEST = {}, []
for capability, source in SCENES.items():
    SCENE_PATHS[capability] = []
    for filename, size, digest in source["files"]:
        path = download_hub_file(source["repo_id"], filename, source["revision"], source["repo_type"])
        verify_file(path, size, digest, filename)
        SCENE_PATHS[capability].append(path)
        SAMPLE_MANIFEST.append({
            "capability": capability, "role": "foundation-demo" if capability == "foundation" else "new-scene",
            "id": Path(filename).stem, "source": f"{source['repo_id']}@{source['revision']}:{filename}",
            "license": source["license"], "bytes": size, "sha256": digest, "labelled": False,
        })

EVAL_PATHS = {}
flood_set = EVAL_SETS["flood"]
flood_cache = CACHE_ROOT / "eval" / "flood"
flood_cache.mkdir(parents=True, exist_ok=True)
EVAL_PATHS["flood"] = []
for chip, role, img_bytes, img_sha, lbl_bytes, lbl_sha in flood_set["records"]:
    pair = []
    for folder, size, digest in (("S2Hand", img_bytes, img_sha), ("LabelHand", lbl_bytes, lbl_sha)):
        rel = f"data/flood_events/HandLabeled/{folder}/{chip}_{folder}.tif"
        local = flood_cache / Path(rel).name
        if not (local.is_file() and local.stat().st_size == size and sha256_file(local) == digest):
            local.write_bytes(fetch_url(flood_set["base_url"] + rel))
        verify_file(local, size, digest, rel)
        pair.append(local)
    EVAL_PATHS["flood"].append((chip, *pair))

for capability in ("burnscar", "crop"):
    spec = EVAL_SETS[capability]
    wanted = {}
    for _key, _role, img_member, img_bytes, img_sha, mask_member, mask_bytes, mask_sha in spec["records"]:
        wanted[img_member] = (img_bytes, img_sha)
        wanted[mask_member] = (mask_bytes, mask_sha)
    kept = stream_pinned_members(spec["archive_url"], spec["archive_bytes"], spec["archive_sha256"], wanted, CACHE_ROOT / "eval" / capability, label=capability)
    EVAL_PATHS[capability] = [(r[0], kept[r[2]], kept[r[5]]) for r in spec["records"]]

for capability, spec in EVAL_SETS.items():
    for row in spec["records"]:
        if capability == "flood":
            chip, role, ib, isha, lb, lsha = row
            source = f"{spec['base_url']}data/flood_events/HandLabeled/{{S2Hand,LabelHand}}/{chip}_*.tif"
        else:
            chip, role, im, ib, isha, mm, lb, lsha = row
            source = f"{spec['archive_url']}#{im}"
        SAMPLE_MANIFEST.append({
            "capability": capability, "role": f"evaluation ({role})", "id": chip, "source": source,
            "license": spec["license"], "bytes": ib + lb, "image_sha256": isha, "label_sha256": lsha, "labelled": True,
            "label_semantics": spec["label_semantics"],
        })

# ---- validate everything before any model runs -------------------------------------------------------------------
foundation_stack = [validate_foundation_scene(load_tiff(p), name=p.name) for p in SCENE_PATHS["foundation"]]
if len({s.shape for s in foundation_stack}) != 1:
    raise ValueError("foundation scenes must share one shape to form a time series")

EVAL_DATA = {"flood": [], "burnscar": [], "crop": []}
for capability in ("flood", "burnscar"):
    for chip, img, lbl in EVAL_PATHS[capability]:
        image = validate_segmentation_chip(load_tiff(img), name=f"{capability}/{chip}")
        label = validate_label(load_tiff(lbl), name=f"{capability}/{chip} label", shape=(512, 512), num_classes=2)
        EVAL_DATA[capability].append({"id": chip, "image": image, "label": label})
for chip, img, lbl in EVAL_PATHS["crop"]:
    image = validate_crop_scene(load_tiff(img), name=f"crop/{chip}")
    raw = load_tiff(lbl)
    raw = require_integer_labels(raw[0] if raw.ndim == 3 else raw, name=f"crop/{chip} label")
    label = validate_label(np.where(raw == 0, IGNORE_INDEX, raw - 1), name=f"crop/{chip} label", shape=(224, 224), num_classes=13)
    EVAL_DATA["crop"].append({"id": chip, "image": image, "label": label})

NEW_SCENES = {
    "flood": {"id": SCENE_PATHS["flood"][0].stem, "image": validate_segmentation_chip(load_tiff(SCENE_PATHS["flood"][0]), name="flood new scene")},
    "burnscar": {"id": SCENE_PATHS["burnscar"][0].stem, "image": validate_segmentation_chip(load_tiff(SCENE_PATHS["burnscar"][0]), name="burn-scar new scene")},
    "crop": {"id": SCENE_PATHS["crop"][0].stem, "image": validate_crop_scene(load_tiff(SCENE_PATHS["crop"][0]), name="crop new scene")},
}
for capability, scene in NEW_SCENES.items():
    if scene["id"] in {r["id"] for r in EVAL_DATA[capability]}:
        raise ValueError(f"{capability}: the new scene {scene['id']} is also an evaluation chip")

save_json(OUTPUT_ROOT / "provenance" / "sample_manifest.json", {
    "band_order": list(BANDS),
    "scaling": {
        "foundation": "HLS digital numbers (reflectance × 10 000; units decided once from valid pixels, -9999 no-data kept), standardised with the checkpoint's config.json statistics",
        "flood/burnscar": "reflectance in [0, 1] (× 10 000 inputs scaled by 1e-4), standardised with the fine-tune's datamodule statistics",
        "crop": "HLS digital numbers (units decided once from valid pixels; -9999 no-data rejected), standardised with the checkpoint's training statistics",
    },
    "unit_decisions": UNIT_DECISIONS,
    "archive_verification": ARCHIVE_STATUS,
    "items": SAMPLE_MANIFEST,
})
print(pd.DataFrame([
    {"capability": m["capability"], "role": m["role"], "items": 1, "MiB": m["bytes"] / 2**20, "license": m["license"]}
    for m in SAMPLE_MANIFEST
]).groupby(["capability", "role", "license"], as_index=False).sum().round(2).to_string(index=False))
report_preprocessing()
print({k: len(v) for k, v in EVAL_DATA.items()}, "labelled evaluation chips validated;",
      "foundation stack shape:", (6, len(foundation_stack), *foundation_stack[0].shape[-2:]))
''')

md("6ebe33af", r"""
## 6. Understand the Earth-observation data

Unlike an ordinary photograph, the Prithvi models operate on **multispectral observations**. The six-band HLS-style contract used in these models is:

| Band | Meaning | Why it can matter |
|---|---|---|
| Blue | visible blue | water, haze, visible appearance |
| Green | visible green | vegetation/visible appearance |
| Red | visible red | chlorophyll absorption, visible appearance |
| Narrow NIR | near infrared | vegetation structure and vigor |
| SWIR 1 | short-wave infrared | moisture and burn response |
| SWIR 2 | short-wave infrared | moisture, char, soil/mineral response |

An RGB display is only a human-readable projection of the model input. The model sees all six bands.

For the crop model, the input contains **three dates × six bands = 18 raster bands**. Repeated observations add phenological information: crops that look similar on one date can behave differently across a growing season.
""")

code("492ccc64", r'''
import datetime
import re

flood_example = EVAL_DATA["flood"][0]
burn_example = EVAL_DATA["burnscar"][0]
crop_example = EVAL_DATA["crop"][0]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(percentile_rgb(foundation_stack[0]))
axes[0, 0].set_title("Foundation scene — natural colour")
axes[0, 1].imshow(false_color(foundation_stack[0]))
axes[0, 1].set_title("Foundation scene — NIR false colour")
axes[0, 2].imshow(percentile_rgb(crop_example["image"][:6]))
axes[0, 2].set_title(f"Crop chip {crop_example['id']} — date 1")
axes[1, 0].imshow(false_color(flood_example["image"], (4, 3, 2)))
axes[1, 0].set_title(f"Flood chip {flood_example['id']} — SWIR/NIR/red")
axes[1, 1].imshow(flood_example["label"], vmin=-1, vmax=1, cmap="Blues")
axes[1, 1].set_title("Flood reference mask (water = dark)")
axes[1, 2].imshow(false_color(burn_example["image"], (5, 3, 2)))
axes[1, 2].set_title(f"Burn chip {burn_example['id']} — SWIR2/NIR/red")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
save_figure(OUTPUT_ROOT / "figures" / "eo_inputs_overview.png", dpi=150, bbox_inches="tight")
plt.show()

band_stats = pd.DataFrame(
    [
        {"input": name, "band": band, "min": float(img[i].min()), "mean": float(img[i].mean()), "max": float(img[i].max())}
        for name, img in (("foundation date 1 (DN)", foundation_stack[0]), (f"flood {flood_example['id']} (reflectance)", flood_example["image"]))
        for i, band in enumerate(BANDS)
    ]
).round(4)
print(band_stats.to_string(index=False))


# The foundation time series: one location on four dates. HLS file names carry the acquisition as YYYYDDD
# (year and day of year), e.g. ...2018026T173609 -> 2018-01-26.
def acquisition_date(stem):
    match = re.search(r"\.(\d{4})(\d{3})T\d{6}", stem)
    if match is None:
        return stem
    return (datetime.date(int(match[1]), 1, 1) + datetime.timedelta(days=int(match[2]) - 1)).isoformat()


FOUNDATION_DATES = [acquisition_date(p.stem) for p in SCENE_PATHS["foundation"]]
dates_limits = rgb_limits(foundation_stack)
fig, axes = plt.subplots(1, len(foundation_stack), figsize=(4.5 * len(foundation_stack), 4.8))
for i, (ax, scene) in enumerate(zip(axes, foundation_stack)):
    ax.imshow(percentile_rgb(scene, limits=dates_limits))
    ax.set_title(f"Foundation date {i + 1}: {FOUNDATION_DATES[i]}")
    ax.axis("off")
fig.suptitle("Foundation time series: one location, natural colour, one shared stretch (brightness differences are real)")
plt.tight_layout()
save_figure(OUTPUT_ROOT / "figures" / "foundation_dates.png", dpi=130, bbox_inches="tight")
plt.show()
print(pd.DataFrame({"date": [f"date {i + 1}" for i in range(len(FOUNDATION_DATES))], "acquired": FOUNDATION_DATES}).to_string(index=False))
''')

md("2a046a3c", r"""
# 7. Capability A — Prithvi EO foundation representations

A foundation model does not inherently answer "is this flooded?" or "what crop is this?" Its encoder first produces a representation of the observation.

We will inspect two things:

1. **Embeddings** — 1024-dimensional representations from the final Prithvi encoder layer: the CLS token and the mean over all patch tokens.
2. **Masked reconstruction** — hide 75% of image patches and ask the pretrained model to reconstruct them.

The reconstruction score is compared with a simple **mean-fill baseline** that fills every hidden patch with the mean of the visible pixels of the same band and date. This is a check on the pretraining objective, not a measure of downstream environmental skill. An embedding has no intrinsic accuracy: judging whether it is useful for, say, flood mapping needs labelled chips and a downstream probe, which the Prithvi feature-extraction DIMER E2E tutorial demonstrates.

**Workshop prediction:** Look at the four dated foundation images at the end of §6. Before running the next cells, which two dates do you expect to have the most similar embeddings, and why?
""")

code("46cac0ce", r'''
# Foundation-model loader and inference helpers.
from terratorch.models.backbones.prithvi_mae import PrithviMAE

FOUNDATION_CONFIG = {
    "img_size": 224, "patch_size": (1, 16, 16), "num_frames": 4, "in_chans": 6, "embed_dim": 1024, "depth": 24,
    "num_heads": 16, "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16, "mlp_ratio": 4.0,
    "coords_encoding": [], "coords_scale_learn": False, "mask_ratio": 0.75, "norm_pix_loss": False,
}
FOUNDATION_MEANS = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
FOUNDATION_STDS = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)


def build_foundation_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PrithviMAE(**FOUNDATION_CONFIG)


def normalise_foundation(frames, *, name="frames"):
    """(6, T, H, W) HLS digital numbers from `validate_foundation_scene` -> standardised; -9999 no-data -> 0.0001 as the
    upstream inference script. Units were decided once, at validation: this function never rescales its input."""
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim != 4 or x.shape[0] != 6:
        raise ValueError(f"{name}: expected (6, T, H, W), got {x.shape}")
    if not 1 <= x.shape[1] <= 4:
        raise ValueError(f"{name}: the checkpoint was pretrained on up to 4 dates; got {x.shape[1]}")
    mean = np.asarray(FOUNDATION_MEANS, dtype=np.float32)[:, None, None, None]
    std = np.asarray(FOUNDATION_STDS, dtype=np.float32)[:, None, None, None]
    x = np.where(x == NODATA, 0.0001, (x - mean) / std)
    return np.ascontiguousarray(x.astype(np.float32))

def load_foundation_model():
    ckpt, audit = fetch_checkpoint("foundation")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("foundation checkpoint did not unpickle to a state dict of tensors")
    model = build_foundation_model()
    model.load_state_dict(state, strict=True)  # every key must match: no silently random weights
    del state
    fresh, converted = convert_and_reload("foundation", model, build_foundation_model)
    del model
    gc.collect()
    return fresh, audit, converted


def foundation_embedding(model, frames, *, name="frames"):
    batch = torch.from_numpy(normalise_foundation(frames, name=name)[None]).to(DEVICE)
    with torch.inference_mode():
        features = model.encoder.forward_features(batch)
    final = features[-1]
    dim = FOUNDATION_CONFIG["embed_dim"]
    if final.ndim != 3 or final.shape[0] != 1 or final.shape[1] < 2 or final.shape[-1] != dim:
        raise InvalidModelOutput(f"{name}: unexpected encoder output shape {tuple(final.shape)}")
    cls = final[:, 0].float().cpu().numpy()[0]
    mean_patch = final[:, 1:].float().mean(dim=1).cpu().numpy()[0]
    check_embedding(cls, name=f"{name} CLS embedding", dim=dim)
    check_embedding(mean_patch, name=f"{name} mean-patch embedding", dim=dim)
    return cls, mean_patch

def foundation_reconstruction(model, frames, mask_ratio=0.75, seed=42):
    """Seeded masked-patch reconstruction. TerraTorch returns the prediction and the (B, T, H, W) pixel mask
    already un-patchified (1 = masked). MSE over masked patches in standardised units vs a mean-fill baseline."""
    torch.manual_seed(seed)
    batch = torch.from_numpy(normalise_foundation(frames)[None]).to(DEVICE)
    with torch.inference_mode():
        loss, pred, mask = model(batch, None, None, mask_ratio)
    loss_value = float(loss["loss"] if isinstance(loss, Mapping) else loss)
    if mask.ndim != 4 or pred.shape != batch.shape:
        raise InvalidModelOutput(f"unexpected PrithviMAE output shapes: pred {tuple(pred.shape)}, mask {tuple(mask.shape)}")
    if not np.isfinite(loss_value) or not bool(torch.isfinite(pred).all()):
        raise InvalidModelOutput("masked reconstruction returned a non-finite loss or prediction; no result is recorded")
    mask_b = mask[:, None].expand(-1, batch.shape[1], -1, -1, -1).to(batch.dtype)  # (1, 6, T, H, W)
    visible = mask_b == 0
    baseline = batch.clone()
    for band in range(batch.shape[1]):
        for frame in range(batch.shape[2]):
            plane, keep = batch[0, band, frame], visible[0, band, frame]
            baseline[0, band, frame][~keep] = plane[keep].mean() if bool(keep.any()) else 0.0
    target = model.patchify(batch)
    patch_mask = model.patchify(mask_b).mean(-1)
    baseline_loss = (((model.patchify(baseline) - target) ** 2).mean(-1) * patch_mask).sum() / patch_mask.sum()
    merged = torch.where(mask_b.bool(), pred.to(batch.dtype), batch)
    return {
        "masked_mse": round(loss_value, 6),
        "baseline_mean_fill_mse": round(float(baseline_loss), 6),
        "masked_fraction": round(float(mask.float().mean()), 4),
        "masked_pixel_positions": int(mask.sum()),
        "mask_ratio": mask_ratio,
        "seed": seed,
        "units": "standardised (the pretraining loss)",
        "reconstruction_standardised": merged[0].float().cpu().numpy(),
        "mask": mask[0].float().cpu().numpy(),
    }


def reconstruction_record(result, scene_id):
    """The JSON-serialisable part of a reconstruction result (arrays dropped)."""
    return {"scene_id": scene_id, **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)}}


def reconstruction_panels(ax_masked, ax_rec, frames, result, *, date=0, limits=None):
    """One date as the model saw it (hidden patches grey) and with the hidden patches reconstructed, both on the
    observed image's display stretch, so observed and reconstructed pixels can be told apart."""
    mean = np.asarray(FOUNDATION_MEANS, dtype=np.float32)[:, None, None, None]
    std = np.asarray(FOUNDATION_STDS, dtype=np.float32)[:, None, None, None]
    observed = np.asarray(frames, dtype=np.float32)[:, date]
    limits = limits or rgb_limits([observed])
    shown = percentile_rgb(observed, limits=limits)
    shown[result["mask"][date] > 0] = 0.5
    ax_masked.imshow(shown)
    ax_masked.set_title(f"Date {date + 1} as the model saw it: {result['mask_ratio']:.0%} of patches hidden (grey)")
    reconstructed = (result["reconstruction_standardised"] * std + mean)[:, date]
    ax_rec.imshow(percentile_rgb(reconstructed, limits=limits))
    ax_rec.set_title(
        f"Visible patches + reconstruction of the {result['mask_ratio']:.0%} hidden\n"
        f"masked MSE={result['masked_mse']:.4f}; mean-fill={result['baseline_mean_fill_mse']:.4f}"
    )
''')

code("f369c51e", r'''
foundation_model, foundation_audit, foundation_converted = load_foundation_model()

# Four chronological images of the same location -> (6, T, H, W)
foundation_frames = np.stack(foundation_stack, axis=1)
date_ids = [p.stem for p in SCENE_PATHS["foundation"]]
date_labels = [f"date {i + 1} ({d})" for i, d in enumerate(FOUNDATION_DATES)]
FOUNDATION_SCENE_ID = "+".join(date_ids)
# The canonical ratio is the pretraining ratio. Do not edit it here: the controlled-change activity (§14) runs its own
# ratio in a separate cell, with separate state and output files.
CANONICAL_MASK_RATIO = FOUNDATION_CONFIG["mask_ratio"]

scene_cls, scene_mean = foundation_embedding(foundation_model, foundation_frames, name="foundation stack")
date_cls, date_mean = [], []
for i in range(foundation_frames.shape[1]):
    c, m = foundation_embedding(foundation_model, foundation_frames[:, i:i + 1], name=date_ids[i])
    date_cls.append(c)
    date_mean.append(m)
date_mean = np.stack(date_mean)
normed = date_mean / np.linalg.norm(date_mean, axis=1, keepdims=True)
similarity = normed @ normed.T

save_npz(
    OUTPUT_ROOT / "embeddings" / "prithvi_embeddings.npz",
    scene_id=np.array([FOUNDATION_SCENE_ID]),
    scene_cls=scene_cls[None],
    scene_mean_patch=scene_mean[None],
    date_ids=np.array(date_ids),
    date_acquired=np.array(FOUNDATION_DATES),
    date_cls=np.stack(date_cls),
    date_mean_patch=date_mean,
    date_cosine_similarity=similarity,
)

reconstruction_result = None
if RUN_RECONSTRUCTION:
    reconstruction_result = foundation_reconstruction(foundation_model, foundation_frames, mask_ratio=CANONICAL_MASK_RATIO, seed=SEED)
    save_json(OUTPUT_ROOT / "metrics" / "reconstruction.json", reconstruction_record(reconstruction_result, FOUNDATION_SCENE_ID))

del foundation_model
print("GPU MiB still allocated after release:", free_accelerator())

fig, axes = plt.subplots(1, 4, figsize=(23, 5.5))
im = axes[0].imshow(similarity, vmin=0, vmax=1)
axes[0].set_title("Cosine similarity of one-date embeddings")
axes[0].set_xticks(range(len(FOUNDATION_DATES)), FOUNDATION_DATES, rotation=45, ha="right")
axes[0].set_yticks(range(len(FOUNDATION_DATES)), FOUNDATION_DATES)
plt.colorbar(im, ax=axes[0], fraction=0.046)
date1_limits = rgb_limits([foundation_stack[0]])
axes[1].imshow(percentile_rgb(foundation_stack[0], limits=date1_limits))
axes[1].set_title(f"Date 1 observed ({FOUNDATION_DATES[0]})")
if reconstruction_result is not None:
    reconstruction_panels(axes[2], axes[3], foundation_frames, reconstruction_result, date=0, limits=date1_limits)
else:
    axes[2].text(0.5, 0.5, "Reconstruction disabled", ha="center", va="center")
for ax in axes[1:]:
    ax.axis("off")
plt.tight_layout()
save_figure(OUTPUT_ROOT / "figures" / "foundation_representation.png", dpi=150, bbox_inches="tight")
plt.show()

print(pd.DataFrame(similarity, index=date_labels, columns=date_labels).round(3).to_string())
print({
    "embedding_dim": int(scene_mean.shape[0]),
    "token_semantics": "final normalised encoder layer; CLS token and mean over all patch tokens",
    "reconstruction": None if reconstruction_result is None else {
        "mask_ratio": reconstruction_result["mask_ratio"],
        "masked_mse": reconstruction_result["masked_mse"],
        "mean_fill_mse": reconstruction_result["baseline_mean_fill_mse"],
    },
})
''')

keep("57172310")

md("8780246e", r"""
# 8. Capability B — flood mapping

**Input:** six-band Sentinel-2 chip, 512×512 (a 13-band L1C stack is reduced to the six Prithvi bands)
**Output:** one class per pixel: water versus no water, plus the model's water score
**Decision rule:** argmax over the two class scores (no threshold)

The live DIMER profile is a Prithvi-EO-2.0 model fine-tuned on Sen1Floods11. The notebook builds the same architecture (UPerNet decoder over four encoder depths) and loads the converted weights.

**Evaluation:** 12 hand-labelled chips from the official Sen1Floods11 **test** split, which the checkpoint did not train on. Metrics are pooled over all labelled pixels (cloud / no-data pixels excluded). The baseline predicts **no water everywhere**: it shows how much pixel accuracy a useless map can score when water is the minority class.

A predicted flood mask is **not a flood forecast**. It describes what the model detects in the supplied observation.
""")

code("e3d309fd", r'''
from terratorch.models import EncoderDecoderFactory

_NECKS = (
    {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
    {"name": "ReshapeTokensToImage"},
    {"name": "LearnedInterpolateToPyramidal"},
)
SEG_SPECS = {
    "flood": {
        "backbone": "prithvi_eo_v2_300_tl", "decoder": "UperNetDecoder",
        "decoder_args": {"decoder_channels": 256}, "head_args": {"head_dropout": 0.1},
        "means": (0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015),
        "stds": (0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938),
        "class_names": ("no water", "water"),
    },
    "burnscar": {
        "backbone": "prithvi_eo_v2_300", "decoder": "UNetDecoder",
        "decoder_args": {"decoder_channels": [512, 256, 128, 64]}, "head_args": {},
        "means": (0.033349706741586264, 0.05701185520536176, 0.05889748132001316, 0.2323245113436119, 0.1972854853760658, 0.11944914225186566),
        "stds": (0.02269135568823774, 0.026807560223070237, 0.04004109844362779, 0.07791732423672691, 0.08708738838140137, 0.07241979477437814),
        "class_names": ("not burned", "burn scar"),
    },
}


def build_segmentation_model(capability):
    spec = SEG_SPECS[capability]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EncoderDecoderFactory().build_model(
            task="segmentation", backbone=spec["backbone"], backbone_pretrained=False, backbone_bands=list(BANDS),
            decoder=spec["decoder"], num_classes=2, rescale=True, necks=[dict(n) for n in _NECKS],
            **spec["decoder_args"], **spec["head_args"],
        )


def load_segmentation_model(capability):
    """Audited checkpoint -> weights-only unpickle -> strict load -> SafeTensors (pinned digest) -> fresh strict load."""
    ckpt, audit = fetch_checkpoint(capability)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{capability}: checkpoint is not a Lightning checkpoint with a state_dict")
    state = {}
    for key, value in payload["state_dict"].items():
        if not key.startswith("model."):
            raise ValueError(f"{capability}: unexpected state-dict key {key!r}")
        state[key[len("model."):]] = value
    del payload
    model = build_segmentation_model(capability)
    model.load_state_dict(state, strict=True)
    del state
    fresh, converted = convert_and_reload(capability, model, lambda: build_segmentation_model(capability))
    del model
    gc.collect()
    return fresh, audit, converted


def segment(model, capability, images, batch_size=2):
    """(N, 6, 512, 512) reflectance -> masks (N, H, W) uint8 and positive-class scores (N, H, W) float32. The class
    scores are checked (shape, finite values, probabilities, label domain) before any mask is formed: an invalid
    output raises InvalidModelOutput instead of becoming an ordinary-looking map."""
    spec = SEG_SPECS[capability]
    n_classes = len(spec["class_names"])
    mean = np.asarray(spec["means"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(spec["stds"], dtype=np.float32)[None, :, None, None]
    masks, scores = [], []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(((np.asarray(images[start:start + batch_size]) - mean) / std).astype(np.float32)).to(DEVICE)
        with torch.inference_mode(), torch.autocast(device_type=DEVICE.split(":")[0], dtype=torch.float16, enabled=USE_AMP):
            out = model(batch)
        logits = (out.output if hasattr(out, "output") else out).float()
        name = f"{capability} model output (images {start}..{start + batch.shape[0] - 1})"
        if logits.ndim == 4 and tuple(logits.shape[-2:]) != tuple(batch.shape[-2:]):
            note(name, f"class scores resized {tuple(logits.shape[-2:])} -> {tuple(batch.shape[-2:])} (bilinear) before argmax")
            logits = torch.nn.functional.interpolate(logits, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        check_class_scores(logits, name=name, batch=batch.shape[0], num_classes=n_classes, spatial=batch.shape[-2:])
        prob = torch.softmax(logits, dim=1)
        check_probabilities(prob, name=name)
        mask = prob.argmax(dim=1).to(torch.uint8).cpu().numpy()
        check_class_map(mask, name=name, num_classes=n_classes)
        masks.append(mask)
        scores.append(prob[:, 1].cpu().numpy().astype(np.float32))
    return np.concatenate(masks), np.concatenate(scores)

def run_binary_capability(capability, title):
    """Load, evaluate on the labelled test chips against the all-negative baseline, infer the new scene, export."""
    positive = SEG_SPECS[capability]["class_names"][1]
    model, audit, converted = load_segmentation_model(capability)
    records = EVAL_DATA[capability]
    masks, scores = segment(model, capability, [r["image"] for r in records])
    new_mask, new_score = segment(model, capability, [NEW_SCENES[capability]["image"]])
    del model
    freed = free_accelerator()

    labels = [r["label"] for r in records]
    metrics = binary_metrics(masks, labels, positive)
    baseline = binary_metrics([np.zeros_like(l) for l in labels], labels, positive)
    per_chip = [
        {"id": r["id"], **{k: v for k, v in binary_metrics([m], [r["label"]], positive).items() if k != "confusion_rows_truth_cols_pred"}}
        for r, m in zip(records, masks)
    ]
    report = {
        "capability": capability,
        "model": {"id": MODEL_SPECS[capability]["model_id"], "revision": MODEL_SPECS[capability]["revision"]},
        "evaluation": {
            "dataset": EVAL_SETS[capability]["dataset"], "n_chips": len(records),
            "procedure": "held-out test chips; pixel-pooled over labelled pixels, ignore index excluded; no model selection in this notebook",
            "evidence": "tutorial sample metrics (12 chips), not a benchmark or operational estimate",
        },
        "decision_rule": "argmax over the two class scores (no threshold)",
        "scores_are_calibrated_probabilities": False,
        "precision": PRECISION[capability],
        "model_metrics": metrics,
        "baseline_all_negative": baseline,
        "per_chip": per_chip,
        "new_scene": {"id": NEW_SCENES[capability]["id"], "predicted_positive_fraction": round(float(new_mask[0].mean()), 4)},
        "pickle_audit": audit,
        "converted_weights": str(converted),
    }
    save_json(OUTPUT_ROOT / "metrics" / f"{capability}_metrics.json", report)
    pred_dir = OUTPUT_ROOT / "predictions" / capability
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_npz(pred_dir / "evaluation_predictions.npz", ids=np.array([r["id"] for r in records]),
                        mask=masks, positive_score=scores.astype(np.float16), class_names=np.array(SEG_SPECS[capability]["class_names"]))
    save_npz(pred_dir / "new_scene_prediction.npz", ids=np.array([NEW_SCENES[capability]["id"]]),
                        mask=new_mask, positive_score=new_score.astype(np.float16), class_names=np.array(SEG_SPECS[capability]["class_names"]))

    rgb_idx = (2, 1, 0) if capability == "flood" else (5, 3, 2)
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    for row, i in enumerate((0, 1)):
        r = records[i]
        axes[row, 0].imshow(percentile_rgb(r["image"], rgb_idx))
        axes[row, 0].set_title(f"{r['id']} — input")
        axes[row, 1].imshow(np.where(r["label"] < 0, np.nan, r["label"]), vmin=0, vmax=1)
        axes[row, 1].set_title("Reference")
        axes[row, 2].imshow(masks[i], vmin=0, vmax=1)
        axes[row, 2].set_title(f"Prediction (chip IoU {per_chip[i]['positive_iou']:.2f})")
        axes[row, 3].imshow(disagreement_map(masks[i], r["label"]), cmap=DISAGREE_CMAP, vmin=0, vmax=3)
        axes[row, 3].set_title("Disagreement: red = false +, blue = missed, grey = unlabelled")
    axes[2, 0].imshow(percentile_rgb(NEW_SCENES[capability]["image"], rgb_idx))
    axes[2, 0].set_title(f"New scene {NEW_SCENES[capability]['id'][:28]}")
    axes[2, 1].imshow(new_mask[0], vmin=0, vmax=1)
    axes[2, 1].set_title("New-scene prediction")
    im = axes[2, 2].imshow(new_score[0], vmin=0, vmax=1)
    axes[2, 2].set_title("New-scene positive score\n(not a calibrated probability)")
    plt.colorbar(im, ax=axes[2, 2], fraction=0.046)
    axes[2, 3].axis("off")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    save_figure(OUTPUT_ROOT / "figures" / f"{capability}_mapping.png", dpi=130, bbox_inches="tight")
    plt.show()

    table = pd.DataFrame(
        {"model": {k: metrics[k] for k in ("positive_iou", "f1", "precision", "recall", "pixel_accuracy", "predicted_positive_fraction")},
         f"baseline (all '{SEG_SPECS[capability]['class_names'][0]}')": {k: baseline[k] for k in ("positive_iou", "f1", "precision", "recall", "pixel_accuracy", "predicted_positive_fraction")}}
    )
    print(table)
    print({"reference_positive_fraction": metrics["reference_positive_fraction"], "labelled_pixels": metrics["labelled_pixels"],
           "GPU MiB still allocated after release": freed})
    return report
''')

code("61369c4a", r'''
flood_report = None
if RUN_FLOOD_MAPPING:
    flood_report = run_binary_capability("flood", "Flood / water mapping — Sen1Floods11 test chips and a new scene")
else:
    print("Flood mapping disabled.")
''')

md("e2e24050", r"""
### Interpretation checkpoint

Compare the model row with the baseline row. The baseline's pixel accuracy is simply the share of labelled pixels that are not water, yet its water IoU and F1 are zero. That gap is why this workshop reports IoU and F1 first.

A pixel map can look convincing even when it is wrong. Twelve chips from a public test split say little about skill in another region, season or sensor. Before operational use, a flood-mapping model must be evaluated against representative local reference data across different flood types, land covers, cloud conditions, seasons, and sensors.

The positive-class score shown above is a model score. It is **not** presented as a calibrated probability of flooding.
""")

md("923e87ea", r"""
# 9. Capability C — burn-scar mapping

**Input:** six-band HLS scene, 512×512
**Output:** burned versus not-burned class per pixel, plus the model's burn score
**Decision rule:** argmax over the two class scores (no threshold)

The live DIMER profile is the upstream Prithvi-EO-2.0 burn-scar fine-tune (UNet decoder), loaded here from the converted weights.

**Evaluation:** 12 labelled scenes from the test role of the burn-scar model repository's splits, streamed from the HLS Burn Scars archive. The baseline predicts **not burned everywhere**.

Burned-area mapping benefits from near-infrared and short-wave infrared information that is absent from ordinary RGB imagery; the inputs below are shown as SWIR 2 / NIR / red false colour.

**Exercise:** Before running the cell, predict: will the not-burned baseline have high or low pixel accuracy? What will its burn-scar IoU be?
""")

code("b887062e", r'''
burn_report = None
if RUN_BURNSCAR_MAPPING:
    burn_report = run_binary_capability("burnscar", "Burn-scar mapping — HLS Burn Scars test scenes and a new scene")
else:
    print("Burn-scar mapping disabled.")
''')

md("70e1d6c6", r"""
### What to notice

Overall accuracy can be dominated by the background class when the environmental phenomenon occupies only a small fraction of pixels. The baseline row above shows this directly: a map with no burn scars at all scores high pixel accuracy and zero burn-scar IoU. Positive-class IoU and F1, together with precision (how many predicted burn pixels are real) and recall (how many real burn pixels were found), describe the minority class that matters.

This workshop does not infer "fire severity" or "cause." It demonstrates the checkpoint's burned/not-burned mapping capability.
""")

md("a57ba84a", r"""
# 10. Capability D — multi-temporal crop / land-cover mapping

The crop model demonstrates why **time** matters in Earth observation.

Its input is one 224×224 site observed at **three dates**, with six HLS bands per date. The model predicts one of 13 crop/land-cover classes for every pixel (decision rule: argmax over the 13 class scores).

The legacy upstream checkpoint was trained with mmsegmentation. To keep the workshop compatible with the current hosted runtime and avoid the legacy mmcv stack, this notebook carries the small plain-PyTorch architecture used by the live DIMER profile, verifies the upstream checkpoint digest, performs a restricted one-time conversion to SafeTensors, and reloads the resulting inference network.

**Evaluation:** 12 labelled chips from spatially blocked test cells of the upstream **validation** archive, scored with mean IoU, mean class accuracy, per-class IoU and recall, against the **majority-class** constant map. The upstream authors may have used this validation split to select their checkpoint, so these numbers can be optimistic; they are tutorial evidence only.

The model's classes are specific to this checkpoint (a US cropland taxonomy); they are not a universal land-cover taxonomy.
""")

crop_src = "".join(orig_cells["4797da73"]["source"]) if isinstance(orig_cells["4797da73"]["source"], list) else orig_cells["4797da73"]["source"]
start, end = crop_src.index("CROP_ALLOWED_GLOBALS = TORCH_STATE_DICT_GLOBALS"), crop_src.index("def crop_normalise")
crop_src = crop_src[:start] + crop_src[end:]
start, end = crop_src.index("def load_crop_model():"), crop_src.index("def predict_crop")
crop_src = crop_src[:start] + CROP_LOADER + crop_src[end:]
crop_src = crop_src.replace("CROP_CONVERTED_SIZE = 537_722_508\n", "")
start, end = crop_src.index("def crop_normalise"), crop_src.index("def load_crop_model():")
crop_src = crop_src[:start] + CROP_NORMALISE + crop_src[end:]
crop_src = crop_src[:crop_src.index("def predict_crop")] + CROP_PREDICT
code("4797da73", crop_src)

code("cb180a5b", r'''
crop_report = None
if RUN_CROP_MAPPING:
    crop_model, crop_ckpt, crop_converted, crop_audit = load_crop_model()
    records = EVAL_DATA["crop"]
    predicted = [predict_crop(crop_model, r["image"]) for r in records]
    new_mask, new_scores, new_fractions = predict_crop(crop_model, NEW_SCENES["crop"]["image"])
    del crop_model
    freed = free_accelerator()

    masks = [p[0] for p in predicted]
    labels = [r["label"] for r in records]
    metrics = multiclass_metrics(masks, labels, CROP_CLASS_NAMES)
    counts = np.bincount(np.concatenate([l[l != IGNORE_INDEX] for l in labels]), minlength=13)
    majority = int(counts.argmax())
    baseline = multiclass_metrics([np.full_like(l, majority) for l in labels], labels, CROP_CLASS_NAMES)
    baseline["majority_class"] = CROP_CLASS_NAMES[majority]
    baseline["note"] = "the most frequent class of the scored labels: the best any constant map can do on these pixels"

    crop_report = {
        "capability": "crop",
        "model": {"id": MODEL_SPECS["crop"]["model_id"], "revision": MODEL_SPECS["crop"]["revision"]},
        "classes": list(CROP_CLASS_NAMES),
        "evaluation": {
            "dataset": EVAL_SETS["crop"]["dataset"], "n_chips": len(records),
            "procedure": "spatially blocked chips of the upstream validation archive; pixel-pooled, ignore index excluded",
            "caveat": "the upstream checkpoint may have been selected on this split; numbers can be optimistic",
            "evidence": "tutorial sample metrics (12 chips), not a benchmark or operational estimate",
        },
        "decision_rule": "argmax over 13 softmax scores",
        "scores_are_calibrated_probabilities": False,
        "precision": PRECISION["crop"],
        "model_metrics": metrics,
        "baseline_majority_class": baseline,
        "new_scene": {"id": NEW_SCENES["crop"]["id"], "class_fractions": {k: round(v, 4) for k, v in new_fractions.items()}},
        "pickle_audit": crop_audit,
        "converted_weights": str(crop_converted),
    }
    save_json(OUTPUT_ROOT / "metrics" / "crop_metrics.json", crop_report)
    pred_dir = OUTPUT_ROOT / "predictions" / "crop"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_npz(pred_dir / "evaluation_predictions.npz", ids=np.array([r["id"] for r in records]),
                        mask=np.stack(masks), scores=np.stack([p[1] for p in predicted]).astype(np.float16),
                        class_names=np.array(CROP_CLASS_NAMES))
    save_npz(pred_dir / "new_scene_prediction.npz", ids=np.array([NEW_SCENES["crop"]["id"]]),
                        mask=new_mask[None], scores=new_scores[None].astype(np.float16), class_names=np.array(CROP_CLASS_NAMES))

    cmap = matplotlib.colors.ListedColormap(plt.get_cmap("tab20").colors[:13])
    r0 = records[0]
    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    for date in range(3):
        axes[0, date].imshow(percentile_rgb(r0["image"][date * 6:(date + 1) * 6]))
        axes[0, date].set_title(f"{r0['id']} — date {date + 1}")
    axes[0, 3].imshow(np.ma.masked_less(r0["label"], 0), cmap=cmap, vmin=0, vmax=12, interpolation="nearest")
    axes[0, 3].set_title("Reference crop map")
    axes[1, 0].imshow(masks[0], cmap=cmap, vmin=0, vmax=12, interpolation="nearest")
    axes[1, 0].set_title("Model crop map")
    error = np.where(r0["label"] < 0, 2, (masks[0] != r0["label"]).astype(int))
    axes[1, 1].imshow(error, cmap=matplotlib.colors.ListedColormap(["#f2f2f2", "#d7301f", "#969696"]), vmin=0, vmax=2)
    axes[1, 1].set_title("Errors: red = wrong class, grey = unlabelled")
    axes[1, 2].imshow(new_mask, cmap=cmap, vmin=0, vmax=12, interpolation="nearest")
    axes[1, 2].set_title(f"New scene {NEW_SCENES['crop']['id']}")
    axes[1, 3].legend(
        handles=[matplotlib.patches.Patch(color=cmap(i), label=n) for i, n in enumerate(CROP_CLASS_NAMES)],
        loc="center", fontsize=9, title="Classes",
    )
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    save_figure(OUTPUT_ROOT / "figures" / "crop_mapping.png", dpi=130, bbox_inches="tight")
    plt.show()

    print(pd.DataFrame({
        "model": {k: metrics[k] for k in ("mean_iou", "mean_class_accuracy", "pixel_accuracy")},
        f"baseline (all '{CROP_CLASS_NAMES[majority]}')": {k: baseline[k] for k in ("mean_iou", "mean_class_accuracy", "pixel_accuracy")},
    }))
    print(pd.DataFrame({"IoU": metrics["per_class_iou"], "recall": metrics["per_class_recall"],
                        "reference fraction": metrics["reference_class_fraction"]}).to_string())
    print({"GPU MiB still allocated after release": freed})
else:
    print("Crop mapping disabled.")
''')

keep("f1ee360c")

md("613f3a7f", r"""
# 11. Cross-capability synthesis

These capabilities should **not** be placed on one leaderboard. They solve different problems with different outputs and different notions of error, and **the metrics are not comparable across tasks**. The comparison is between capabilities and problem formulations.

| Capability | Input | Output | Main information used | Primary tutorial measure | Baseline |
|---|---|---|---|---|---|
| EO foundation representation | 1–4 dates × six bands | embeddings + reconstruction | spectral, spatial, temporal | masked reconstruction MSE | mean fill |
| Flood mapping | six-band 512×512 chip | binary pixel map | spectral + spatial | water IoU / F1 | no water |
| Burn-scar mapping | six-band 512×512 HLS scene | binary pixel map | spectral + spatial | burn-scar IoU / F1 | not burned |
| Crop mapping | three dates × six bands | 13-class pixel map | spectral + spatial + temporal | mean IoU | majority class |
""")

code("18696399", r'''
summary_rows = []
if reconstruction_result is not None:
    summary_rows.append({
        "capability": "EO representation / reconstruction", "model": MODEL_SPECS["foundation"]["model_id"],
        "primary_measure": "masked MSE (standardised)", "model_value": reconstruction_result["masked_mse"],
        "baseline": "mean fill", "baseline_value": reconstruction_result["baseline_mean_fill_mse"],
    })
for report in (flood_report, burn_report):
    if report is not None:
        summary_rows.append({
            "capability": report["capability"], "model": report["model"]["id"], "primary_measure": f"{report['model_metrics']['positive_class']} IoU",
            "model_value": report["model_metrics"]["positive_iou"], "baseline": "all negative",
            "baseline_value": report["baseline_all_negative"]["positive_iou"],
        })
if crop_report is not None:
    summary_rows.append({
        "capability": "crop", "model": crop_report["model"]["id"], "primary_measure": "mean IoU",
        "model_value": crop_report["model_metrics"]["mean_iou"],
        "baseline": f"majority class ({crop_report['baseline_majority_class']['majority_class']})",
        "baseline_value": crop_report["baseline_majority_class"]["mean_iou"],
    })

summary_df = pd.DataFrame(summary_rows)
display(summary_df)
save_json(OUTPUT_ROOT / "workshop_summary.json", {"note": "tutorial sample measures; not comparable across capabilities", "rows": summary_rows})
''')

md("1408e000", r"""
# 12. Bring Your Own Data — optional

BYOD is deliberately outside the automatic sample path.

### Supported first-release input contracts

- **embedding** — one six-band HLS-style GeoTIFF (HLS digital numbers or reflectance in [0, 1]); height and width multiples of 16; the workshop treats it as one time step;
- **flood** — one 512×512 GeoTIFF with either the six Prithvi bands in HLS order or a 13-band Sentinel-2 L1C stack (bands `[1,2,3,8,11,12]` are selected);
- **burnscar** — one six-band, 512×512 HLS-style GeoTIFF (a 13-band Sentinel-2 L1C stack is also accepted and reduced the same way);
- **crop** — one 18-band, 224×224 GeoTIFF: three dates × six bands, date-major.

The workshop does not silently insert missing bands, reorder ambiguous inputs, or resize, crop or pad incompatible data. Unit conversions (reflectance ↔ digital numbers) are applied only in the documented direction and are reported. BYOD runs inference only; for evaluation or adaptation on your own labelled chips, use the corresponding model's DIMER E2E notebook.

### Units and no-data

`BYOD_UNITS = "auto"` decides once, from the valid pixels, whether the file holds reflectance or HLS digital numbers (reflectance × 10 000). For the embedding and crop inputs, `-9999` is treated as no-data and excluded from that decision; a file whose valid values are ambiguous (a maximum between 1.5 and 100) is rejected, and you declare the units with `BYOD_UNITS = "reflectance"` or `"dn"`. Flood and burn-scar inputs follow their pipelines' rule (values above 2 are read as reflectance × 10 000; 0 and `-9999` are no-data). The crop checkpoint has no documented no-data handling, so crop input containing `-9999` is rejected.

### Outputs and their limits

Each BYOD run writes, before any model is loaded, its own directory `outputs/byod/<capability>/<file>-<sha256 prefix>-<RUN_ID>/` containing the prediction (or embedding) and `receipt.json`. The receipt binds the input file's size and SHA-256, the capability, the effective unit decision and preprocessing, the model identity, and the SHA-256 of every output.

The outputs are **pixel arrays** (NPZ). CRS, affine transform, extent, resolution and no-data metadata are **not** carried over from a GeoTIFF input, so the outputs are not georeferenced GIS products. Keep the input file to recover the spatial reference, and validate alignment before overlaying a prediction on a map.

### Data movement and privacy

The standalone BYOD path processes your file **inside the selected notebook runtime**. It is not submitted to a DIMER worker or DIMER API. A hosted notebook is still a third-party compute environment: do not upload confidential, restricted, sensitive, personal, regulated, export-controlled, or commercially licensed imagery unless you are authorized to process it there.
""")

code("f23033f9", r'''
# @title Optional BYOD inference
import datetime

BYOD_RECEIPT = None
if not USE_BYOD:
    print("BYOD disabled — canonical Run all path complete.")
else:
    if BYOD_CAPABILITY not in {"embedding", "flood", "burnscar", "crop"}:
        raise ValueError(f"Unsupported BYOD_CAPABILITY: {BYOD_CAPABILITY}")
    if BYOD_PATH:
        byod_path = Path(BYOD_PATH)
    else:
        # Interactive fallback is reachable only when USE_BYOD=True and no path is set.
        try:
            from google.colab import files
        except ImportError as exc:
            raise ValueError("Set BYOD_PATH to a compatible local GeoTIFF.") from exc
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Upload exactly one compatible GeoTIFF.")
        name, payload = next(iter(uploaded.items()))
        byod_path = Path("byod") / Path(name).name
        byod_path.parent.mkdir(parents=True, exist_ok=True)
        byod_path.write_bytes(payload)

    if not byod_path.is_file():
        raise FileNotFoundError(f"BYOD file not found: {byod_path}")
    byod_id = byod_path.stem
    byod_sha256 = sha256_file(byod_path)
    raw = load_tiff(byod_path)
    _log_start = len(PREPROCESSING_LOG)
    input_name = f"BYOD {BYOD_CAPABILITY} scene {byod_id}"

    # 1. Validate against the capability's contract before any model is loaded.
    if BYOD_CAPABILITY == "embedding":
        scene = validate_foundation_scene(raw, name=input_name, units=BYOD_UNITS)
    elif BYOD_CAPABILITY in {"flood", "burnscar"}:
        scene = validate_segmentation_chip(raw, name=input_name, units=BYOD_UNITS)
    else:
        scene = validate_crop_scene(raw, name=input_name, units=BYOD_UNITS)

    # 2. Create and check the output destination before any model is loaded; it does not depend on the sample stages.
    byod_dir = OUTPUT_ROOT / "byod" / BYOD_CAPABILITY / f"{byod_id}-{byod_sha256[:12]}-{RUN_ID}"
    byod_dir.mkdir(parents=True, exist_ok=True)
    (byod_dir / ".write_check").write_bytes(b"")
    (byod_dir / ".write_check").unlink()

    # 3. Inference (the model is released even if inference fails).
    spec_key = "foundation" if BYOD_CAPABILITY == "embedding" else BYOD_CAPABILITY
    if BYOD_CAPABILITY == "embedding":
        model, _, _ = load_foundation_model()
        try:
            cls, emb = foundation_embedding(model, scene[:, None], name=input_name)
        finally:
            del model
            free_accelerator()
        out = save_npz(byod_dir / "embedding.npz", role="byod", ids=np.array([byod_id]), cls=cls[None], mean_patch=emb[None])
        summary = {"embedding_dim": len(emb)}
    elif BYOD_CAPABILITY in {"flood", "burnscar"}:
        model, _, _ = load_segmentation_model(BYOD_CAPABILITY)
        try:
            mask, score = segment(model, BYOD_CAPABILITY, [scene])
        finally:
            del model
            free_accelerator()
        out = save_npz(byod_dir / "prediction.npz", role="byod", ids=np.array([byod_id]), mask=mask,
                       positive_score=score.astype(np.float16), class_names=np.array(SEG_SPECS[BYOD_CAPABILITY]["class_names"]))
        summary = {"predicted_positive_fraction": round(float(mask.mean()), 4)}
    else:
        model, _, _, _ = load_crop_model()
        try:
            mask, scores, fractions = predict_crop(model, scene)
        finally:
            del model
            free_accelerator()
        out = save_npz(byod_dir / "prediction.npz", role="byod", ids=np.array([byod_id]), mask=mask[None],
                       scores=scores[None].astype(np.float16), class_names=np.array(CROP_CLASS_NAMES))
        summary = {"top_classes": sorted(fractions.items(), key=lambda kv: -kv[1])[:5]}

    # 4. A durable receipt binding the input bytes, the settings actually applied and the outputs.
    receipt = {
        "run_id": RUN_ID,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "capability": BYOD_CAPABILITY,
        "input": {"file_name": byod_path.name, "bytes": byod_path.stat().st_size, "sha256": byod_sha256,
                  "pixel_array_shape": list(raw.shape), "dtype": str(raw.dtype)},
        "units": {"requested": BYOD_UNITS, **UNIT_DECISIONS.get(input_name, {})},
        "preprocessing": PREPROCESSING_LOG[_log_start:],
        "model": {"id": MODEL_SPECS[spec_key]["model_id"], "revision": MODEL_SPECS[spec_key]["revision"],
                  "converted_safetensors_sha256": MODEL_SPECS[spec_key]["converted_sha256"], "precision": PRECISION[spec_key]},
        "outputs": {out.name: {"bytes": out.stat().st_size, "sha256": sha256_file(out)}},
        "georeferencing": "not preserved: pixel arrays without CRS, affine transform, extent, resolution or no-data metadata",
        "evidence": "inference only on user data without reference labels; no accuracy is implied",
    }
    receipt_path = save_json(byod_dir / "receipt.json", receipt, role="byod")
    BYOD_RECEIPT = {**receipt, "receipt_file": receipt_path.relative_to(OUTPUT_ROOT).as_posix()}
    print({"capability": BYOD_CAPABILITY, "id": byod_id, **summary, "output": str(out), "receipt": str(receipt_path)})
    report_preprocessing(_log_start)
''')

md("9b6e5ec2", r"""
# 13. Interpretation, limits, and climate-science context

## What the evidence establishes

This notebook demonstrates that the pinned upstream model assets can be acquired, verified, converted to the same SafeTensors weights DIMER serves, and executed locally in a standalone runtime; and it measures each task model on a small labelled sample against a naive baseline.

- The masked-reconstruction score is measurable because the notebook deliberately hides pixels whose truth is known.
- The flood, burn-scar and crop metrics are measured on 12 labelled chips each. They are **tutorial sample metrics**: no cross-validation, no confidence intervals, no region-stratified estimate.
- The flood chips come from the official test split and the burn-scar chips from the model repository's test role; the crop chips come from the upstream validation archive and may have influenced checkpoint selection.
- Public samples may overlap the data the foundation model was pretrained on; that cannot be ruled out here.

## What it does not establish

The workshop does not establish:

- model skill in the Philippines;
- robustness across tropical cloud regimes;
- cross-sensor equivalence;
- performance under different atmospheric-correction pipelines;
- operational flood, fire, agricultural, or land-use accuracy;
- calibrated per-pixel uncertainty;
- georeferenced map products: predictions are exported as pixel arrays (NPZ) without CRS, affine transform or extent, and must be re-aligned with their source scene before GIS use;
- causal attribution to climate change; or
- suitability for safety-critical decisions.

## Important mechanisms of domain shift

Earth-observation models can degrade when any of these change:

- sensor or spectral-response functions;
- band ordering;
- surface-reflectance processing;
- cloud, haze, smoke, snow, or shadow;
- spatial resolution;
- season and acquisition timing;
- geographic region and biome;
- crop calendar;
- terrain;
- label taxonomy or annotation procedure.

## Climate-science distinction

**Earth-observation AI** measures conditions or impacts visible in remotely sensed data.

**Earth-system models** such as Aurora forecast atmospheric/environmental states through time.

Both are useful to climate science, but they are different problem classes and require different data contracts and evaluation designs.

## Operational-use boundary

Do not treat these tutorial maps or metrics as sufficient evidence for emergency response, insurance, compensation, enforcement, environmental compliance, agricultural policy, or other consequential decisions. Such uses require locally representative reference data, appropriate uncertainty analysis, and human review.
""")

keep("651cc593")
keep("5b67565a")

code("33f19388", r'''
import datetime
import zipfile

model_manifest = {
    capability: {
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "source_checkpoint_sha256": spec["checkpoint_sha256"],
        "converted_safetensors_sha256": spec["converted_sha256"],
        "license": spec["license"],
        "precision": PRECISION[capability],
    }
    for capability, spec in MODEL_SPECS.items()
}

output_root = OUTPUT_ROOT.resolve()
manifest_rel = "provenance/experiment_manifest.json"
activity = globals().get("MASK_RATIO_ACTIVITY")  # set only by the optional controlled-change activity (§14)
byod = globals().get("BYOD_RECEIPT")


def produced_inventory():
    """The files this run wrote (registered in §4), with role, size and SHA-256; paths relative to OUTPUT_DIR."""
    rows = {}
    for path, role in sorted(PRODUCED_FILES.items()):
        p = Path(path)
        if p.is_file() and p.is_relative_to(output_root) and p.relative_to(output_root).as_posix() != manifest_rel:
            rows[p.relative_to(output_root).as_posix()] = {"role": role, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return rows


produced = produced_inventory()
not_from_this_run = sorted(
    p.relative_to(output_root).as_posix() for p in output_root.rglob("*")
    if p.is_file() and str(p.resolve()) not in PRODUCED_FILES and p.relative_to(output_root).as_posix() != manifest_rel
)

experiment_manifest = {
    "notebook_spec": "2.1",
    "notebook_profile": "MULTI-CAPABILITY",
    "notebook_mode": "WORKSHOP",
    "workshop_revision": "0.2.1-candidate",
    "run_id": RUN_ID,
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "random_seed": SEED,
    "runtime": {**RUNTIME, "device": DEVICE},
    "models": model_manifest,
    "datasets": {
        capability: {k: v for k, v in spec.items() if k != "records"} | {"record_ids": [r[0] for r in spec["records"]]}
        for capability, spec in EVAL_SETS.items()
    },
    "archive_verification": ARCHIVE_STATUS,
    "new_scenes": {k: v["id"] for k, v in NEW_SCENES.items()},
    "samples": "provenance/sample_manifest.json",
    "unit_decisions": UNIT_DECISIONS,
    "preprocessing_log": PREPROCESSING_LOG,
    "capabilities": {
        "foundation": {
            "executed": True, "reconstruction": reconstruction_result is not None, "baseline": "mean fill",
            "mask_ratio": None if reconstruction_result is None else reconstruction_result["mask_ratio"],
            "seed": None if reconstruction_result is None else reconstruction_result["seed"],
            "metrics": "metrics/reconstruction.json" if reconstruction_result is not None else None,
        },
        "flood": {"executed": flood_report is not None, "decision_rule": "argmax", "baseline": "all no water",
                  "metrics": "metrics/flood_metrics.json" if flood_report is not None else None},
        "burnscar": {"executed": burn_report is not None, "decision_rule": "argmax", "baseline": "all not burned",
                     "metrics": "metrics/burnscar_metrics.json" if burn_report is not None else None},
        "crop": {"executed": crop_report is not None, "decision_rule": "argmax", "baseline": "majority class",
                 "metrics": "metrics/crop_metrics.json" if crop_report is not None else None},
    },
    "activities": {} if activity is None else {"mask_ratio": activity},
    "byod": None if byod is None else {
        "capability": byod["capability"], "receipt": byod["receipt_file"], "input_sha256": byod["input"]["sha256"],
        "outputs": byod["outputs"],
    },
    "outputs": produced,
    "files_not_from_this_run": {
        "count": len(not_from_this_run), "paths": not_from_this_run[:100],
        "policy": "left in place, never deleted and not packaged; they belong to earlier runs",
    },
    "spatial_reference": "outputs are pixel arrays (NPZ); CRS, transform and extent are not preserved",
    "evidence_boundary": (
        "All metrics are tutorial sample metrics measured in this notebook on 12 labelled chips per task "
        "(masked reconstruction on one four-date scene). They are not benchmark or operational estimates. "
        "Optional activities and BYOD outputs are listed separately and are not canonical results."
    ),
    "standalone_contract": {
        "repository_clone_required": False,
        "dimer_source_runtime_fetch_required": False,
        "external_dimer_worker_required": False,
        "credential_required": False,
        "default_upload_required": False,
    },
}

manifest_path = save_json(OUTPUT_ROOT / manifest_rel, experiment_manifest)
bundle_path = Path(f"{output_root}_DIMER_EO_Workshop_Report_{RUN_ID}.zip")
with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for rel in [*produced, manifest_rel]:
        bundle.write(output_root / rel, arcname=rel)

print({
    "run_id": RUN_ID, "manifest": str(manifest_path), "report_bundle": str(bundle_path),
    "report_bundle_sha256": sha256_file(bundle_path), "files_packaged": len(produced) + 1,
    "optional_outputs": sorted({v["role"] for v in produced.values()} - {"canonical"}),
    "files_not_from_this_run": len(not_from_this_run),
})
''')

md("982c1134", r"""
# 16. Troubleshooting

| Symptom | Likely cause | Corrective action |
|---|---|---|
| `The isolated environment's Python process exited` | the worker process in `dimer_eo_env/` crashed, usually out of host memory | restart the session and choose **Run all** again; the variables of the crashed process are gone |
| CUDA is unavailable | CPU runtime selected | Change to a T4 GPU runtime before running from the top |
| Size / SHA-256 mismatch on a checkpoint, scene or chip | interrupted or changed asset | delete the cached file (`workshop_cache/` or the Hugging Face cache) and rerun; do not bypass the integrity check |
| Archive stream reports "retries exhausted" | the network dropped more than 20 times during one archive | rerun the acquisition cell on a steadier connection; the archive stream then starts from the beginning |
| Archive digest mismatch after streaming | a changed upstream archive | stop: the pinned archive changed upstream; do not update the pin without reviewing the new data |
| Converted SafeTensors digest mismatch | TerraTorch/PyTorch version drift changed the rebuilt architecture | confirm the pinned environment and rerun from a fresh runtime |
| Pickle audit refuses a checkpoint | serialized globals differ from the reviewed asset | stop; do not add globals to the allow-list without reviewing the changed asset |
| Out of GPU memory | runtime has less VRAM, or an earlier model is still referenced | restart fresh and use the canonical top-to-bottom path; each stage deletes its model and prints the GPU memory still allocated |
| BYOD band-count or size error | wrong sensor/product, band layout or chip size | provide the exact documented capability input; the workshop does not resize |
| Crop input rejected | not 18 bands or not 224×224 | provide three dates × six bands at the checkpoint's expected size |
| Units reported as ambiguous | valid values fall between reflectance (≤ 1.5) and HLS digital numbers (≥ 100) | check the product's scaling and set `BYOD_UNITS` to `"reflectance"` or `"dn"` |
| Crop input rejected for `-9999` no-data | the crop checkpoint has no documented no-data handling | fill or crop the no-data pixels before inference |
| `InvalidModelOutput` (NaN/Inf or wrong-shaped class scores) | a numerical failure or an unexpected model output; no map, metric or export was produced from it | rerun the capability in a fresh runtime; if it persists, record the input, runtime and precision and report it |
| Archive re-streamed although its members are cached | no whole-archive verification receipt for the cached members (for example after an earlier digest failure) | let the stream finish; the cache is reused only after the whole archive has verified |
| Results look plausible but unexpected | domain shift or preprocessing mismatch | inspect sensor, units, band order, season, resolution, and local reference data before interpreting the map |

An explicit error is preferable to silently changing a user's Earth-observation data contract.
""")

keep("5f327d9b")

code("7a2c0b79", r'''
# @title Run-all completion summary
completion = {
    "notebook_spec": "2.1",
    "profile": "MULTI-CAPABILITY",
    "mode": "WORKSHOP",
    "device": DEVICE,
    "foundation_reconstruction": None if reconstruction_result is None else reconstruction_result["masked_mse"],
    "flood_water_iou": None if flood_report is None else flood_report["model_metrics"]["positive_iou"],
    "burnscar_iou": None if burn_report is None else burn_report["model_metrics"]["positive_iou"],
    "crop_mean_iou": None if crop_report is None else crop_report["model_metrics"]["mean_iou"],
    "byod_enabled": bool(USE_BYOD),
    "byod_receipt": None if BYOD_RECEIPT is None else BYOD_RECEIPT["receipt_file"],
    "run_id": RUN_ID,
    "output_directory": str(OUTPUT_ROOT.resolve()),
    "report_bundle": str(bundle_path),
}
display(pd.Series(completion, name="value").to_frame())
print(
    "Canonical workshop path complete. Metrics are tutorial sample measures on 12 labelled chips per task; "
    "use labelled, representative local data before making performance claims."
)
''')


# Guided self-paced learning layer ---------------------------------------------------------------
_GUIDED_INTRO = r"""
## How to use this notebook

**Who it is for.** Learners who can run Python cells in Colab/Jupyter and are new to Earth-observation machine learning. Basic familiarity with arrays and plots is sufficient; remote-sensing expertise is not required.

**Runtime.** Use a fresh NVIDIA Tesla T4-class GPU runtime or the documented equivalent. The pinned stack is installed into an isolated environment, so a single **Run all** completes without a runtime restart.

**How to run it.**
1. Select the documented GPU runtime.
2. Choose **Run all** and keep the canonical defaults on your first pass.
3. Read the explanatory markdown while the notebook executes.
4. Sections marked **Infrastructure** handle installation, asset verification, model loading, or provenance. Run those cells as written; their implementation is not a learning objective.

### Task at a glance

`multispectral / multi-temporal Earth-observation data → foundation representation or task model → embedding / pixel map → evaluation and interpretation`

### Roadmap

1. Understand the spectral, spatial, and temporal input contracts.
2. Inspect the built-in scenes and labelled evaluation chips.
3. Establish naive task baselines.
4. Run the foundation representation and the flood, burn-scar, and crop/land-cover capabilities.
5. Compare maps with labels and inspect domain-shift mechanisms.
6. Try one controlled reconstruction change.
7. Write an evidence-based conclusion; optionally continue with BYOD.

### What successful execution looks like

You should finish with verified model/data identities, a foundation-model representation/reconstruction result, task metrics against naive baselines for the labelled samples, unseen-scene outputs, and exported predictions/provenance. These are tutorial sample measures, not operational or Philippines-specific validation results.
"""

_GUIDED_TRY = r"""
## Try it yourself — one controlled change

Use the masked-reconstruction capability because it gives you known truth for the deliberately hidden pixels.

**Predict → change one variable → run → observe → explain**

1. **Predict.** The canonical run hid 75% of the patches (`mask_ratio = 0.75`). If only 50% are hidden, should the masked-pixel reconstruction error go up or down? What about the mean-fill baseline?
2. **Change one variable.** In the cell below, set `RUN_MASK_RATIO_ACTIVITY = True` and keep `ACTIVITY_MASK_RATIO = 0.50`. Run **only that cell**. It loads its own copy of the foundation model (the §7 cell has already released its copy), uses the canonical seed, and writes to `outputs/activities/mask_ratio_0.50/`. It does not change the canonical result, `metrics/reconstruction.json`, or the §7 figure, and it checks this before finishing.
3. **Observe.** Compare the two rows of the table and the two pairs of images: hidden patches are grey on the left, reconstructed on the right.
4. **Explain** whether the direction of change matches your prediction.

**Read the comparison carefully.** The MSE is averaged over the hidden pixels only, so the two ratios score different pixel populations, and at 0.50 the model sees more context around each hidden patch. Compare each model score with **its own** mean-fill baseline (the `model / mean-fill` column), not only the two raw MSE values. This exercise probes the pretraining objective; it is not an estimate of downstream mapping skill.

To add the activity to the report bundle, rerun the §15 export cell afterwards. The bundle and manifest list it under `activities`, separately from the canonical results.
"""

_GUIDED_ACTIVITY_CODE = r'''
# @title Controlled change: reconstruction mask ratio (optional; canonical results are not modified)
RUN_MASK_RATIO_ACTIVITY = False  # @param {type:"boolean"}
ACTIVITY_MASK_RATIO = 0.50       # @param {type:"number"}
ACTIVITY_SEED = SEED             # the canonical seed, so that only the mask ratio changes

if not RUN_MASK_RATIO_ACTIVITY:
    print("Activity not run. Set RUN_MASK_RATIO_ACTIVITY = True and run this cell only.")
else:
    ratio = float(ACTIVITY_MASK_RATIO)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ACTIVITY_MASK_RATIO must be strictly between 0 and 1; got {ACTIVITY_MASK_RATIO}")
    canonical_paths = [OUTPUT_ROOT / "metrics" / "reconstruction.json", OUTPUT_ROOT / "figures" / "foundation_representation.png"]
    canonical_digests = {str(p): sha256_file(p) for p in canonical_paths if p.is_file()}
    canonical_values = None if reconstruction_result is None else reconstruction_record(reconstruction_result, FOUNDATION_SCENE_ID)

    activity_dir = OUTPUT_ROOT / "activities" / f"mask_ratio_{ratio:.2f}"
    activity_model, _, _ = load_foundation_model()  # the §7 cell released its model; this copy is released below
    try:
        activity_result = foundation_reconstruction(activity_model, foundation_frames, mask_ratio=ratio, seed=ACTIVITY_SEED)
    finally:
        del activity_model
        free_accelerator()

    activity_json = save_json(activity_dir / "reconstruction.json", {
        "activity": "controlled change: masked-reconstruction mask ratio",
        "run_id": RUN_ID,
        "changed_variable": "mask_ratio",
        **reconstruction_record(activity_result, FOUNDATION_SCENE_ID),
        "canonical": canonical_values,
        "interpretation_note": (
            "MSE is averaged over hidden pixels only, so different mask ratios score different pixel populations; "
            "compare each score with its own mean-fill baseline. Not an estimate of downstream mapping skill."
        ),
    }, role="activity")

    rows = [(label, r) for label, r in (("canonical", reconstruction_result), ("activity", activity_result)) if r is not None]
    date1_limits = rgb_limits([foundation_stack[0]])
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 6.3 * len(rows)), squeeze=False)
    for (label, result), (ax_masked, ax_rec) in zip(rows, axes):
        reconstruction_panels(ax_masked, ax_rec, foundation_frames, result, date=0, limits=date1_limits)
        ax_masked.set_title(f"{label}: " + ax_masked.get_title())
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    save_figure(activity_dir / "reconstruction_comparison.png", role="activity", dpi=130, bbox_inches="tight")
    plt.show()

    display(pd.DataFrame([
        {
            "run": label, "mask_ratio": r["mask_ratio"], "seed": r["seed"], "masked_fraction": r["masked_fraction"],
            "hidden_pixel_positions": r["masked_pixel_positions"], "masked_mse": r["masked_mse"],
            "mean_fill_mse": r["baseline_mean_fill_mse"], "model / mean-fill": round(r["masked_mse"] / r["baseline_mean_fill_mse"], 4),
        }
        for label, r in rows
    ]).set_index("run"))

    unchanged = {str(p): sha256_file(p) for p in canonical_paths if p.is_file()} == canonical_digests and (
        canonical_values is None or reconstruction_record(reconstruction_result, FOUNDATION_SCENE_ID) == canonical_values
    )
    if not unchanged:
        raise RuntimeError("the activity changed a canonical reconstruction output; restart and run the notebook from the top")
    MASK_RATIO_ACTIVITY = {
        "mask_ratio": activity_result["mask_ratio"], "seed": activity_result["seed"],
        "masked_mse": activity_result["masked_mse"], "baseline_mean_fill_mse": activity_result["baseline_mean_fill_mse"],
        "record": activity_json.relative_to(OUTPUT_ROOT).as_posix(), "canonical_outputs_unchanged": True,
    }
    print({"activity": MASK_RATIO_ACTIVITY["record"], "canonical_outputs_unchanged": True,
           "next": "rerun the §15 export cell to add this activity to the report bundle"})
'''

_GUIDED_CHECKPOINT = r"""
## Self-paced checkpoint

Before opening the sample interpretation, answer:

1. How is an EO foundation representation different from a flood/burn/crop task output?
2. Why are IoU/F1-style positive-class measures more informative than pixel accuracy when the positive class is sparse?
3. Which sensor, place, season, resolution, or preprocessing changes could cause domain shift?
4. What additional local evidence would you require before using these maps in the Philippines?

<details>
<summary><b>Show a sample interpretation</b></summary>

The foundation model produces a reusable representation (and can be probed through masked reconstruction), while the task-specialized models produce semantic pixel maps. A high overall pixel accuracy can be misleading when most pixels are background, so class-aware overlap/precision/recall measures and naive baselines matter. None of the small labelled samples establishes Philippines performance; local sensor/preprocessing compatibility, geographically and seasonally representative labels, uncertainty analysis, and human review would be required.

</details>
"""

_GUIDED_CONCLUSION = r"""
## Write an evidence-based conclusion

Use the outputs from your run.

1. **State the question.** Which EO capability did you evaluate?
2. **Report the primary result against its baseline.** Use masked reconstruction MSE for the reconstruction probe, positive-class IoU/F1 for binary maps, or mean IoU for crop/land-cover mapping as appropriate.
3. **Add supporting evidence.** Mention a qualitative map pattern, precision/recall tradeoff, class behavior, or unseen-scene observation.
4. **Name the domain-shift risks.** Sensor, band order, surface-reflectance processing, geography, season, spatial resolution, and label taxonomy can all matter.
5. **State the limits.** These small samples demonstrate executable capability and tutorial-scale evaluation; they do not establish operational accuracy, climate attribution, or Philippines-specific performance.
"""

def _guided_text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src

def _set_guided_text(cell, text):
    cell["source"] = text

def apply_guided_layer(items):
    out = [dict(c) for c in items]
    for cell in out:
        text = _guided_text(cell)
        if cell.get("cell_type") == "markdown":
            if text.startswith("# DIMER "):
                lines = text.splitlines()
                lines[0] = "# DIMER Notebook: AI for Earth Observation and Climate Applications"
                text = "\n".join(lines)
            text = (
                text.replace("This standalone workshop", "This standalone notebook")
                    .replace("This workshop", "This notebook")
                    .replace("this workshop", "this notebook")
                    .replace("The workshop", "The notebook")
                    .replace("the workshop", "the notebook")
                    .replace("## 2. Workshop controls", "## 2. Notebook controls")
                    .replace("Candidate workshop carrier", "Candidate notebook carrier")
                    .replace("applications** workshop.", "applications** notebook.")
                    .replace("belongs in a separate workshop.", "belongs in a separate notebook.")
                    .replace("**Workshop prediction:**", "**Prediction:**")
                    .replace("# 14. Workshop exercises and transfer prompts", "# 14. Exercises and transfer prompts")
                    .replace("The workshop does not", "The notebook does not")
                    .replace("the workshop treats", "the notebook treats")
            )
            first = text.splitlines()[0] if text.splitlines() else ""
            if any(k in first.lower() for k in ("runtime", "model and sample identities", "model identities", "provenance", "install")) and "**Infrastructure.**" not in text:
                text += "\n\n> **Infrastructure.** This section supports reproducibility and execution. Run the associated setup code as written; understanding its implementation is not a learning objective for this notebook."
            _set_guided_text(cell, text)
        elif cell.get("cell_type") == "code":
            text = (
                text.replace("# @title Install the tested workshop runtime", "# @title Install the tested notebook runtime")
                    .replace("# @title Workshop controls", "# @title Notebook controls")
                    .replace("Canonical workshop path complete.", "Canonical notebook path complete.")
                    .replace("supported workshop runtime", "supported notebook runtime")
            )
            _set_guided_text(cell, text)

    if not any("## How to use this notebook" in _guided_text(c) for c in out):
        out.insert(1, {"cell_type":"markdown","id":"guided-how-to-use","metadata":{},"source":_GUIDED_INTRO.strip("\n")})

    insert_at = next(
        (i for i,c in enumerate(out) if c.get("cell_type")=="markdown" and _guided_text(c).lstrip().startswith("# 16. Troubleshooting")),
        len(out)
    )
    joined = "\n".join(_guided_text(c) for c in out if c.get("cell_type")=="markdown")
    guided = []
    if "## Try it yourself — one controlled change" not in joined:
        guided.append({"cell_type":"markdown","id":"guided-controlled-change","metadata":{},"source":_GUIDED_TRY.strip("\n")})
        guided.append({"cell_type":"code","id":"guided-controlled-change-run","metadata":{},"execution_count":None,"outputs":[],"source":_GUIDED_ACTIVITY_CODE.strip("\n")})
    if "## Self-paced checkpoint" not in joined:
        guided.append({"cell_type":"markdown","id":"guided-self-check","metadata":{},"source":_GUIDED_CHECKPOINT.strip("\n")})
    if "## Write an evidence-based conclusion" not in joined:
        guided.append({"cell_type":"markdown","id":"guided-conclusion","metadata":{},"source":_GUIDED_CONCLUSION.strip("\n")})
    out[insert_at:insert_at] = guided
    return out


# ------------------------------------------------------------------------------------------------------------------
cells = apply_guided_layer(cells)
nb = dict(orig)
nb["cells"] = cells
nb["metadata"] = dict(orig["metadata"])
nb["metadata"]["workshop_revision"] = "0.2.1-candidate"
nb["metadata"]["dimer"] = dict(orig["metadata"]["dimer"]) | {"clean_runtime_evidence": "pending"}
content = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
if _args.check:
    if not OUT.exists() or OUT.read_text(encoding="utf-8") != content:
        raise SystemExit(f"STALE: {OUT}; regenerate with python tools/eo_workshop/build_workshop.py")
    print("OK:", OUT)
else:
    OUT.write_text(content, encoding="utf-8", newline="\n")
    print("wrote", OUT, len(cells), "cells")
