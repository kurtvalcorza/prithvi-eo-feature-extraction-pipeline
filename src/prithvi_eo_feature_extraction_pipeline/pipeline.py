"""Prithvi-EO-2.0-300M (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`) DIMER pipeline: verified snapshot, one-time
conversion of the pickled MAE checkpoint into safetensors, token and scene embeddings of HLS reflectance stacks,
masked-patch reconstruction scored against a mean-fill baseline, and a bounded linear probe on the frozen token
embeddings for a user's labelled chips, with a portable adapter.

Prithvi-EO-2.0 (Szwarcman et al., 2024) is a ViT-L masked autoencoder for Harmonized Landsat Sentinel-2 imagery:
3-D patch embeddings over (time, height, width) cubes of 1 × 16 × 16, 24 encoder blocks of width 1024, and an
8-block decoder of width 512 that reconstructs masked patches. The checkpoint packaged here is the 300 M-parameter
variant without temporal/location embeddings, pretrained on 4.2 M HLS samples with a 75 % mask ratio, published
by IBM and NASA as `Prithvi_EO_V2_300M.pt` — a state dict of the encoder and the decoder (398 tensors).

The upstream asset is a torch zip archive whose pickle references only `collections.OrderedDict`,
`torch._utils._rebuild_tensor_v2` and `torch.FloatStorage` (verified statically by `audit_pickle`). Under the fleet
asset specification (§11) that is executable serialization, so this package converts it once — `torch.load(
weights_only=True)`, a strict load into the architecture built from the `terratorch` package on PyPI — into
safetensors with a pinned digest, and serves only the converted file. Nothing is fetched from the Hub at load time
except the manifest-listed files, and the `prithvi_mae.py` module the Hub repository ships is never imported.

Everything model-related is imported lazily so that snapshot verification, the pickle audit and input validation
run (and can refuse) before `torch` or `terratorch` are imported (fleet RTM-001). `numpy` and `tifffile` are used
for chips and are imported freely.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickletools
import time
import warnings
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
MODEL_REVISION = "9eb1b1102806593963daa333bcc491b1c6f8562f"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "prithvi-eo-2.0-300m"
ARTIFACT_FORMAT = "org.valcorza.prithvi-eo-feature-extraction.probe.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Immutable upstream source asset (a torch zip archive holding a state dict, i.e. a pickle; see docs/WEIGHTS.md).
SOURCE_CKPT_NAME = "Prithvi_EO_V2_300M.pt"
SOURCE_CKPT_BYTES = 1_326_658_396
SOURCE_CKPT_SHA256 = "faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7"
# Code-free serving file produced deterministically by `convert_model` (asset spec §11.2).
CONVERTED_WEIGHTS_NAME = "prithvi-eo-2.0-300m.safetensors"
CONVERTED_SHA256 = "15b8ed6dacf8b3ca02c542bb8eaffc92c7a7addc5351b3d967873ad3f73a24b0"
CONVERTED_BYTES = 1_326_543_456
# Static-audit digest of the source pickle (sorted global names), see `audit_pickle`.
PICKLE_AUDIT_SHA256 = "e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932"
CKPT_ALLOWED_GLOBALS = frozenset({"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage"})

# Architecture (config.json `pretrained_cfg` of the pinned snapshot) and data-contract facts.
ARCHITECTURE = "prithvi_eo_v2_300"
MAE_CONFIG: dict[str, Any] = {
    "img_size": 224,
    "patch_size": (1, 16, 16),
    "num_frames": 4,
    "in_chans": 6,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "decoder_embed_dim": 512,
    "decoder_depth": 8,
    "decoder_num_heads": 16,
    "mlp_ratio": 4.0,
    "coords_encoding": [],
    "coords_scale_learn": False,
    "mask_ratio": 0.75,
    "norm_pix_loss": False,
}
PARAMETER_COUNT = 330_419_712  # nn.Parameters of the encoder + decoder (no buffers)
STATE_TENSORS = 398
STATE_NUMEL = 331_625_472  # includes the fixed (non-parameter) positional-embedding buffers
ENCODER_TENSORS = 294
DECODER_TENSORS = 104
EMBED_DIM = 1024
PATCH = 16
MAX_FRAMES = 4  # the pretraining sequence length; fewer frames are accepted (the 3-D positional embedding is sliced)
NUM_CLASSES = 2
CLASS_NAMES: tuple[str, ...] = ("not burned", "burn scar")  # the tutorial's labelled task; a probe carries its own names
IGNORE_INDEX = -1
BANDS: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
S2_L1C_BAND_INDICES: tuple[int, ...] = (1, 2, 3, 8, 11, 12)
# The band statistics of the pinned config.json (reflectance × 10 000 scale, i.e. HLS digital numbers).
MEANS: tuple[float, ...] = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
STDS: tuple[float, ...] = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)
REFLECTANCE_SCALE = 10_000.0  # a chip in [0, 1] reflectance is multiplied by this before standardisation
CONSTANT_SCALE = 1e-4  # applied only when a chip arrives as reflectance × 10 000; HLS scenes in [0, 1] are left alone
# The plausible reflectance ceiling. A chip whose maximum exceeds it is read as reflectance × 10 000 and scaled by
# CONSTANT_SCALE; every checked chip is then refused unless it lies within [-0.5, REFLECTANCE_MAX]. Scaling and
# refusal share one threshold, so a checked record never triggers scaling again: re-checking it (as predict,
# evaluate and adapt do) is a no-op. A lower trigger (the former 1.0) rescaled bright checked chips a second time.
REFLECTANCE_MAX = 2.0
NO_DATA_VALUES: tuple[float, ...] = (0.0, -9999.0)  # replaced by 0 before normalisation, as the upstream inference script
NO_DATA_FLOAT = 0.0001  # what the upstream inference script writes into no-data pixels after normalisation
IMAGE_SIZE = 512  # labelled chips (the probe's training contract)
MIN_SIDE, MAX_SIDE = 64, 1024  # embedding stacks: any H, W multiple of 16 in this range
MIN_RECORDS = 4
MAX_RECORDS = 2_000
PATCH_VALID_FRACTION = 0.5  # a 16 × 16 patch needs at least this fraction of labelled pixels to carry a label
PATCH_POSITIVE_FRACTION = 0.5  # and is class 1 when at least this fraction of its labelled pixels are class 1
ADAPTATION_MODES = ("probe",)  # the only scope an adapter may declare: the linear head on frozen tokens
FINAL_LAYER = 23  # zero-based index of the last encoder block; its (post-norm) output is the embedding `embed` returns
FEATURE_LAYER = 11  # zero-based encoder block whose tokens feed the probe (block 12 of 24), chosen on the validation split
PROBE_TENSORS = ("probe.weight", "probe.bias", "feature_mean", "feature_std")


# --------------------------------------------------------------------------------------------------
# manifest, staging, static pickle audit and conversion
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    if SOURCE_CKPT_NAME not in listed:
        raise ValueError(f"manifest does not list {SOURCE_CKPT_NAME}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == SOURCE_CKPT_NAME and (size, digest) != (SOURCE_CKPT_BYTES, SOURCE_CKPT_SHA256):
            raise ValueError(f"{entry['path']}: manifest digest disagrees with the package constant")
    return manifest


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check the converted serving file (safetensors) against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    file_path = root / CONVERTED_WEIGHTS_NAME
    if not file_path.is_file():
        raise FileNotFoundError(f"converted file missing: {file_path}")
    size = file_path.stat().st_size
    if size != CONVERTED_BYTES:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: size {size} != pinned {CONVERTED_BYTES}")
    digest = _sha256_file(file_path)
    if digest != CONVERTED_SHA256:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: sha256 {digest} != pinned {CONVERTED_SHA256}")
    return {"files": [{"path": CONVERTED_WEIGHTS_NAME, "bytes": size, "sha256": digest}]}


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed Hub file) and, when the
    converted serving file is present, that against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    converted = (root / CONVERTED_WEIGHTS_NAME).is_file()
    if converted:
        verify_converted(root)
    return {**manifest, "converted": converted}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest and git-ignores the
    1.33 GB checkpoint, the example tiles and the safetensors the checkpoint converts to)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_globals(data: bytes) -> dict[str, int]:
    """Every global a pickle stream would import, collected with `pickletools.genops` (no execution)."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
    return found


def audit_pickle(path: str | Path, *, allowed: frozenset[str] = CKPT_ALLOWED_GLOBALS) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, or inside a torch zip archive) would import and refuse any
    outside `allowed`. Executes nothing. Returns the sorted globals and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    nested = 0
    if data[:4] == b"PK\x03\x04":
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                nested += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "torch_archive": data[:4] == b"PK\x03\x04",
        "pickles": nested if nested else 1,
        "globals": sorted(found),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}")
    return summary


def _check_pinned_source(root: Path) -> dict[str, Any]:
    source = root / SOURCE_CKPT_NAME
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != SOURCE_CKPT_BYTES:
        raise ValueError(f"{SOURCE_CKPT_NAME}: size {size} != pinned {SOURCE_CKPT_BYTES}")
    digest = _sha256_file(source)
    if digest != SOURCE_CKPT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: sha256 {digest} != pinned {SOURCE_CKPT_SHA256}")
    audit = audit_pickle(source)
    if audit["audit_sha256"] != PICKLE_AUDIT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: pickle audit digest {audit['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256}")
    return {"path": SOURCE_CKPT_NAME, "bytes": size, "sha256": digest, "audit": audit}


def build_model() -> Any:
    """Instantiate the masked autoencoder (encoder + decoder) from the installed `terratorch` package with the
    pinned configuration (no pretrained download)."""
    from terratorch.models.backbones.prithvi_mae import PrithviMAE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PrithviMAE(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in MAE_CONFIG.items()})


def convert_model(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the pinned checkpoint into safetensors, deterministically, after size, digest and static-audit
    checks: torch's weights-only unpickler, a strict load into the architecture built from `terratorch`, and the
    model's own state dict saved."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    source = _check_pinned_source(root)
    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    payload = torch.load(root / SOURCE_CKPT_NAME, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or any(not isinstance(v, torch.Tensor) for v in payload.values()):
        raise ValueError(f"{SOURCE_CKPT_NAME} did not unpickle to a state dict of tensors")
    if len(payload) != STATE_TENSORS:
        raise ValueError(f"{SOURCE_CKPT_NAME}: {len(payload)} tensors, expected {STATE_TENSORS}")
    model = build_model()
    model.load_state_dict(payload, strict=True)
    canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
    n_elements = sum(v.numel() for v in canonical.values())
    if len(canonical) != STATE_TENSORS or n_elements != STATE_NUMEL:
        raise ValueError(
            f"converted state dict has {len(canonical)} tensors / {n_elements} elements; expected {STATE_TENSORS} / {STATE_NUMEL}"
        )
    save_file(canonical, str(root / CONVERTED_WEIGHTS_NAME), metadata={"format": "pt"})
    report = verify_converted(root)
    return {
        "source": {k: v for k, v in source.items() if k != "audit"},
        "audit": source["audit"],
        "checkpoint": {
            "tensors": len(payload),
            "encoder_tensors": sum(k.startswith("encoder.") for k in payload),
            "decoder_tensors": sum(k.startswith("decoder.") for k in payload),
        },
        "converted": report["files"],
        "seconds": round(time.perf_counter() - started, 2),
    }


# --------------------------------------------------------------------------------------------------
# chips, labels and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": (
        "{id, image, label?}: image = (6, 512, 512) float32 reflectance chip (or a GeoTIFF path); "
        "label = (512, 512) int mask with 0/1/-1 (or a GeoTIFF path), optional"
    ),
    "bands": list(BANDS),
    "image_size": IMAGE_SIZE,
    "value_units": (
        "surface reflectance in [0, 1] (the HLS Burn Scars encoding, float32) or reflectance × 10 000; "
        "values above 2 (REFLECTANCE_MAX) are read as reflectance × 10 000 and scaled by 1e-4; "
        "re-checking a checked record never rescales it"
    ),
    "no_data": list(NO_DATA_VALUES),
    "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
    "ignore_index": IGNORE_INDEX,
    "records": [MIN_RECORDS, MAX_RECORDS],
    "validation": (
        "record shape, band count, chip size, finiteness, value range and label values only. Nothing checks that "
        "the bands are the six HLS bands in the right order, that the reflectance is atmospherically corrected, "
        "or that the label was drawn for this chip -- any six-band 512 × 512 array is embedded and probed without complaint"
    ),
}


def _read_tiff(path: Path) -> Any:
    """Read a GeoTIFF's pixel array with tifffile as (bands, H, W) or (H, W); no georeferencing is used."""
    import numpy as np
    import tifffile

    with tifffile.TiffFile(path) as tf:
        array = tf.asarray()
        planar = tf.pages[0].planarconfig
    if array.ndim == 3 and planar is not None and int(planar) == 1 and array.shape[-1] <= 16:
        array = np.moveaxis(array, -1, 0)  # pixel-interleaved -> band-sequential
    return np.asarray(array)


def read_chip(path: str | Path, *, band_indices: Sequence[int] | None = None) -> Any:
    """Load a chip from a GeoTIFF as float32 (6, H, W); `band_indices` selects the six HLS bands from a wider
    stack (e.g. S2_L1C_BAND_INDICES for a 13-band Sentinel-2 L1C file)."""
    import numpy as np

    array = _read_tiff(Path(path))
    if array.ndim != 3:
        raise ValueError(f"{Path(path).name}: expected a multi-band raster, got shape {array.shape}")
    if band_indices is not None:
        array = array[list(band_indices)]
    elif array.shape[0] != len(BANDS) and array.shape[0] == 13:
        array = array[list(S2_L1C_BAND_INDICES)]
    return np.ascontiguousarray(array.astype(np.float32))


def read_mask(path: str | Path) -> Any:
    """Load a label raster from a GeoTIFF as int64 (H, W)."""
    import numpy as np

    array = _read_tiff(Path(path))
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{Path(path).name}: a label raster must have one band, got shape {array.shape}")
        array = array[0]
    return np.ascontiguousarray(array.astype(np.int64))


def _check_record(record: Any, index: int) -> dict[str, Any]:
    import numpy as np

    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image[/label]")
    for key in ("id", "image"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image = record["id"], record["image"]
    if not isinstance(rid, str) or not rid or len(rid) > 128:
        raise ValueError(f"{label_name}: id must be a non-empty string of at most 128 characters")
    if isinstance(image, str | Path):
        if not Path(image).is_file():
            raise ValueError(f"{label_name}: image file not found: {image}")
        image = read_chip(image)
    try:
        array = np.asarray(image, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_name}: image must be a numeric array") from exc
    if array.shape != (len(BANDS), IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"{label_name}: image must have shape {(len(BANDS), IMAGE_SIZE, IMAGE_SIZE)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label_name}: image contains non-finite values")
    for value in NO_DATA_VALUES:
        array = np.where(array == value, 0.0, array)
    if float(array.max()) > REFLECTANCE_MAX:  # only reflectance × 10 000 exceeds the ceiling; see REFLECTANCE_MAX
        array = array * CONSTANT_SCALE
    if float(array.min()) < -0.5 or float(array.max()) > REFLECTANCE_MAX:
        span = (float(array.min()), float(array.max()))
        raise ValueError(f"{label_name}: reflectance outside the plausible range after scaling: {span}")
    item: dict[str, Any] = {"id": rid, "image": np.ascontiguousarray(array.astype(np.float32))}
    label = record.get("label")
    if label is not None:
        if isinstance(label, str | Path):
            if not Path(label).is_file():
                raise ValueError(f"{label_name}: label file not found: {label}")
            label = read_mask(label)
        try:
            mask = np.asarray(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label_name}: label must be an integer array") from exc
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{label_name}: label must have shape {(IMAGE_SIZE, IMAGE_SIZE)}, got {mask.shape}")
        if not np.issubdtype(mask.dtype, np.integer) and not np.all(mask == np.round(mask)):
            raise ValueError(f"{label_name}: label values must be integers")
        allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
        found = set(np.unique(mask).astype(int).tolist())
        if not found <= allowed:
            raise ValueError(f"{label_name}: label values {sorted(found - allowed)} outside {sorted(allowed)}")
        item["label"] = np.ascontiguousarray(mask.astype(np.int64))
    for key in ("split", "region", "source", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def check_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record and return its normalised copy (float32 reflectance, no-data replaced, int64 label)."""
    return _check_record(record, 0)


def chip_digest(record: Mapping[str, Any]) -> str:
    checked = _check_record(record, 0)
    digest = hashlib.sha256(checked["image"].tobytes())
    if "label" in checked:
        digest.update(checked["label"].tobytes())
    return digest.hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], chip_digest(r)] for r in records]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Structural validation of a chip dataset; raises ValueError before any model import."""
    import numpy as np

    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        if require_labels and "label" not in item:
            raise ValueError(f"records[{index}] has no label; every record of a labelled dataset needs one")
        if "label" in item:
            for c in range(NUM_CLASSES):
                class_pixels[c] += int((item["label"] == c).sum())
            ignored += int((item["label"] == IGNORE_INDEX).sum())
        checked.append(item)
    labelled = sum("label" in r for r in checked)
    if require_labels and labelled and class_pixels[1] == 0:
        raise ValueError(f"no pixel of class 1 ({CLASS_NAMES[1]}) in the dataset; nothing to learn or evaluate")
    total = int(class_pixels.sum())
    return {
        "records": checked,
        "n_records": len(checked),
        "n_labelled": labelled,
        "image_size": IMAGE_SIZE,
        "bands": list(BANDS),
        "class_pixel_fraction": {
            CLASS_NAMES[c]: round(float(class_pixels[c]) / total, 4) if total else None for c in range(NUM_CLASSES)
        },
        "ignored_pixels": ignored,
        "reflectance_range": [
            round(float(min(r["image"].min() for r in checked)), 4),
            round(float(max(r["image"].max() for r in checked)), 4),
        ],
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record; returns its id, shape, reflectance range and label class fractions."""
    item = _check_record(record, 0)
    report = {
        "id": item["id"],
        "shape": tuple(item["image"].shape),
        "reflectance_range": [round(float(item["image"].min()), 4), round(float(item["image"].max()), 4)],
        "has_label": "label" in item,
    }
    if "label" in item:
        label = item["label"]
        valid = int((label != IGNORE_INDEX).sum())
        report["label_fraction"] = {
            CLASS_NAMES[c]: round(float((label == c).sum()) / max(valid, 1), 4) for c in range(NUM_CLASSES)
        }
        report["ignored_pixels"] = int((label == IGNORE_INDEX).sum())
    return report


# --------------------------------------------------------------------------------------------------
# embedding stacks (time series) — a second, label-free record kind
# --------------------------------------------------------------------------------------------------


def _check_stack(record: Any, index: int) -> dict[str, Any]:
    """`{id, frames}`: frames = (6, T, H, W) reflectance (or a list of 1..4 GeoTIFF paths in chronological order)
    with T in 1..MAX_FRAMES and H, W multiples of 16 in [MIN_SIDE, MAX_SIDE]; the same scaling and no-data rules
    as chips."""
    import numpy as np

    label_name = f"records[{index}]"
    if not isinstance(record, Mapping) or "id" not in record or "frames" not in record:
        raise ValueError(f"{label_name} must be a mapping with id/frames")
    rid, frames = record["id"], record["frames"]
    if not isinstance(rid, str) or not rid or len(rid) > 128:
        raise ValueError(f"{label_name}: id must be a non-empty string of at most 128 characters")
    if isinstance(frames, str | Path):
        frames = [frames]
    if isinstance(frames, list | tuple) and frames and all(isinstance(f, str | Path) for f in frames):
        loaded = []
        for f in frames:
            if not Path(f).is_file():
                raise ValueError(f"{label_name}: frame file not found: {f}")
            chip = read_chip(f)
            if chip.shape[0] != len(BANDS):
                raise ValueError(f"{label_name}: frame {Path(f).name} has {chip.shape[0]} bands, expected {len(BANDS)}")
            loaded.append(chip)
        frames = np.stack(loaded, axis=1)
    try:
        array = np.asarray(frames, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_name}: frames must be a numeric array") from exc
    if array.ndim == 3:
        array = array[:, None]
    if array.ndim != 4 or array.shape[0] != len(BANDS):
        raise ValueError(f"{label_name}: frames must have shape (6, T, H, W), got {array.shape}")
    _bands, t, h, w = array.shape
    if not 1 <= t <= MAX_FRAMES:
        raise ValueError(f"{label_name}: T must be in 1..{MAX_FRAMES}, got {t}")
    for side in (h, w):
        if not MIN_SIDE <= side <= MAX_SIDE or side % PATCH:
            raise ValueError(f"{label_name}: H and W must be multiples of {PATCH} in [{MIN_SIDE}, {MAX_SIDE}], got {h} × {w}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label_name}: frames contain non-finite values")
    for value in NO_DATA_VALUES:
        array = np.where(array == value, 0.0, array)
    if float(array.max()) > REFLECTANCE_MAX:  # only reflectance × 10 000 exceeds the ceiling; see REFLECTANCE_MAX
        array = array * CONSTANT_SCALE
    if float(array.min()) < -0.5 or float(array.max()) > REFLECTANCE_MAX:
        span = (float(array.min()), float(array.max()))
        raise ValueError(f"{label_name}: reflectance outside the plausible range after scaling: {span}")
    item: dict[str, Any] = {"id": rid, "frames": np.ascontiguousarray(array.astype(np.float32))}
    for key in ("source", "dates", "region"):
        if key in record:
            item[key] = record[key]
    return item


def validate_stacks(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Structural validation of embedding records; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, frames} mappings")
    if not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError(f"{len(records)} records; 1..{MAX_RECORDS} are required")
    checked = []
    ids: set[str] = set()
    for index, record in enumerate(records):
        item = _check_stack(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "shapes": sorted({tuple(int(x) for x in r["frames"].shape) for r in checked}),
        "model_id": MODEL_ID,
    }


def read_stack(paths: Sequence[str | Path]) -> Any:
    """Read 1..4 chronologically ordered GeoTIFFs into a (6, T, H, W) float32 reflectance stack."""
    import numpy as np

    return np.stack([read_chip(p) for p in paths], axis=1)


def patch_labels(label: Any, *, patch: int = PATCH) -> Any:
    """Reduce a (H, W) pixel mask with 0 / 1 / -1 to the (H/16, W/16) patch grid: a patch is class 1 when at least
    `PATCH_POSITIVE_FRACTION` of its labelled pixels are class 1, class 0 otherwise, and ignored (-1) when fewer
    than `PATCH_VALID_FRACTION` of its pixels are labelled."""
    import numpy as np

    mask = np.asarray(label)
    h, w = mask.shape
    if h % patch or w % patch:
        raise ValueError(f"label shape {mask.shape} is not a multiple of {patch}")
    blocks = mask.reshape(h // patch, patch, w // patch, patch).transpose(0, 2, 1, 3).reshape(h // patch, w // patch, -1)
    valid = blocks != IGNORE_INDEX
    n_valid = valid.sum(axis=-1)
    positive = ((blocks == 1) & valid).sum(axis=-1)
    out = np.full((h // patch, w // patch), IGNORE_INDEX, dtype=np.int64)
    labelled = n_valid >= PATCH_VALID_FRACTION * patch * patch
    out[labelled] = 0
    out[labelled & (positive >= PATCH_POSITIVE_FRACTION * np.maximum(n_valid, 1))] = 1
    return out


# --------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------


def _normalise(frames: Any) -> Any:
    """(B, 6, T, H, W) reflectance in [0, 1] -> HLS digital-number scale, standardised with the pinned statistics."""
    import numpy as np

    mean = np.asarray(MEANS, dtype=np.float32)[None, :, None, None, None]
    std = np.asarray(STDS, dtype=np.float32)[None, :, None, None, None]
    return (frames * REFLECTANCE_SCALE - mean) / std


@dataclass
class PrithviFeaturePipeline:
    """Embeddings, masked reconstruction and a bounded linear probe on top of the verified Prithvi-EO-2.0-300M MAE."""

    model: Any
    device: str
    weights_dir: Path
    source: str
    probe: Any = None  # torch.nn.Linear(EMBED_DIM, NUM_CLASSES) once adapted or loaded
    feature_stats: Any = None  # (mean, std) tensors of shape (EMBED_DIM,) used to standardise tokens for the probe
    feature_layer: int = FEATURE_LAYER  # the encoder block the probe reads (recorded in the adapter, checked on load)
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
        report: Callable[[dict[str, Any]], None] | None = None,
    ) -> PrithviFeaturePipeline:
        """Verify, convert if needed, build from the installed package and strictly load. With
        `require_source=False` the checkpoint may be absent (the DIMER-hosted case) as long as the converted file
        verifies. `report` receives the audit and conversion records when a conversion happens."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if require_source:
            stage_missing_files(root, allow_download=allow_download)
            snapshot = verify_snapshot(root)
            if not snapshot["converted"]:
                conversion = convert_model(root)
                if report is not None:
                    report({"conversion": conversion})
                snapshot = verify_snapshot(root)
            elif report is not None:
                report({"conversion": "converted file already present and digest-verified"})
            source = "converted from the manifest-verified source checkpoint"
        else:
            verify_converted(root)
            source = "converted file, pinned digest (source checkpoint not required)"
        import torch
        from safetensors.torch import load_file

        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if chosen.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        model = build_model()
        state = load_file(str(root / CONVERTED_WEIGHTS_NAME))
        model.load_state_dict(state, strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != PARAMETER_COUNT:
            raise ValueError(f"rebuilt model has {n_params} parameters, expected {PARAMETER_COUNT}")
        model.to(torch.device(chosen)).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model=model, device=chosen, weights_dir=root, source=source)

    # ---- forward ---------------------------------------------------------------------------------------

    def _tokens(self, frames: Any, *, layer: int = FINAL_LAYER) -> Any:
        """(6, T, H, W) reflectance -> (T, H/16, W/16, 1024) float32 patch tokens of encoder block `layer` (in
        spatial order; every patch visible, no masking) and the (1024,) CLS embedding of the final, normalised
        layer. The final layer's tokens are post-norm; earlier layers' are the raw block outputs."""
        import numpy as np
        import torch

        if not 0 <= layer <= FINAL_LAYER:
            raise ValueError(f"layer must be in 0..{FINAL_LAYER}")
        batch = torch.from_numpy(_normalise(np.asarray(frames, dtype=np.float32)[None])).to(self.device)
        _b, _c, t, h, w = batch.shape
        with torch.inference_mode():
            features = self.model.encoder.forward_features(batch)
        cls = features[-1][0, 0].float().cpu().numpy()
        tokens = features[layer][0, 1:].float().cpu()
        return tokens.reshape(t, h // PATCH, w // PATCH, EMBED_DIM).numpy(), cls

    # ---- embeddings and reconstruction ---------------------------------------------------------------------

    def embed(self, records: Sequence[Mapping[str, Any]], *, keep_tokens: bool = False) -> dict[str, Any]:
        """Scene embeddings of reflectance stacks: per record the CLS embedding (1024,), the mean over all patch
        tokens (1024,), the token grid shape and, with `keep_tokens=True`, the (T, H/16, W/16, 1024) token grid.
        Embeddings are float32 encoder outputs without any normalisation or projection."""
        checked = validate_stacks(records)["records"]
        started = time.perf_counter()
        out = []
        for record in checked:
            tokens, cls = self._tokens(record["frames"])
            entry = {
                "id": record["id"],
                "cls": cls,
                "mean": tokens.reshape(-1, EMBED_DIM).mean(axis=0),
                "token_grid": [int(x) for x in tokens.shape[:3]],
                "frames": int(record["frames"].shape[1]),
            }
            if keep_tokens:
                entry["tokens"] = tokens
            out.append(entry)
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "embedding": (
                "final (normalised) encoder layer, every patch visible; CLS token and the mean of the patch tokens; "
                "float32, unnormalised"
            ),
            "dim": EMBED_DIM,
            "embeddings": out,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def reconstruct(
        self, records: Sequence[Mapping[str, Any]], *, mask_ratio: float = 0.75, seed: int = 0, keep_images: bool = False
    ) -> dict[str, Any]:
        """Masked-patch reconstruction (the pretraining task) with a seeded random mask: per record the MSE over the
        masked patches in standardised units, and the same MSE for a **mean-fill baseline** that predicts every
        masked patch as the per-band, per-frame mean of the visible pixels. With `keep_images=True` the
        reconstructed stack (visible patches kept, masked patches predicted) and the mask are returned in
        reflectance units."""
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")
        checked = validate_stacks(records)["records"]
        import numpy as np
        import torch

        started = time.perf_counter()
        results = []
        for index, record in enumerate(checked):
            torch.manual_seed(seed + index)
            batch = torch.from_numpy(_normalise(record["frames"][None])).to(self.device)
            with torch.inference_mode():
                loss, pred, mask = self.model(batch, None, None, mask_ratio)  # pred and mask are un-patchified
            loss_value = float(loss["loss"] if isinstance(loss, Mapping) else loss)
            mask_b = mask[:, None].expand(-1, len(BANDS), -1, -1, -1).to(batch.dtype)  # (1, 6, T, H, W); 1 = masked
            target_patches = self.model.patchify(batch)
            mask_patches = self.model.patchify(mask_b).mean(-1)
            visible = mask_b == 0
            baseline = batch.clone()
            for band in range(len(BANDS)):
                for frame in range(batch.shape[2]):
                    plane = batch[0, band, frame]
                    keep = visible[0, band, frame]
                    baseline[0, band, frame][~keep] = plane[keep].mean() if bool(keep.any()) else 0.0
            baseline_loss = (
                ((self.model.patchify(baseline) - target_patches) ** 2).mean(-1) * mask_patches
            ).sum() / mask_patches.sum()
            entry = {
                "id": record["id"],
                "masked_fraction": round(float(mask.float().mean()), 4),
                "masked_mse": round(loss_value, 6),
                "baseline_mean_fill_mse": round(float(baseline_loss), 6),
            }
            if keep_images:
                mean = np.asarray(MEANS, dtype=np.float32)[:, None, None, None]
                std = np.asarray(STDS, dtype=np.float32)[:, None, None, None]
                merged = torch.where(mask_b.bool(), pred.to(batch.dtype), batch)[0].float().cpu().numpy()
                entry["reconstruction"] = ((merged * std + mean) / REFLECTANCE_SCALE).astype(np.float32)
                entry["mask"] = mask[0].float().cpu().numpy().astype(np.uint8)
            results.append(entry)
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
            "metric": "MSE over the masked patches in standardised units (the pretraining loss), against a mean-fill baseline",
            "mask_ratio": mask_ratio,
            "seed": seed,
            "results": results,
            "seconds": round(time.perf_counter() - started, 3),
        }

    # ---- probe: prediction and evaluation on labelled chips ----------------------------------------------

    def _probe_logits(self, tokens: Any) -> Any:
        """(N, 1024) numpy or torch tokens -> (N, 2) torch logits with the attached probe (a zero head before adaptation)."""
        import numpy as np
        import torch

        feats = (
            tokens.to(self.device)
            if isinstance(tokens, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(tokens)).to(self.device)
        )
        if self.feature_stats is not None:
            mean, std = self.feature_stats
            feats = (feats - mean.to(self.device)) / std.to(self.device)
        if self.probe is None:
            return torch.zeros(feats.shape[0], NUM_CLASSES, device=self.device)
        return self.probe(feats)

    def predict(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Patch-level class map of labelled-chip records (6, 512, 512) with the attached probe: per record the
        (32, 32) argmax map, the (2, 32, 32) softmax scores and the class fractions. Before adaptation the probe
        is a zero head, so every patch is class 0 — the majority baseline, printed as such."""
        import numpy as np
        import torch

        checked = validate_dataset(records, min_records=1, require_labels=False)["records"]
        started = time.perf_counter()
        predictions = []
        for record in checked:
            tokens, _cls = self._tokens(record["image"][:, None], layer=self.feature_layer)
            grid = tokens.reshape(-1, EMBED_DIM)
            with torch.inference_mode():
                scores = torch.softmax(self._probe_logits(grid), dim=1).cpu().numpy()
            side = tokens.shape[1], tokens.shape[2]
            scores = scores.T.reshape(NUM_CLASSES, *side).astype(np.float32)
            mask = scores.argmax(axis=0).astype(np.uint8)
            predictions.append(
                {
                    "id": record["id"],
                    "mask": mask,
                    "scores": scores,
                    "class_fraction": {CLASS_NAMES[c]: round(float((mask == c).mean()), 4) for c in range(NUM_CLASSES)},
                }
            )
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "classes": list(CLASS_NAMES),
            "resolution": f"one prediction per {PATCH} × {PATCH} pixel patch",
            "decision_rule": "argmax over the two probe scores (no threshold); a zero head before adaptation",
            "predictions": predictions,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Patch-level metrics on labelled chips (patches with too few labelled pixels excluded): per-class IoU,
        mean IoU, accuracy and the class-1 F1, precision and recall, with the all-not-burned baseline scored on the
        same patches."""
        from .metrics import majority_baseline, segmentation_metrics

        checked = validate_dataset(records, min_records=1)["records"]
        started = time.perf_counter()
        result = self.predict(checked)
        masks = [p["mask"] for p in result["predictions"]]
        labels = [patch_labels(r["label"]) for r in checked]
        metrics = segmentation_metrics(masks, labels)
        return {
            "n_records": len(checked),
            "metric": f"patch-level ({PATCH} × {PATCH} pixels) IoU / F1 over the labelled patches of the held-out chips",
            "model": metrics,
            "baseline_not_burned": majority_baseline(labels),
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 3),
        }

    # ---- adaptation: a linear probe on frozen tokens -------------------------------------------------------

    def _probe_cache(self, records: Sequence[Mapping[str, Any]]) -> list[tuple[Any, Any]]:
        """One encoder pass per labelled chip: (tokens (32 × 32, 1024) float32, patch-label grid (32, 32)) per record."""
        return [
            (self._tokens(r["image"][:, None], layer=self.feature_layer)[0].reshape(-1, EMBED_DIM), patch_labels(r["label"]))
            for r in records
        ]

    @staticmethod
    def _flatten_cache(cache: Sequence[tuple[Any, Any]]) -> tuple[Any, Any]:
        """Labelled patches of a cache, flattened: (N, 1024) features and (N,) labels."""
        import numpy as np

        feats = [tokens[grid.reshape(-1) != IGNORE_INDEX] for tokens, grid in cache]
        labels = [grid.reshape(-1)[grid.reshape(-1) != IGNORE_INDEX] for _tokens, grid in cache]
        return np.concatenate(feats), np.concatenate(labels)

    def _metrics_from_cache(self, cache: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
        """Patch-level metrics of the attached probe on cached tokens (no encoder pass)."""
        import torch

        from .metrics import segmentation_metrics

        masks = []
        with torch.inference_mode():
            for tokens, grid in cache:
                masks.append(self._probe_logits(tokens).argmax(dim=1).cpu().numpy().reshape(grid.shape).astype("uint8"))
        return segmentation_metrics(masks, [grid for _tokens, grid in cache])

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 30,
        lr: float = 1e-2,
        batch_size: int = 4096,
        trainable: str = "probe",
        class_balance: bool = False,
        feature_layer: int = FEATURE_LAYER,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of a **linear probe** (a 1024 → 2 head on standardised frozen token embeddings) on
        labelled chips: the encoder is run once per chip and stores no gradient, the head is trained with
        cross-entropy (class-balanced by inverse training frequency when `class_balance`) and Adam at a fixed learning
        rate over seeded mini-batches of tokens, and the epoch with the lowest validation loss is kept. Epoch 0
        records the untrained (zero) head, which is the majority baseline."""
        if trainable not in ADAPTATION_MODES:
            raise ValueError(f"trainable must be one of {ADAPTATION_MODES}")
        if not isinstance(epochs, int) or not 1 <= epochs <= 500:
            raise ValueError("epochs must be an int in 1..500")
        if not (0.0 < lr <= 1.0):
            raise ValueError("lr must be in (0, 1]")
        if not isinstance(batch_size, int) or not 16 <= batch_size <= 65_536:
            raise ValueError("batch_size must be an int in 16..65536")
        if not isinstance(feature_layer, int) or not 0 <= feature_layer <= FINAL_LAYER:
            raise ValueError(f"feature_layer must be an int in 0..{FINAL_LAYER}")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        import numpy as np
        import torch

        started = time.perf_counter()
        previous = (self.probe, self.feature_stats, self.feature_layer, self.adapter)
        try:
            self.feature_layer = feature_layer
            x_train, y_train = self._flatten_cache(self._probe_cache(train_checked))
            if len(np.unique(y_train)) < 2:
                raise ValueError("the training chips carry only one patch class; nothing to learn")
            mean = torch.from_numpy(x_train.mean(axis=0))
            std = torch.from_numpy(x_train.std(axis=0) + 1e-6)
            self.feature_stats = (mean, std)
            torch.manual_seed(seed)
            probe = torch.nn.Linear(EMBED_DIM, NUM_CLASSES).to(self.device)
            torch.nn.init.zeros_(probe.weight)
            torch.nn.init.zeros_(probe.bias)
            self.probe = probe
            xt = torch.from_numpy(x_train).to(self.device)
            yt = torch.from_numpy(y_train).to(self.device)
            positive = float(y_train.mean())
            weight = None
            if class_balance:
                weight = torch.tensor(
                    [0.5 / max(1.0 - positive, 1e-6), 0.5 / max(positive, 1e-6)], dtype=torch.float32, device=self.device
                )
            xv = yv = val_cache = None
            if val_checked is not None:
                val_cache = self._probe_cache(val_checked)
                x_val, y_val = self._flatten_cache(val_cache)
                xv, yv = torch.from_numpy(x_val).to(self.device), torch.from_numpy(y_val).to(self.device)

            def val_loss() -> float | None:
                if xv is None:
                    return None
                with torch.inference_mode():
                    return float(torch.nn.functional.cross_entropy(self._probe_logits(xv), yv, weight=weight))

            optimiser = torch.optim.Adam(probe.parameters(), lr=lr)
            rng = np.random.default_rng(seed)
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {
                "epoch": 0,
                "train_loss": None,
                "val_loss": val_loss(),
                "note": "zero head (majority baseline)",
            }
            if val_cache is not None:
                entry["val"] = self._metrics_from_cache(val_cache)
            history.append(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            best_epoch = 0
            if progress:
                progress(entry)
            n_steps = 0
            for epoch in range(1, epochs + 1):
                order = rng.permutation(len(y_train)).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    idx = torch.as_tensor(order[start : start + batch_size], device=self.device)
                    with torch.enable_grad():
                        loss = torch.nn.functional.cross_entropy(self._probe_logits(xt[idx]), yt[idx], weight=weight)
                        optimiser.zero_grad(set_to_none=True)
                        loss.backward()
                        optimiser.step()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_loss()}
                if val_cache is not None:
                    entry["val"] = self._metrics_from_cache(val_cache)
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
                    best_epoch = epoch
            probe.load_state_dict(best_state)
        except BaseException:
            # Transactional: a failure leaves the pipeline exactly as it was before adapt().
            self.probe, self.feature_stats, self.feature_layer, self.adapter = previous
            raise
        for param in probe.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable": trainable,
            "trainable_names": list(PROBE_TENSORS),
            "n_trainable": sum(p.numel() for p in probe.parameters()),
            "n_total": sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in probe.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "feature_layer": feature_layer,
            "features": (
                f"patch tokens of frozen encoder block {feature_layer} (zero-based), "
                "standardised with the training-set mean and std"
            ),
            "loss": "cross-entropy, class-balanced by inverse training frequency" if class_balance else "cross-entropy",
            "class_balance": class_balance,
            "n_train_records": len(train_checked),
            "n_train_patches": int(len(y_train)),
            "positive_fraction_train": round(positive, 4),
            "n_steps": n_steps,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts -------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the probe (weight, bias) and the feature statistics as safetensors with a manifest."""
        if self.adapter is None or self.probe is None or self.feature_stats is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tensors = {
            "probe.weight": self.probe.weight.detach().cpu().contiguous(),
            "probe.bias": self.probe.bias.detach().cpu().contiguous(),
            "feature_mean": self.feature_stats[0].detach().cpu().contiguous(),
            "feature_std": self.feature_stats[1].detach().cpu().contiguous(),
        }
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "converted_sha256": CONVERTED_SHA256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "classes": list(CLASS_NAMES),
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> tuple[Path, str]:
        """Static checks on a probe manifest, before any model or weights work: format and version, the pinned base
        and converted digest, exactly one weights entry named `adapter.safetensors` inside the artifact directory,
        the declared scope, and the exact tensor list. Returns the weights path and mode."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("converted_sha256") != CONVERTED_SHA256:
            raise ValueError("artifact records a different converted-base digest")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        mode = adapter.get("trainable") if isinstance(adapter, Mapping) else None
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"artifact adapter.trainable must be one of {ADAPTATION_MODES}")
        layer = adapter.get("feature_layer")
        if not isinstance(layer, int) or not 0 <= layer <= FINAL_LAYER:
            raise ValueError(f"artifact adapter.feature_layer must be an int in 0..{FINAL_LAYER}")
        if sorted(manifest.get("tensors") or []) != sorted(PROBE_TENSORS):
            raise ValueError(f"artifact tensor list must be exactly {sorted(PROBE_TENSORS)}")
        return weights_path, mode

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify a probe artifact's manifest, scope and digest, then attach the head and feature statistics."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path, _mode = self.check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        import torch
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != sorted(PROBE_TENSORS):
            raise ValueError("artifact tensor names differ from the validated manifest")
        expected = {
            "probe.weight": (NUM_CLASSES, EMBED_DIM),
            "probe.bias": (NUM_CLASSES,),
            "feature_mean": (EMBED_DIM,),
            "feature_std": (EMBED_DIM,),
        }
        for key, shape in expected.items():
            if tuple(tensors[key].shape) != shape:
                raise ValueError(f"artifact tensor {key} has shape {tuple(tensors[key].shape)}, expected {shape}")
        probe = torch.nn.Linear(EMBED_DIM, NUM_CLASSES).to(self.device)
        probe.load_state_dict({"weight": tensors["probe.weight"].to(self.device), "bias": tensors["probe.bias"].to(self.device)})
        for param in probe.parameters():
            param.requires_grad_(False)
        self.probe = probe
        self.feature_stats = (tensors["feature_mean"].float(), tensors["feature_std"].float())
        self.feature_layer = int(manifest["adapter"]["feature_layer"])
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
    ) -> PrithviFeaturePipeline:
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.check_artifact_manifest(root, manifest)
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download, require_source=require_source
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
