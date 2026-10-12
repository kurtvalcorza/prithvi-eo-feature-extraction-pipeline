"""Stage runner for the DIMER Philippines Rice Mapping Capstone (E2E, WORKSHOP; capstone specification
docs/philippines-rice-mapping-capstone-spec.md).

The notebook writes this file, the repository's verbatim `pipeline.py` (as `prithvi_pipeline.py`), the hash lock and
the manifests to a fresh run directory and runs each stage in a child process of the isolated environment:

    prepare -> baseline -> features -> train-head -> activity -> [finetune] -> evaluate -> export -> reload -> report
    byod (optional, after reload)

Everything above the stage functions is numpy-only and importable without torch, h5py, lightgbm or matplotlib, so the
repository's CPU tests exercise the scoring and bookkeeping contract on synthetic arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:  # the notebook carries the repository's pipeline.py beside this file
    import prithvi_pipeline as pp
except ImportError:  # repository checkout (tests)
    from prithvi_eo_feature_extraction_pipeline import pipeline as pp

# --------------------------------------------------------------------------------------------------
# identities and frozen protocol constants
# --------------------------------------------------------------------------------------------------

DATASET_SLUG = "kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching"
DATASET_VERSION = 1  # v2 and v3 (2026-10-11) change only the card and rights wording; Kaggle built no download
# archive for them, so the notebook pins v1, whose data, INDEX.jsonl and member hashes are identical
INDEX_SHA256 = "e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d"
# Label recipe of the dataset build (issue #22 M2): every INDEX row records it; it pins the annual-layer, code-map,
# nearest-neighbour and edge-band rules described in Section 2.
LABEL_RECIPE_SHA256 = "7adf68fe9d23622a1cf008f71cacd7c5006ed1377e34a5fc8c17c046953dc63e"
DATASET_LICENSE = "CC BY 4.0"
DEVELOPMENT_EXAMPLE_GROUP = "ph_96_766"  # a parent training sample: never used for selection or scoring

N_BINS = 24
N_BANDS = 6
CHIP = 128
QUARTERS = 4
BINS_PER_QUARTER = N_BINS // QUARTERS
BAND_NAMES = ("B2", "B3", "B4", "B8A", "B11", "B12")  # = pp.BANDS order: blue, green, red, narrow NIR, SWIR1, SWIR2
IGNORE = 255
PIXEL_METRES = 20.0
EPSG = 32651
REFLECTANCE_MIN = -0.5
REFLECTANCE_MAX = pp.REFLECTANCE_MAX  # 2.0: larger values are reflectance x 10 000 and are refused, never rescaled
FEATURE_LAYER = pp.FEATURE_LAYER  # 11 = block 12 of 24 (D3, fixed)
PATCH = pp.PATCH
EMBED_DIM = pp.EMBED_DIM

SPLIT_SALT = "rice-capstone-split"
DEV_FRACTION = 0.25
MIN_DEV_RICE_CHIPS = 2
THRESHOLDS = tuple(i / 100 for i in range(10, 91, 5))
SPLIT_RULE = (
    "whole regions ordered by SHA-256('rice-capstone-split:<k>:<region>') are added to development while it holds at "
    "most round(0.25 x training chips); the first k = 0, 1, ... giving at least target - 2 development chips, at least "
    "two rice development chips and both classes in fit is used"
)

SEED = 0
PIXEL_SEED = 20261003
PIXEL_CAP = 65_536
LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary", "metric": "None", "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 1.0, "lambda_l2": 1.0, "deterministic": True, "force_col_wise": True,
    "device_type": "cpu", "num_threads": 2, "seed": SEED, "feature_fraction_seed": SEED, "data_random_seed": SEED,
    "verbosity": -1,
}
LGBM_ROUND_CAP = 200
LGBM_PATIENCE = 50

HEAD_HIDDEN = 256
HEAD_CONFIG: dict[str, Any] = {
    "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.01, "batch_size": 8, "max_epochs": 30,
    "seed": SEED, "loss": "cross-entropy, ignore_index=255, no class weights", "augmentation": "none",
    "epoch_rule": "highest development rice F1 at threshold 0.5; earliest epoch on ties",
}
FT_CONFIG: dict[str, Any] = {
    "trainable": "patch_embed + cls_token + encoder blocks 1-12 (the blocks feeding block 12); blocks 13-24 and the "
                 "decoder frozen",
    "optimizer": "AdamW", "encoder_learning_rate": 1e-5, "head_learning_rate": 1e-3, "weight_decay": 0.01,
    "batch_size": 4, "epochs": 10, "seed": SEED, "precision": "fp16 autocast + GradScaler",
    "epoch_rule": "highest development rice F1 at threshold 0.5; earliest epoch on ties",
}
ARTIFACT_FORMAT = "org.valcorza.prithvi-rice-capstone.head.v1"
ARTIFACT_WEIGHTS = "adapter.safetensors"
ARTIFACT_MANIFEST = "manifest.json"
HEAD_TENSORS = ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias")
RELOAD_TOLERANCE = 1e-4
BYOD_MAX_FILES = 200
BYOD_MAX_BYTES = 4 * 2**30  # expanded size of everything collected for one BYOD run
BYOD_MIN_ADAPT_CHIPS = 6

M1_STATS = ("mean", "std", "min", "max", "p10", "p50", "p90")
M1_NAMES = [f"{b}_{s}" for b in BAND_NAMES for s in M1_STATS]
M1_NAMES += [f"q{q + 1}_{b}_mean" for q in range(QUARTERS) for b in BAND_NAMES]
M1_NAMES += [f"q{q + 1}_observed_fraction" for q in range(QUARTERS)]

ATTRIBUTION = (
    "Rice labels: Chen et al., 20 m paddy rice cropping-intensity maps of Monsoon Asia (Zenodo, "
    "https://doi.org/10.5281/zenodo.15124936), CC BY 4.0 - annual layer of each chip's year, codes 1-3 -> rice, "
    "0 -> non-rice, nearest-neighbour to the 20 m chip grid, class-edge pixels ignored (label recipe "
    "7adf68fe9d23622a1cf008f71cacd7c5006ed1377e34a5fc8c17c046953dc63e).\n"
    "Contains modified Copernicus Sentinel data 2019-2021.\n"
    "AlphaEarth Foundations Satellite Embedding (Google / Google DeepMind; Brown et al., 2025), CC BY 4.0 - carried "
    "in the dataset files; not read or shown by this notebook.\n"
    "Administrative regions: geoBoundaries PHL ADM1, CC BY 3.0 IGO.\n"
    "Dataset compilation: Kurt Valcorza, rice-philippines-s2-alphaearth-v1-teaching (Kaggle, version 1), CC BY 4.0.\n"
    "Model: ibm-nasa-geospatial/Prithvi-EO-2.0-300M (Apache-2.0), revision " + pp.MODEL_REVISION + ".\n"
)


# --------------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing: run the earlier stages in order (Runtime > Run all).")
    return json.loads(path.read_text(encoding="utf-8"))


def chip_id(row: Mapping[str, Any]) -> str:
    return f"{row['group_id']}_{row['year']}"


# --------------------------------------------------------------------------------------------------
# chip schema (the BYOD contract is the dataset's own schema)
# --------------------------------------------------------------------------------------------------


class ChipError(ValueError):
    """A chip does not satisfy the BYOD schema stated in Section 12; the message names the failed rule."""


def validate_chip(arrays: Mapping[str, Any], *, name: str = "chip", require_label: bool = True) -> dict[str, Any]:
    """Check one chip against the BYOD schema and return a normalised copy: S2 float32 with NaN wherever S2_valid is 0,
    S2_valid uint8, label uint8 (or None). Raises ChipError with an actionable message."""
    for key in ("S2", "S2_valid"):
        if key not in arrays:
            raise ChipError(f"{name}: missing key {key!r}; expected keys S2, S2_valid and label (see the schema)")
    s2 = np.asarray(arrays["S2"])
    valid = np.asarray(arrays["S2_valid"])
    if s2.shape != (N_BINS, N_BANDS, CHIP, CHIP):
        raise ChipError(f"{name}: S2 has shape {s2.shape}; expected {(N_BINS, N_BANDS, CHIP, CHIP)} (bins, bands, rows, cols)")
    if not np.issubdtype(s2.dtype, np.floating):
        raise ChipError(f"{name}: S2 has dtype {s2.dtype}; expected float32 reflectance")
    if valid.shape != (N_BINS, CHIP, CHIP):
        raise ChipError(f"{name}: S2_valid has shape {valid.shape}; expected {(N_BINS, CHIP, CHIP)}")
    if not np.isin(valid, (0, 1)).all():
        raise ChipError(f"{name}: S2_valid must contain only 0 and 1")
    valid = valid.astype(np.uint8)
    observed = np.broadcast_to(valid[:, None].astype(bool), s2.shape)
    values = s2[observed]
    if values.size and not np.isfinite(values).all():
        raise ChipError(f"{name}: S2 has non-finite values where S2_valid = 1; mark those observations invalid")
    if values.size and float(values.max()) > REFLECTANCE_MAX:
        raise ChipError(
            f"{name}: S2 maximum {float(values.max()):.1f} exceeds {REFLECTANCE_MAX}; the file looks like reflectance "
            "x 10 000 - multiply by 0.0001 before uploading"
        )
    if values.size and float(values.min()) < REFLECTANCE_MIN:
        raise ChipError(f"{name}: S2 minimum {float(values.min()):.3f} is below {REFLECTANCE_MIN}; not surface reflectance")
    clean = np.where(observed, s2, np.nan).astype(np.float32)
    label = None
    if "label" in arrays:
        label = np.asarray(arrays["label"])
        if label.shape != (CHIP, CHIP):
            raise ChipError(f"{name}: label has shape {label.shape}; expected {(CHIP, CHIP)}")
        if not np.isin(label, (0, 1, IGNORE)).all():
            bad = sorted(set(np.unique(label).tolist()) - {0, 1, IGNORE})[:5]
            raise ChipError(f"{name}: label values {bad} are outside 0 (non-rice), 1 (rice), 255 (ignore)")
        label = label.astype(np.uint8)
    elif require_label:
        raise ChipError(f"{name}: missing key 'label'; scoring and adaptation need the 0/1/255 rice label")
    return {"S2": clean, "S2_valid": valid, "label": label}


def georeference_attrs(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Upper-left x_min/y_max and the site group from HDF5 attributes. Accepts this dataset's own schema (`affine` as a
    JSON list or numeric array [a, b, c, d, e, f], `crs`, `group_id`) and the plain `x_min`/`y_max`/`group` form.
    Coordinates must be finite; plain ones are refused with any CRS other than EPSG:32651, and an affine grid is used
    only when it is EPSG:32651, 20 m and north-up. Anything else stays ungeoreferenced rather than mislocated."""

    def text(value: Any) -> str | None:
        try:
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        except UnicodeDecodeError:
            return None  # an undecodable optional attribute is unusable, not fatal

    crs = (text(raw["crs"]) or "<undecodable>") if "crs" in raw else None
    attrs: dict[str, Any] = {}
    if "x_min" in raw and "y_max" in raw and crs in (None, "EPSG:32651"):
        try:
            x_min, y_max = float(raw["x_min"]), float(raw["y_max"])
        except (TypeError, ValueError):
            x_min = y_max = math.nan
        if math.isfinite(x_min) and math.isfinite(y_max):
            attrs.update(x_min=x_min, y_max=y_max)
    if "affine" in raw and not attrs:
        try:
            value = raw["affine"]
            numbers = json.loads(text(value)) if isinstance(value, (str, bytes)) else list(value)
            a, b, c, d, e, f = (float(v) for v in numbers)
        except (TypeError, ValueError):
            a = b = c = d = e = f = math.nan
        if crs == "EPSG:32651" and (a, b, d, e) == (20.0, 0.0, 0.0, -20.0) and math.isfinite(c) and math.isfinite(f):
            attrs.update(x_min=c, y_max=f)
    for key in ("group", "group_id"):
        group = text(raw[key]) if key in raw else None
        if group:
            attrs["group"] = group
            break
    return attrs


def read_chip_h5(path: Path, *, require_label: bool = True) -> dict[str, Any]:
    """Read only S2, S2_valid, label (and the georeferencing attributes) from one HDF5 chip, then validate it."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            arrays = {}
            for key, shape in (("S2", (N_BINS, N_BANDS, CHIP, CHIP)), ("S2_valid", (N_BINS, CHIP, CHIP)),
                               ("label", (CHIP, CHIP))):
                if key not in handle:
                    continue
                dataset = handle[key]
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != shape or dataset.dtype.kind not in "fiub":
                    declared = getattr(dataset, "shape", "a group")
                    raise ChipError(f"{path.name}: {key} is {declared} ({getattr(dataset, 'dtype', 'no dtype')}); "
                                    f"expected a numeric dataset of shape {shape} (checked before reading)")
                arrays[key] = dataset[...]
            attrs = georeference_attrs(dict(handle.attrs))
    except OSError as exc:
        raise ChipError(f"{path.name}: not a readable HDF5 file ({exc})") from exc
    chip = validate_chip(arrays, name=path.name, require_label=require_label)
    chip["attrs"] = attrs
    return chip


# --------------------------------------------------------------------------------------------------
# Prithvi input: quarterly frame selection, NaN accounting, normalisation
# --------------------------------------------------------------------------------------------------


def valid_fractions(valid: np.ndarray) -> np.ndarray:
    return np.asarray(valid, dtype=np.float64).reshape(N_BINS, -1).mean(axis=1)


def select_frames(valid: np.ndarray) -> dict[str, Any]:
    """The clearest bin of each calendar quarter (highest valid fraction), earliest bin on ties (study protocol)."""
    fractions = valid_fractions(valid)
    bins = []
    for quarter in range(QUARTERS):
        candidates = range(quarter * BINS_PER_QUARTER, (quarter + 1) * BINS_PER_QUARTER)
        bins.append(min(candidates, key=lambda b: (-fractions[b], b)))
    return {"bins": bins, "valid_fractions": [round(float(fractions[b]), 6) for b in bins]}


def prithvi_input(chip: Mapping[str, Any], bins: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
    """(6, T, 128, 128) standardised model input for the given bins and the fill record. Reflectance x 10 000, then the
    pinned config means/stds (the repository's own `_normalise`), then every missing value -> 0, the pretraining mean.
    Pure: the chip is not modified and repeated calls return identical arrays."""
    s2 = np.asarray(chip["S2"], dtype=np.float32)
    frames = s2[list(bins)].transpose(1, 0, 2, 3)  # (6, T, H, W) reflectance with NaN where unobserved
    standard = pp._normalise(frames[None])[0].astype(np.float32)
    missing = ~np.isfinite(standard)
    filled = np.where(missing, np.float32(0.0), standard).astype(np.float32)
    per_frame = missing.any(axis=0).reshape(len(bins), -1).mean(axis=1)
    no_observation = missing.any(axis=0).all(axis=0)
    record = {
        "bins": [int(b) for b in bins],
        "filled_fraction_per_frame": [round(float(v), 6) for v in per_frame],
        "filled_fraction": round(float(missing.mean()), 6),
        "pixels_without_any_observation": int(no_observation.sum()),
    }
    return filled, record


def single_quarter_bins(selection: Mapping[str, Any], quarter: int) -> list[int]:
    return [selection["bins"][quarter]]


# --------------------------------------------------------------------------------------------------
# fit / development split (whole regions, declared hash order, no RNG)
# --------------------------------------------------------------------------------------------------


def split_fit_dev(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assign the training chips to fit and development by whole region. Regions are ordered by
    SHA-256("rice-capstone-split:<k>:<region>") and added to development while it holds at most round(0.25 n) chips;
    the first attempt k = 0, 1, ... whose development set has at least target - 2 chips, at least two rice chips,
    and leaves both classes in fit is used. Deterministic across library versions."""
    train = sorted((r for r in rows if r["split"] == "train"), key=chip_id)
    if not train:
        raise ValueError("no training chips to split")
    target = round(DEV_FRACTION * len(train))
    regions = sorted({r["region_iso"] for r in train})
    for attempt in range(100):
        order = sorted(regions, key=lambda g: sha256_bytes(f"{SPLIT_SALT}:{attempt}:{g}".encode()))
        dev_regions: list[str] = []
        count = 0
        for region in order:
            size = sum(r["region_iso"] == region for r in train)
            if count + size <= target:
                dev_regions.append(region)
                count += size
            if count == target:
                break
        dev = [r for r in train if r["region_iso"] in dev_regions]
        fit = [r for r in train if r["region_iso"] not in dev_regions]
        rice_dev = sum(r["stratum"] == "rice" for r in dev)
        fit_strata = {r["stratum"] for r in fit}
        if len(dev) >= target - 2 and rice_dev >= MIN_DEV_RICE_CHIPS and fit_strata == {"rice", "nonrice"}:
            return {
                "rule": SPLIT_RULE,
                "attempt": attempt, "target_dev_chips": target,
                "development_regions": sorted(dev_regions), "fit_regions": sorted({r["region_iso"] for r in fit}),
                "fit": [chip_id(r) for r in fit], "development": [chip_id(r) for r in dev],
                "development_rice_chips": rice_dev,
            }
    raise ValueError("no region assignment satisfies the declared development constraints")


CHIP_METRES = CHIP * PIXEL_METRES  # 2560 m


def footprint_gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[float, bool]:
    """Ground distance in metres between two chip squares given by upper-left x_min / y_max (0 when they touch or
    overlap) and whether they share positive area."""
    dx = abs(float(a["x_min"]) - float(b["x_min"])) - CHIP_METRES
    dy = abs(float(a["y_max"]) - float(b["y_max"])) - CHIP_METRES
    return math.hypot(max(dx, 0.0), max(dy, 0.0)), dx < 0 and dy < 0


def spatial_audit(chips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit of role separation (issue #22 S2). `chips` rows: id, site, year, role and, when georeferenced, x_min and
    y_max (EPSG:32651 metres). Reports unique sites, sites seen in several years, sites or footprints shared across
    roles, and the minimum ground distance between chips of different roles. Grouping by region or site reduces
    leakage risk; it does not by itself guarantee distance, which is what this measures."""
    by_site: dict[str, set] = {}
    for c in chips:
        by_site.setdefault(str(c["site"]), set()).add(c["role"])
    years: dict[str, set] = {}
    for c in chips:
        years.setdefault(str(c["site"]), set()).add(c["year"])
    placed = [c for c in chips if c.get("x_min") is not None and c.get("y_max") is not None]
    roles = sorted({c["role"] for c in chips})
    nearest: dict[str, Any] = {}
    overlapping, touching = [], []
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if a["role"] == b["role"]:
                continue
            gap, overlap = footprint_gap(a, b)
            pair = "-".join(sorted((a["role"], b["role"])))
            if pair not in nearest or gap < nearest[pair][0]:
                nearest[pair] = (round(gap, 1), " / ".join(sorted((a["id"], b["id"]))))
            if overlap:
                overlapping.append(sorted((a["id"], b["id"])))
            elif gap == 0:
                touching.append(sorted((a["id"], b["id"])))
    return {
        "chips": len(chips), "unique_sites": len(by_site), "roles": roles,
        "sites_with_repeated_years": sum(len(v) > 1 for v in years.values()),
        "max_years_per_site": max((len(v) for v in years.values()), default=0),
        "sites_in_more_than_one_role": sorted(s for s, r in by_site.items() if len(r) > 1),
        "georeferenced_chips": len(placed),
        "cross_role_overlapping_footprints": sorted(overlapping),
        "cross_role_touching_footprints": sorted(touching),
        "min_cross_role_distance_m": {k: v[0] for k, v in sorted(nearest.items())},
        "nearest_cross_role_chips": {k: v[1] for k, v in sorted(nearest.items())},
        "meaning": "ground distance between chip squares of different roles (0 = touching or overlapping); grouping "
                   "by region or site reduces leakage risk but does not by itself guarantee distance",
    }


def check_roles(roles: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for role, ids in roles.items():
        for cid in ids:
            if cid in seen:
                raise ValueError(f"{cid} is in both {seen[cid]} and {role}")
            if cid.startswith(DEVELOPMENT_EXAMPLE_GROUP + "_"):
                raise ValueError("the development example is a training sample and must never be a scoring chip")
            seen[cid] = role


# --------------------------------------------------------------------------------------------------
# metrics (pooled over label != 255; undefined ratios are null with a reason)
# --------------------------------------------------------------------------------------------------


def confusion(probability: np.ndarray, label: np.ndarray, threshold: float) -> dict[str, int]:
    probability = np.asarray(probability, dtype=np.float64)
    label = np.asarray(label)
    if probability.shape != label.shape:
        raise ValueError(f"prediction shape {probability.shape} != label shape {label.shape}")
    if not np.isin(label, (0, 1, IGNORE)).all():
        raise ValueError("label values outside 0/1/255")
    scored = label != IGNORE
    if not np.isfinite(probability[scored]).all():
        raise ValueError("non-finite score on a scored pixel; pixels are never hidden")
    if ((probability[scored] < 0) | (probability[scored] > 1)).any():
        raise ValueError("scores must lie in [0, 1]")
    positive = probability >= threshold
    return {
        "tp": int((scored & positive & (label == 1)).sum()), "fp": int((scored & positive & (label == 0)).sum()),
        "fn": int((scored & ~positive & (label == 1)).sum()), "tn": int((scored & ~positive & (label == 0)).sum()),
    }


def scores(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    values = {
        "f1": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }
    reasons = {
        "f1": "no reference rice and no predicted rice", "iou": "no reference rice and no predicted rice",
        "precision": "no pixel predicted rice", "recall": "no reference rice pixel", "accuracy": "no scored pixel",
    }
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "scored_pixels": tp + fp + fn + tn, "reference_rice_pixels": tp + fn,
        **values, "undefined": "; ".join(f"{k}: {reasons[k]}" for k, v in values.items() if v is None),
    }


def pooled(probabilities: Sequence[np.ndarray], labels: Sequence[np.ndarray], threshold: float) -> dict[str, Any]:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for probability, label in zip(probabilities, labels, strict=True):
        for key, value in confusion(probability, label, threshold).items():
            total[key] += value
    return scores(total)


def select_threshold(probabilities: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> dict[str, Any]:
    """Highest pooled development rice F1 over THRESHOLDS; ties closest to 0.5, then the lower threshold."""
    sweep = [{"threshold": t, **pooled(probabilities, labels, t)} for t in THRESHOLDS]
    defined = [row for row in sweep if row["f1"] is not None]
    if not defined:
        raise ValueError("development F1 is undefined at every threshold (no reference rice on development chips)")
    best = min(defined, key=lambda row: (-row["f1"], abs(row["threshold"] - 0.5), row["threshold"]))
    return {"threshold": best["threshold"], "f1": best["f1"], "sweep": sweep}


def select_epoch(curve: Sequence[Mapping[str, Any]]) -> int:
    """Epoch (1-based) with the highest development F1 at 0.5; earliest on ties; undefined F1 counts as -1."""
    if not curve:
        raise ValueError("empty training curve")
    return min(curve, key=lambda row: (-(row["dev_f1"] if row["dev_f1"] is not None else -1.0), row["epoch"]))["epoch"]


# --------------------------------------------------------------------------------------------------
# baseline B: the study's M1 spectral-temporal features (70 axes), class-stratified pixel sample, fit-only fill
# --------------------------------------------------------------------------------------------------


def m1_features(chip: Mapping[str, Any]) -> np.ndarray:
    """(H*W, 70) float32: per band mean/std/min/max/p10/p50/p90 of observed values, quarterly band means, quarterly
    observed fractions. Missing statistics stay NaN here and are filled by the fit-only fill (study recipe)."""
    import warnings

    valid = np.asarray(chip["S2_valid"]).astype(bool)
    values = np.where(valid[:, None], chip["S2"], np.nan)
    blocks: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for band in range(N_BANDS):
            x = values[:, band]
            blocks += [np.nanmean(x, axis=0), np.nanstd(x, axis=0), np.nanmin(x, axis=0), np.nanmax(x, axis=0)]
            blocks += list(np.nanpercentile(x, [10, 50, 90], axis=0))
        for quarter in range(QUARTERS):
            blocks.extend(np.nanmean(values[quarter * BINS_PER_QUARTER:(quarter + 1) * BINS_PER_QUARTER], axis=0))
    blocks += [valid[q * BINS_PER_QUARTER:(q + 1) * BINS_PER_QUARTER].mean(axis=0) for q in range(QUARTERS)]
    return np.stack(blocks, axis=-1).reshape(-1, len(M1_NAMES)).astype(np.float32)


def fit_fill(features: np.ndarray) -> np.ndarray:
    finite = np.isfinite(features)
    count = finite.sum(axis=0)
    if not count.all():
        raise ValueError("a feature has no finite fit observation")
    return np.where(finite, features, 0).sum(axis=0, dtype=np.float64) / count


def fill_features(features: np.ndarray, fill: np.ndarray) -> np.ndarray:
    if features.shape[-1] != len(fill) or not np.isfinite(fill).all():
        raise ValueError("feature/fill mismatch")
    return np.where(np.isfinite(features), features, fill).astype(np.float32)


def pixel_sample(labels: Sequence[np.ndarray], cap: int, seed: int) -> list[np.ndarray]:
    """Round-robin quotas over non-empty (chip, class) strata, then seeded selection without replacement (study)."""
    if cap < 1:
        raise ValueError("cap must be positive")
    strata = [(i, c, np.flatnonzero(np.asarray(y).reshape(-1) == c)) for i, y in enumerate(labels) for c in (0, 1)]
    strata = [s for s in strata if len(s[2])]
    quotas = np.zeros(len(strata), dtype=int)
    budget = min(cap, sum(len(s[2]) for s in strata))
    while budget:
        active = [j for j, s in enumerate(strata) if quotas[j] < len(s[2])]
        step = max(1, budget // len(active))
        for j in active:
            take = min(step, len(strata[j][2]) - quotas[j], budget)
            quotas[j] += take
            budget -= take
            if not budget:
                break
    selected: list[list[int]] = [[] for _ in labels]
    rng = np.random.default_rng(seed)
    for (i, _c, pixels), n in zip(strata, quotas, strict=True):
        selected[i].extend(rng.choice(pixels, n, replace=False).tolist())
    return [np.asarray(sorted(p), dtype=np.int64) for p in selected]


# --------------------------------------------------------------------------------------------------
# head artifact (safetensors + manifest), GeoTIFF, CSV and ZIP helpers
# --------------------------------------------------------------------------------------------------


def head_shapes(in_channels: int, hidden: int) -> dict[str, tuple[int, ...]]:
    return {
        "conv1.weight": (hidden, in_channels, 1, 1), "conv1.bias": (hidden,),
        "conv2.weight": (2, hidden, 1, 1), "conv2.bias": (2,),
    }


def save_head(params: Mapping[str, np.ndarray], out_dir: Path, metadata: Mapping[str, Any]) -> Path:
    """Write adapter.safetensors (float32 head tensors only; no base weights) and a manifest with digests."""
    from safetensors.numpy import save_file

    in_channels = int(params["conv1.weight"].shape[1])
    hidden = int(params["conv1.weight"].shape[0])
    expected = head_shapes(in_channels, hidden)
    if set(params) != set(HEAD_TENSORS):
        raise ValueError(f"head tensors {sorted(params)} != {sorted(HEAD_TENSORS)}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape:
            raise ValueError(f"{name}: shape {tuple(params[name].shape)} != {shape}")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / ARTIFACT_WEIGHTS
    save_file({k: np.ascontiguousarray(params[k], dtype=np.float32) for k in HEAD_TENSORS}, str(weights),
              metadata={"format": ARTIFACT_FORMAT})
    manifest = {
        "format": ARTIFACT_FORMAT, "format_version": "1.0", "in_channels": in_channels, "hidden": hidden,
        "tensors": {k: list(v) for k, v in expected.items()},
        "files": [{"path": ARTIFACT_WEIGHTS, "bytes": weights.stat().st_size, "sha256": sha256_file(weights)}],
        **dict(metadata),
    }
    write_json(out_dir / ARTIFACT_MANIFEST, manifest)
    return out_dir


def load_head(artifact_dir: Path, *, expect: Mapping[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read and verify an exported head: format, digests, tensor names and shapes, and any `expect` manifest keys."""
    from safetensors.numpy import load_file

    manifest = read_json(artifact_dir / ARTIFACT_MANIFEST)
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    for key, value in (expect or {}).items():
        if manifest.get(key) != value:
            raise ValueError(f"artifact {key} {manifest.get(key)!r} != expected {value!r}")
    for entry in manifest["files"]:
        path = artifact_dir / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{entry['path']}: size or SHA-256 differs from the manifest")
    params = load_file(str(artifact_dir / ARTIFACT_WEIGHTS))
    expected = head_shapes(int(manifest["in_channels"]), int(manifest["hidden"]))
    if set(params) != set(expected):
        raise ValueError(f"artifact tensors {sorted(params)} != {sorted(expected)}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape or list(shape) != manifest["tensors"][name]:
            raise ValueError(f"{name}: shape mismatch against the manifest")
    return params, manifest


def head_forward_numpy(params: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """Patch-grid rice logits margin (B, h, w) of the 1x1-conv head, numpy reference (before upsampling)."""
    from math import sqrt

    w1 = params["conv1.weight"][:, :, 0, 0]
    hidden = np.einsum("oc,bchw->bohw", w1, features) + params["conv1.bias"][None, :, None, None]
    erf = np.vectorize(math.erf)
    hidden = 0.5 * hidden * (1.0 + erf(hidden / sqrt(2.0)))
    logits = np.einsum("oc,bchw->bohw", params["conv2.weight"][:, :, 0, 0], hidden) + params["conv2.bias"][None, :, None, None]
    return logits[:, 1] - logits[:, 0]


def write_geotiff(path: Path, array: np.ndarray, *, x_min: float | None, y_max: float | None,
                  nodata: int | None = IGNORE, dtype: Any = np.uint8) -> None:
    """Single-band GeoTIFF (EPSG:32651, 20 m, upper-left x_min/y_max) via tifffile GeoTIFF tags; without
    coordinates a plain TIFF is written and the manifest says so. uint8 decisions by default; float32 for scores,
    which have no nodata value (nodata=None)."""
    import tifffile

    array = np.ascontiguousarray(array, dtype=dtype)
    extratags = [] if nodata is None else [(42113, "s", 0, str(nodata), True)]  # GDAL_NODATA
    if x_min is not None and y_max is not None:
        extratags += [
            (33550, "d", 3, (PIXEL_METRES, PIXEL_METRES, 0.0), True),  # ModelPixelScale
            (33922, "d", 6, (0.0, 0.0, 0.0, float(x_min), float(y_max), 0.0), True),  # ModelTiepoint
            # GeoKeyDirectory: version 1.1.0, 3 keys: GTModelType=Projected, GTRasterType=PixelIsArea, ProjectedCRS
            (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, EPSG), True),
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array, photometric="minisblack", compression=None, extratags=extratags)


def read_geotiff(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        array = page.asarray()
        tags = {tag.code: tag.value for tag in page.tags.values()}
    geo: dict[str, Any] = {"nodata": int(tags[42113]) if 42113 in tags else None}
    if 33922 in tags:
        geo.update(x_min=float(tags[33922][3]), y_max=float(tags[33922][4]), pixel=float(tags[33550][0]),
                   epsg=int(tags[34735][-1]))
    return array, geo


def geotiff_placement(path: Path) -> dict[str, Any]:
    """Full placement record of one GeoTIFF from its tags (issue #22 m1): raster shape, pixel scale, tiepoint, the
    GeoKey directory as {key: value}, the affine transform (a, b, c, d, e, f) and bounds (left, bottom, right, top)."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        shape = tuple(int(v) for v in page.shape[:2])
        tags = {tag.code: tag.value for tag in page.tags.values()}
    if 33550 not in tags or 33922 not in tags or 34735 not in tags:
        return {"shape": shape, "georeferenced": False}
    scale, tie, keys = tuple(map(float, tags[33550])), tuple(map(float, tags[33922])), tuple(map(int, tags[34735]))
    geokeys = {keys[i]: keys[i + 3] for i in range(4, 4 + 4 * keys[3], 4) if keys[i + 1] == 0}  # inline SHORT values
    a, e = scale[0], -scale[1]
    c, f = tie[3] - tie[0] * a, tie[4] - tie[1] * e
    rows, cols = shape
    return {"shape": shape, "georeferenced": True, "pixel_scale": scale[:2], "tiepoint": tie, "geokeys": geokeys,
            "transform": (a, 0.0, c, 0.0, e, f), "bounds": (c, f + rows * e, c + cols * a, f)}


def placement_matches(placement: Mapping[str, Any], x_min: float, y_max: float) -> bool:
    """True when a GeoTIFF sits exactly on its chip: 128 x 128, projected EPSG:32651, pixel-is-area, 20 m north-up
    transform with upper-left corner (x_min, y_max) and bounds one chip (2560 m) wide."""
    return bool(
        placement.get("georeferenced") and placement["shape"] == (CHIP, CHIP)
        and placement["geokeys"].get(1024) == 1 and placement["geokeys"].get(1025) == 1
        and placement["geokeys"].get(3072) == EPSG
        and placement["transform"] == (PIXEL_METRES, 0.0, float(x_min), 0.0, -PIXEL_METRES, float(y_max))
        and placement["bounds"] == (float(x_min), float(y_max) - CHIP * PIXEL_METRES,
                                    float(x_min) + CHIP * PIXEL_METRES, float(y_max))
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(round(value, 6))
    return value


def table_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every key of every row, in first-seen order, so rows with extra columns (labelled BYOD chips) keep them."""
    return list(dict.fromkeys(key for row in rows for key in row))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    columns = list(columns or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})


def read_csv(path: Path) -> list[dict[str, Any]]:
    def parse(value: str) -> Any:
        if value == "":
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def build_zip(zip_path: Path, base: Path, members: Iterable[Path]) -> dict[str, str]:
    """Deterministic ZIP (sorted names, fixed timestamps) with a checksums.json entry; returns the member digests."""
    entries = sorted({p.resolve() for p in members if p.is_file()})
    checksums = {p.relative_to(base.resolve()).as_posix(): sha256_file(p) for p in entries}
    payload = json.dumps(checksums, indent=2, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(checksums):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (base / name).read_bytes())
        info = zipfile.ZipInfo("checksums.json", date_time=(2026, 1, 1, 0, 0, 0))
        archive.writestr(info, payload)
    return checksums


def verify_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe member name {name!r}")
        checksums = json.loads(archive.read("checksums.json"))
        missing = sorted(set(checksums) - set(names))
        mismatched = [n for n, d in checksums.items() if n in names and sha256_bytes(archive.read(n)) != d]
        extra = sorted(set(names) - set(checksums) - {"checksums.json"})
    if mismatched or extra or missing:
        raise ValueError(f"results.zip integrity failure: mismatched {mismatched}, missing {missing}, unlisted {extra}")
    return {"members": len(checksums), "verified": True}


# --------------------------------------------------------------------------------------------------
# dataset acquisition (public, anonymous; INDEX and member SHA-256 verified; expanded or zipped layouts)
# --------------------------------------------------------------------------------------------------


def locate_member(root: Path, row: Mapping[str, Any], cache: Path) -> Path:
    """Find one INDEX member in an expanded (`data-train/<path>`) or zipped (`data-train.zip`) layout, verify its
    SHA-256 against INDEX and return a local path. Zip members are read in memory and written only after verifying."""
    stem = row["archive"].removesuffix(".zip")
    for candidate in (root / stem / row["path"], root / row["path"]):
        if candidate.is_file():
            if sha256_file(candidate) != row["sha256"]:
                raise ValueError(f"{row['path']}: SHA-256 differs from INDEX.jsonl")
            return candidate
    archive_path = root / row["archive"]
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            data = archive.read(row["path"])
        if sha256_bytes(data) != row["sha256"]:
            raise ValueError(f"{row['path']} in {row['archive']}: SHA-256 differs from INDEX.jsonl")
        target = cache / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target
    raise FileNotFoundError(f"{row['path']} not found in {root} (neither expanded nor zipped)")


def read_index(root: Path) -> list[dict[str, Any]]:
    data = (root / "INDEX.jsonl").read_bytes()
    if sha256_bytes(data) != INDEX_SHA256:
        raise ValueError(f"INDEX.jsonl SHA-256 {sha256_bytes(data)} != pinned {INDEX_SHA256}; the dataset changed")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if any(r["group_id"] == DEVELOPMENT_EXAMPLE_GROUP for r in rows):
        raise ValueError("the development example must not appear in the scoring index")
    recipes = sorted({str(r.get("recipe_sha256")) for r in rows})
    if recipes != [LABEL_RECIPE_SHA256]:
        raise ValueError(f"INDEX label recipes {recipes} != pinned {LABEL_RECIPE_SHA256}; the labels changed")
    return rows


def download_dataset(cache: Path) -> Path:
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache / "kagglehub"))
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        os.environ.pop(name, None)  # anonymous: no credential is needed or used
    import kagglehub

    started = time.perf_counter()
    path = Path(kagglehub.dataset_download(f"{DATASET_SLUG}/versions/{DATASET_VERSION}"))
    print(f"dataset resolved anonymously by kagglehub in {time.perf_counter() - started:.1f} s: {path}", flush=True)
    return path


# --------------------------------------------------------------------------------------------------
# stage context
# --------------------------------------------------------------------------------------------------


class Context:
    def __init__(self, root: Path):
        self.root = root
        self.out = root / "outputs"
        self.figures = self.out / "figures"
        self.work = root / "work"
        for path in (self.out, self.figures, self.work):
            path.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()

    def records(self) -> dict[str, Any]:
        return read_json(self.work / "records.json")

    def chip(self, cid: str, *, require_label: bool = True) -> dict[str, Any]:
        info = self.records()["chips"][cid]
        return read_chip_h5(Path(info["path"]), require_label=require_label)

    def receipt(self, stage: str, **extra: Any) -> None:
        record = {"stage": stage, "seconds": round(time.perf_counter() - self.started, 2), **extra}
        try:
            import torch

            if torch.cuda.is_available():
                record["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        except ImportError:
            pass
        write_json(self.work / "receipts" / f"{stage}.json", record)
        print(json.dumps(record), flush=True)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rgb(chip: Mapping[str, Any], b: int, bands: Sequence[int]) -> np.ndarray:
    image = np.stack([chip["S2"][b, i] for i in bands], axis=-1)
    finite = np.isfinite(image)
    hi = np.nanpercentile(image, 98) if finite.any() else 1.0
    return np.clip(np.where(finite, image, 0) / max(hi, 1e-6), 0, 1)


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_prepare(ctx: Context) -> None:
    manifest = read_json(ctx.root / "data_manifest.json")
    if manifest["index_sha256"] != INDEX_SHA256 or manifest["slug"] != DATASET_SLUG:
        raise ValueError("carried data manifest disagrees with the stage runner's pins")
    root = download_dataset(ctx.root / "cache")
    rows = read_index(root)
    layout = "expanded" if (root / "data-train").is_dir() else "zipped"
    package = {}
    if (root / "PACKAGE.sha256").is_file():
        for line in (root / "PACKAGE.sha256").read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                package[name.strip().lstrip("*")] = digest
        if package.get("INDEX.jsonl") != INDEX_SHA256:
            raise ValueError("PACKAGE.sha256 does not list the pinned INDEX.jsonl digest")
    if layout == "zipped":
        for archive in ("data-train.zip", "data-validation.zip"):
            if sha256_file(root / archive) != manifest["package_sha256"][archive]:
                raise ValueError(f"{archive}: SHA-256 differs from the pinned PACKAGE.sha256")
    split = split_fit_dev(rows)
    validation = sorted(chip_id(r) for r in rows if r["split"] == "validation")
    roles = {"fit": split["fit"], "development": split["development"], "validation": validation}
    check_roles(roles)
    role_of = {cid: role for role, ids in roles.items() for cid in ids}
    audit = spatial_audit([{"id": chip_id(r), "site": r["group_id"], "year": r["year"], "role": role_of[chip_id(r)],
                            "x_min": r["x_min"], "y_max": r["y_max"]} for r in rows])
    chips: dict[str, Any] = {}
    table = []
    valid_counts = np.zeros(N_BINS)
    for row in sorted(rows, key=chip_id):
        cid = chip_id(row)
        path = locate_member(root, row, ctx.work / "chips")
        chip = read_chip_h5(path)
        frames = select_frames(chip["S2_valid"])
        _x, fill = prithvi_input(chip, frames["bins"])
        label = chip["label"]
        valid_counts += chip["S2_valid"].reshape(N_BINS, -1).sum(axis=1)
        chips[cid] = {
            "path": str(path), "sha256": row["sha256"], "role": role_of[cid], "region": row["region_iso"],
            "year": row["year"], "stratum": row["stratum"], "x_min": row["x_min"], "y_max": row["y_max"],
            "frames": frames, "fill": fill,
        }
        table.append({
            "chip": cid, "role": role_of[cid], "region": row["region_iso"], "year": row["year"],
            "rice_pixels": int((label == 1).sum()), "nonrice_pixels": int((label == 0).sum()),
            "ignored_pixels": int((label == IGNORE).sum()), "selected_bins": " ".join(map(str, frames["bins"])),
            "min_selected_valid_fraction": min(frames["valid_fractions"]), "filled_fraction": fill["filled_fraction"],
        })
    for role in ("development", "validation"):
        if not any(t["rice_pixels"] for t in table if t["role"] == role):
            raise ValueError(f"no reference rice pixel in the {role} chips; the comparison cannot be scored")
    probes = _refusal_probes(chips[roles["fit"][0]]["path"])  # validation must refuse bad chips before anything is written
    write_json(ctx.work / "records.json", {"chips": chips, "roles": roles, "dataset_root": str(root), "layout": layout})
    write_csv(ctx.out / "chips.csv", table)
    write_json(ctx.out / "preprocessing.json", {
        "frame_rule": "clearest bin (highest valid fraction) in each calendar quarter, earliest bin on ties",
        "normalisation": "reflectance x 10000, then (x - mean) / std with the pinned Prithvi config statistics",
        "means": list(pp.MEANS), "stds": list(pp.STDS), "missing_value_fill": 0.0,
        "missing_fill_meaning": "0 after standardisation = the pretraining mean; pixels with no valid observation "
        "in any selected frame stay scored",
        "resampling": "none: native 20 m Sentinel-2 grid (Prithvi-EO-2.0 was pretrained on 30 m HLS)",
        "chips": {cid: {"role": c["role"], **c["frames"], **c["fill"]} for cid, c in chips.items()},
    })
    write_json(ctx.out / "data_manifest.json", {
        **manifest, "layout": layout, "chips_verified": len(chips), "index_rows": len(rows),
        "roles": roles, "split": split,
        "development_example": f"{DEVELOPMENT_EXAMPLE_GROUP} (a parent training sample) is never read",
        "package_sha256_downloaded": package,
    })
    write_json(ctx.out / "spatial_audit.json", audit)  # issue #22 S2: measured separation, not assumed
    _prepare_figures(ctx, chips, table, valid_counts)
    write_json(ctx.out / "prepare.json", {
        "layout": layout, "chips": len(chips), "roles": {k: len(v) for k, v in roles.items()},
        "development_regions": split["development_regions"], "fit_regions": split["fit_regions"],
        "split_attempt": split["attempt"], "refusal_probes": probes, "table": table,
    })
    ctx.receipt("prepare", chips=len(chips), layout=layout)


def _with_label_two(label: np.ndarray) -> np.ndarray:
    """A copy of `label` with one labelled pixel set to 2, so the probe is invalid whether or not the chip holds rice."""
    bad = np.array(label, dtype=np.uint8, copy=True)
    labelled = np.flatnonzero(bad != 255)
    bad.flat[labelled[0] if labelled.size else 0] = 2
    return bad


def _refusal_probes(path: str) -> list[dict[str, str]]:
    chip = read_chip_h5(Path(path))
    base = {"S2": chip["S2"], "S2_valid": chip["S2_valid"], "label": chip["label"]}
    cases = {
        "12 bins instead of 24": {**base, "S2": base["S2"][:12]},
        "label value 2": {**base, "label": _with_label_two(base["label"])},
        "reflectance x 10 000": {**base, "S2": np.nan_to_num(base["S2"]) * 10000.0},
    }
    results = []
    for case, arrays in cases.items():
        try:
            validate_chip(arrays, name="probe")
        except ChipError as exc:
            results.append({"probe": case, "refused": "yes", "message": str(exc)})
        else:
            raise RuntimeError(f"validation accepted an invalid chip: {case}")
    return results


def _prepare_figures(ctx: Context, chips: Mapping[str, Any], table: Sequence[Mapping[str, Any]], counts: np.ndarray) -> None:
    plt = _plt()
    fit_rice = [t for t in table if t["role"] == "fit" and t["rice_pixels"]]
    show = max(fit_rice, key=lambda t: (t["rice_pixels"], t["chip"]))["chip"]
    chip = read_chip_h5(Path(chips[show]["path"]))
    best = chips[show]["frames"]["bins"][int(np.argmax(chips[show]["frames"]["valid_fractions"]))]
    label = chip["label"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].imshow(_rgb(chip, best, (2, 1, 0)))
    axes[0].set_title(f"True colour (B4/B3/B2), bin {best}")
    axes[1].imshow(_rgb(chip, best, (4, 3, 2)))
    axes[1].set_title("False colour (B11/B8A/B4)")
    overlay = np.zeros((*label.shape, 4))
    overlay[label == 1] = (1.0, 0.85, 0.0, 0.75)
    overlay[label == IGNORE] = (0.6, 0.6, 0.6, 0.6)
    axes[2].imshow(_rgb(chip, best, (2, 1, 0)))
    axes[2].imshow(overlay)
    axes[2].set_title("Label: rice (yellow), ignore (grey)")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"{show} ({chips[show]['region']}, fit chip; 128 x 128 px = 2.56 km)")
    fig.tight_layout()
    fig.savefig(ctx.figures / "overview_composites.png", dpi=110)
    plt.close(fig)

    nir, red = chip["S2"][:, 3], chip["S2"][:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    rice_px = np.argwhere(label == 1)
    non_px = np.argwhere(label == 0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for pixels, name, colour in ((rice_px, "rice pixel", "#c08a00"), (non_px, "non-rice pixel", "#2a7a2a")):
        if len(pixels):
            r, c = pixels[len(pixels) // 2]
            ax.plot(range(N_BINS), ndvi[:, r, c], marker="o", label=f"{name} (row {r}, col {c})", color=colour)
    for q in range(1, QUARTERS):
        ax.axvline(q * BINS_PER_QUARTER - 0.5, color="grey", lw=0.8, ls=":")
    ax.set(xlabel="Calendar bin (24 per year; quarters dotted)", ylabel="NDVI", title=f"NDVI through the year, {show}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ctx.figures / "overview_ndvi_series.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(N_BINS), counts / (len(chips) * CHIP * CHIP), color="#4a6fa5")
    ax.set(xlabel="Calendar bin", ylabel="Valid fraction (all 40 chips)", ylim=(0, 1),
           title="How often each bin has a cloud-free observation")
    fig.tight_layout()
    fig.savefig(ctx.figures / "overview_valid_by_bin.png", dpi=110)
    plt.close(fig)


def _labels(ctx: Context, ids: Sequence[str]) -> list[np.ndarray]:
    return [ctx.chip(cid)["label"] for cid in ids]


def stage_baseline(ctx: Context) -> None:
    import lightgbm as lgb

    roles = ctx.records()["roles"]
    fit_chips = [ctx.chip(cid) for cid in roles["fit"]]
    dev_chips = [ctx.chip(cid) for cid in roles["development"]]
    fit_labels = [c["label"] for c in fit_chips]
    dev_labels = [c["label"] for c in dev_chips]
    picks = pixel_sample(fit_labels, PIXEL_CAP, PIXEL_SEED)
    raw = np.concatenate([m1_features(c)[p] for c, p in zip(fit_chips, picks, strict=True)])
    y = np.concatenate([c["label"].reshape(-1)[p] for c, p in zip(fit_chips, picks, strict=True)]).astype(np.float32)
    fill = fit_fill(raw)
    x = fill_features(raw, fill)
    dev_features = [fill_features(m1_features(c), fill) for c in dev_chips]
    keep = [lab.reshape(-1) != IGNORE for lab in dev_labels]
    xd = np.concatenate([f[k] for f, k in zip(dev_features, keep, strict=True)])
    yd = np.concatenate([lab.reshape(-1)[k] for lab, k in zip(dev_labels, keep, strict=True)]).astype(np.float32)

    def dev_f1(probability, dataset):
        score = pooled([probability], [dataset.get_label().astype(np.uint8)], 0.5)["f1"]
        return "dev_rice_f1_at_0p5", score if score is not None else 0.0, True

    started = time.perf_counter()
    booster = lgb.train(LGBM_PARAMS, lgb.Dataset(x, label=y, feature_name=M1_NAMES), num_boost_round=LGBM_ROUND_CAP,
                        valid_sets=[lgb.Dataset(xd, label=yd, feature_name=M1_NAMES)], valid_names=["development"],
                        feval=dev_f1, callbacks=[lgb.early_stopping(LGBM_PATIENCE, verbose=False)])
    seconds = time.perf_counter() - started
    booster.save_model(str(ctx.work / "baseline_b.txt"), num_iteration=booster.best_iteration)
    np.save(ctx.work / "baseline_b_fill.npy", fill)
    dev_probs = [booster.predict(f, num_iteration=booster.best_iteration).reshape(CHIP, CHIP) for f in dev_features]
    zero = [np.zeros((CHIP, CHIP)) for _ in dev_labels]
    b_sel = select_threshold(dev_probs, dev_labels)
    selection = {
        "protocol": "every choice is made on development chips only and frozen before validation scoring",
        "A": {"system": "all non-rice", "threshold": None},
        "B": {"system": "M1 spectral-temporal LightGBM", "threshold": b_sel["threshold"], "dev_f1": b_sel["f1"],
              "best_iteration": int(booster.best_iteration), "params": LGBM_PARAMS, "pixel_cap": PIXEL_CAP,
              "pixel_seed": PIXEL_SEED, "fit_pixels": int(len(y)), "fit_rice_pixels": int(y.sum()),
              "training_seconds": round(seconds, 2), "sweep": [_sweep_row(r) for r in b_sel["sweep"]]},
    }
    write_json(ctx.out / "selection.json", selection)
    rows = [{"system": "A all-non-rice", "threshold": None, **pooled(zero, dev_labels, 0.5)},
            {"system": "B LightGBM", "threshold": b_sel["threshold"], **pooled(dev_probs, dev_labels, b_sel["threshold"])}]
    write_csv(ctx.out / "development_metrics.csv", rows, METRIC_COLUMNS)
    ctx.receipt("baseline", best_iteration=int(booster.best_iteration))


METRIC_COLUMNS = ["system", "threshold", "f1", "iou", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn",
                  "scored_pixels", "reference_rice_pixels", "undefined"]
CHIP_COLUMNS = ["chip", "region", "system", "threshold", *METRIC_COLUMNS[2:]]


def _sweep_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("threshold", "f1", "iou", "precision", "recall")}


# ---- model stages (torch is imported only here) ------------------------------------------------------


def _torch_setup(deterministic: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(SEED)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def load_prithvi(ctx: Context) -> Any:
    reports: list[Any] = []
    weights = ctx.root / "weights" / pp.MODEL_KEY
    pipe = pp.PrithviFeaturePipeline.from_pretrained(weights_dir=weights, allow_download=True, report=reports.append)
    record = {
        "model_id": pp.MODEL_ID, "revision": pp.MODEL_REVISION, "source_checkpoint_sha256": pp.SOURCE_CKPT_SHA256,
        "converted_sha256": pp.CONVERTED_SHA256, "pickle_audit_sha256": pp.PICKLE_AUDIT_SHA256,
        "parameters": pp.PARAMETER_COUNT, "strict_load": True, "device": pipe.device, "source": pipe.source,
        "feature_layer_zero_based": FEATURE_LAYER, "conversion": reports,
    }
    write_json(ctx.out / "model.json", record)
    return pipe


def encode(pipe: Any, batch: np.ndarray) -> np.ndarray:
    """(B, 6, T, H, W) standardised input -> (B, 1024, H/16, W/16) temporal mean of block-12 patch tokens."""
    import torch

    b, _c, t, h, w = batch.shape
    with torch.inference_mode():
        features = pipe.model.encoder.forward_features(torch.from_numpy(np.ascontiguousarray(batch)).to(pipe.device))
    tokens = features[FEATURE_LAYER][:, 1:].float()
    expected = t * (h // PATCH) * (w // PATCH)
    if tokens.shape[1] != expected:
        raise RuntimeError(f"encoder returned {tokens.shape[1]} patch tokens, expected {expected} (1 + T*64 with CLS)")
    if not torch.isfinite(tokens).all():
        raise RuntimeError("non-finite encoder features")
    grid = tokens.reshape(b, t, h // PATCH, w // PATCH, EMBED_DIM).mean(dim=1).permute(0, 3, 1, 2)
    return grid.contiguous().cpu().numpy()


def stage_features(ctx: Context) -> None:
    torch = _torch_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; select a T4 GPU runtime")
    torch.cuda.reset_peak_memory_stats()
    pipe = load_prithvi(ctx)
    records = ctx.records()
    started = time.perf_counter()
    t4: dict[str, np.ndarray] = {}
    t1: dict[int, dict[str, np.ndarray]] = {q: {} for q in range(QUARTERS)}
    token_checks = {}
    for cid, info in sorted(records["chips"].items()):
        chip = ctx.chip(cid)
        x, _fill = prithvi_input(chip, info["frames"]["bins"])
        t4[cid] = encode(pipe, x[None])[0]
        token_checks[cid] = list(t4[cid].shape)
        if info["role"] in ("fit", "development"):
            for q in range(QUARTERS):
                xq, _ = prithvi_input(chip, single_quarter_bins(info["frames"], q))
                t1[q][cid] = encode(pipe, xq[None])[0]
    np.savez(ctx.work / "features_t4.npz", **t4)
    for q in range(QUARTERS):
        np.savez(ctx.work / f"features_t1_q{q + 1}.npz", **t1[q])
    seconds = time.perf_counter() - started
    write_json(ctx.out / "features.json", {
        "feature": "block 12 (zero-based 11) patch tokens, temporal mean, grid 8 x 8 x 1024 per 128 px chip",
        "precision": "float32, inference mode, deterministic cuDNN", "chips_t4": len(t4),
        "chips_t1_per_quarter": len(t1[0]), "seconds": round(seconds, 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "grid_shapes": sorted({tuple(v) for v in token_checks.values()}),
    })
    ctx.receipt("features")


HEAD_OUTPUT_ORDER = ("both logit channels bilinearly upsampled from 8 x 8 to 128 x 128 px (align_corners=False), then "
                     "softmax over the two channels; the rice score is channel 1")


def _head_module(torch: Any, in_channels: int = EMBED_DIM, hidden: int = HEAD_HIDDEN):
    nn = torch.nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, hidden, 1)
            self.act = nn.GELU()
            self.conv2 = nn.Conv2d(hidden, 2, 1)

        def forward(self, x):
            logits = self.conv2(self.act(self.conv1(x)))
            return torch.nn.functional.interpolate(logits, size=(CHIP, CHIP), mode="bilinear", align_corners=False)

    return Head()


def head_probabilities(torch: Any, head: Any, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    head.eval()
    with torch.inference_mode():
        logits = head(torch.from_numpy(np.ascontiguousarray(features)).float().to(device))
        return torch.softmax(logits, dim=1)[:, 1].double().cpu().numpy()


def train_head(torch: Any, fit_x: np.ndarray, fit_y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray,
               epochs: int, *, select_best: bool = True,
               log: Callable[[str], None] = print) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Train the head on cached features (CPU, seed 0) and return (state, development curve). With select_best the
    state is the epoch chosen by `select_epoch`; without it, the state after exactly `epochs` epochs."""
    torch.manual_seed(SEED)
    head = _head_module(torch, fit_x.shape[1])
    optimiser = torch.optim.AdamW(head.parameters(), lr=HEAD_CONFIG["learning_rate"], weight_decay=HEAD_CONFIG["weight_decay"])
    xs = torch.from_numpy(fit_x).float()
    ys = torch.from_numpy(fit_y.astype(np.int64))
    generator = torch.Generator().manual_seed(SEED)
    curve: list[dict[str, Any]] = []
    states: dict[int, dict[str, np.ndarray]] = {}
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(xs), generator=generator)
        losses = []
        for start in range(0, len(xs), HEAD_CONFIG["batch_size"]):
            idx = order[start:start + HEAD_CONFIG["batch_size"]]
            loss = torch.nn.functional.cross_entropy(head(xs[idx]), ys[idx], ignore_index=IGNORE)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
        probs = head_probabilities(torch, head, dev_x)
        dev = pooled(list(probs), list(dev_y), 0.5)
        with torch.inference_mode():
            dev_loss = float(torch.nn.functional.cross_entropy(head(torch.from_numpy(dev_x).float()),
                                                               torch.from_numpy(dev_y.astype(np.int64)), ignore_index=IGNORE))
        curve.append({"epoch": epoch, "fit_loss": float(np.mean(losses)), "dev_loss": dev_loss, "dev_f1": dev["f1"]})
        states[epoch] = {k: v.detach().cpu().numpy().copy() for k, v in head.state_dict().items()}
        log(f"epoch {epoch:2d}  fit loss {curve[-1]['fit_loss']:.4f}  dev loss {dev_loss:.4f}  dev F1@0.5 {dev['f1']}")
    return states[select_epoch(curve) if select_best else epochs], curve


def _stack(cache: Mapping[str, np.ndarray], ids: Sequence[str]) -> np.ndarray:
    return np.stack([cache[cid] for cid in ids]).astype(np.float32)


def _load_head(torch: Any, state: Mapping[str, np.ndarray]):
    head = _head_module(torch, int(state["conv1.weight"].shape[1]), int(state["conv1.weight"].shape[0]))
    head.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()}, strict=True)
    return head


def stage_train_head(ctx: Context) -> None:
    torch = _torch_setup()
    roles = ctx.records()["roles"]
    cache = dict(np.load(ctx.work / "features_t4.npz"))
    fit_y = np.stack(_labels(ctx, roles["fit"]))
    dev_y = np.stack(_labels(ctx, roles["development"]))
    state, curve = train_head(torch, _stack(cache, roles["fit"]), fit_y, _stack(cache, roles["development"]), dev_y,
                              HEAD_CONFIG["max_epochs"])
    epoch = select_epoch(curve)
    np.savez(ctx.work / "head_c.npz", **state)
    probs = head_probabilities(torch, _load_head(torch, state), _stack(cache, roles["development"]))
    c_sel = select_threshold(list(probs), list(dev_y))
    selection = read_json(ctx.out / "selection.json")
    selection["C"] = {"system": "frozen Prithvi-EO-2.0 block 12 + 1x1-conv head", "threshold": c_sel["threshold"],
                      "dev_f1": c_sel["f1"], "selected_epoch": epoch, "config": HEAD_CONFIG, "frames": 4,
                      "sweep": [_sweep_row(r) for r in c_sel["sweep"]]}
    write_json(ctx.out / "selection.json", selection)
    write_csv(ctx.out / "head_curve.csv", curve)
    rows = read_csv(ctx.out / "development_metrics.csv")
    rows = [r for r in rows if not str(r["system"]).startswith("C ")]
    rows.append({"system": "C frozen Prithvi + head", "threshold": c_sel["threshold"],
                 **pooled(list(probs), list(dev_y), c_sel["threshold"])})
    write_csv(ctx.out / "development_metrics.csv", rows, METRIC_COLUMNS)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot([r["epoch"] for r in curve], [r["fit_loss"] for r in curve], label="fit loss")
    axes[0].plot([r["epoch"] for r in curve], [r["dev_loss"] for r in curve], label="development loss")
    axes[0].axvline(epoch, color="black", ls="--", lw=0.8)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot([r["epoch"] for r in curve], [r["dev_f1"] or 0 for r in curve], color="#c08a00")
    axes[1].axvline(epoch, color="black", ls="--", lw=0.8)
    axes[1].set(xlabel="Epoch", ylabel="Development rice F1 at 0.5", title=f"selected epoch {epoch}")
    fig.tight_layout()
    fig.savefig(ctx.figures / "head_training_curves.png", dpi=110)
    plt.close(fig)
    ctx.receipt("train-head", selected_epoch=epoch)


ACTIVITY_SCOPE = ("fixed-recipe, development-only: every variant reuses the epoch count and threshold selected for "
                  "T=4, so it is not an equally tuned comparison, and it does not isolate seasonal progression from "
                  "observation coverage or from settings that suit T=4")


def stage_activity(ctx: Context) -> None:
    torch = _torch_setup()
    roles = ctx.records()["roles"]
    selection = read_json(ctx.out / "selection.json")
    epochs = int(selection["C"]["selected_epoch"])
    threshold = float(selection["C"]["threshold"])
    fit_y = np.stack(_labels(ctx, roles["fit"]))
    dev_y = np.stack(_labels(ctx, roles["development"]))
    rows = []
    variants = [("T=4 (one frame per quarter)", "features_t4.npz", "T=4")]
    variants += [(f"T=1 (quarter {q + 1} only)", f"features_t1_q{q + 1}.npz", f"T=1, Q{q + 1}") for q in range(QUARTERS)]
    maps = {}
    for name, filename, _short in variants:
        cache = dict(np.load(ctx.work / filename))
        # one head per frame count with C's recipe, seed and selected epoch count; only the input frames differ
        # (the T=4 row reproduces C exactly). Kurt, 2026-10-11: this measures one frame's information, not C's robustness.
        # Issue #22 M1: the T=1 heads borrow T=4's epoch count and threshold, so this is a fixed-recipe comparison, not
        # an equally tuned one, and it does not isolate seasonal progression from observation coverage.
        state, _curve = train_head(torch, _stack(cache, roles["fit"]), fit_y, _stack(cache, roles["development"]), dev_y,
                                   epochs, select_best=False, log=lambda _m: None)
        probs = head_probabilities(torch, _load_head(torch, state), _stack(cache, roles["development"]))
        maps[name] = probs
        rows.append({"variant": name, "epochs": epochs, "threshold": threshold,
                     **pooled(list(probs), list(dev_y), threshold)})
    single = [r for r in rows if r["variant"].startswith("T=1")]
    best_single = max(single, key=lambda r: (r["f1"] if r["f1"] is not None else -1.0))
    write_csv(ctx.out / "activity.csv", rows, ["variant", "epochs", "threshold", *METRIC_COLUMNS[2:]])
    selection["activity"] = {"question": "frames T=1 (each single quarter) vs T=4, development chips only",
                             "comparison": ACTIVITY_SCOPE,
                             "best_single_quarter": best_single["variant"], "canonical_frames": 4,
                             "decision": "canonical validation keeps T=4 (declared before the activity)"}
    write_json(ctx.out / "selection.json", selection)
    plt = _plt()
    dev_ids = roles["development"]
    show = max(range(len(dev_ids)), key=lambda i: int((dev_y[i] == 1).sum()))
    fig, axes = plt.subplots(1, len(variants) + 1, figsize=(3.1 * (len(variants) + 1), 3.4))
    axes[0].imshow(np.where(dev_y[show] == IGNORE, 0.5, dev_y[show]), cmap="YlOrBr", vmin=0, vmax=1)
    axes[0].set_title("reference")
    for ax, (name, _f, short) in zip(axes[1:], variants, strict=True):
        ax.imshow(maps[name][show], cmap="YlOrBr", vmin=0, vmax=1)
        ax.set_title(short, fontsize=9)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"Rice score on development chip {dev_ids[show]}")
    fig.tight_layout()
    fig.savefig(ctx.figures / "activity_frames.png", dpi=110)
    plt.close(fig)
    ctx.receipt("activity")


def stage_finetune(ctx: Context) -> None:
    """Opt-in extension (D2): fine-tune patch_embed + blocks 1-12 with a fresh head under the same fit/dev protocol."""
    if (ctx.work / "frozen.json").exists():
        raise RuntimeError("validation scoring already happened; a fine-tune now could not be selected before it "
                           "and would be exploratory only. Run all from a fresh runtime with RUN_FINETUNE = True.")
    torch = _torch_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; select a T4 GPU runtime")
    torch.cuda.reset_peak_memory_stats()
    pipe = load_prithvi(ctx)
    encoder = pipe.model.encoder
    trainable = 0
    for name, param in encoder.named_parameters():
        parts = name.split(".")
        on = parts[0] == "patch_embed" or name == "cls_token" or (parts[0] == "blocks" and int(parts[1]) <= FEATURE_LAYER)
        param.requires_grad_(on)
        trainable += param.numel() if on else 0
    records = ctx.records()
    roles = records["roles"]

    def inputs(ids):
        return np.stack([prithvi_input(ctx.chip(cid), records["chips"][cid]["frames"]["bins"])[0] for cid in ids])

    fit_x, dev_x, val_x = inputs(roles["fit"]), inputs(roles["development"]), inputs(roles["validation"])
    fit_y = np.stack(_labels(ctx, roles["fit"]))
    dev_y = np.stack(_labels(ctx, roles["development"]))
    head = _head_module(torch).to(pipe.device)
    params = [p for p in encoder.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW([{"params": params, "lr": FT_CONFIG["encoder_learning_rate"]},
                                   {"params": head.parameters(), "lr": FT_CONFIG["head_learning_rate"]}],
                                  weight_decay=FT_CONFIG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    def forward(batch):
        x = torch.from_numpy(batch).to(pipe.device)
        b, _c, t, h, w = x.shape
        tokens = encoder.forward_features(x)[FEATURE_LAYER][:, 1:]
        grid = tokens.float().reshape(b, t, h // PATCH, w // PATCH, EMBED_DIM).mean(1).permute(0, 3, 1, 2)
        return head(grid)

    def predict(x_all):
        encoder.eval()
        head.eval()
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(len(x_all)):
                out.append(torch.softmax(forward(x_all[i:i + 1]).float(), 1)[:, 1].double().cpu().numpy()[0])
        return np.stack(out)

    curve, best = [], None
    started = time.perf_counter()
    for epoch in range(1, FT_CONFIG["epochs"] + 1):
        encoder.train()
        head.train()
        order = torch.randperm(len(fit_x), generator=generator).numpy()
        losses = []
        for start in range(0, len(order), FT_CONFIG["batch_size"]):
            idx = order[start:start + FT_CONFIG["batch_size"]]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = forward(fit_x[idx])
            target = torch.from_numpy(fit_y[idx].astype(np.int64)).to(pipe.device)
            loss = torch.nn.functional.cross_entropy(logits.float(), target, ignore_index=IGNORE)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            losses.append(float(loss))
        dev_probs = predict(dev_x)
        f1 = pooled(list(dev_probs), list(dev_y), 0.5)["f1"]
        curve.append({"epoch": epoch, "fit_loss": float(np.mean(losses)), "dev_f1": f1})
        print(f"C-FT epoch {epoch}: fit loss {curve[-1]['fit_loss']:.4f}, dev F1@0.5 {f1}", flush=True)
        if select_epoch(curve) == epoch:
            best = {"epoch": epoch, "dev_probs": dev_probs, "val_probs": predict(val_x)}
    assert best is not None
    sel = select_threshold(list(best["dev_probs"]), list(dev_y))
    np.savez(ctx.work / "cft_validation_probs.npz", **dict(zip(roles["validation"], best["val_probs"], strict=True)))
    selection = read_json(ctx.out / "selection.json")
    selection["C-FT"] = {"system": "fine-tuned Prithvi blocks 1-12 + head (opt-in)", "selected_before_validation": True,
                         "threshold": sel["threshold"], "dev_f1": sel["f1"], "selected_epoch": best["epoch"],
                         "config": FT_CONFIG, "trainable_encoder_parameters": trainable,
                         "validation_probabilities": "computed label-free at the selected epoch and sealed until scoring"}
    write_json(ctx.out / "selection.json", selection)
    write_csv(ctx.out / "finetune_curve.csv", curve)
    write_json(ctx.out / "finetune.json", {"curve": curve, "seconds": round(time.perf_counter() - started, 2),
                                           "peak_gpu_gib_full_adamw_step": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
    ctx.receipt("finetune", selected_epoch=best["epoch"])


def drop_stale_finetune(ctx: Context) -> bool:
    """Remove the C-FT selection entry and its sealed predictions unless the finetune stage's receipt is current (a
    later upstream restart invalidates it, and a fine-tune that failed after writing files never wrote one). Returns
    whether anything was dropped. Called before the freeze, so stale C-FT results can never be scored."""
    if (ctx.work / "receipts" / "finetune.json").exists():
        return False
    selection_path = ctx.out / "selection.json"
    selection = read_json(selection_path)
    stale = selection.pop("C-FT", None) is not None
    if stale:
        write_json(selection_path, selection)
        print("C-FT dropped: its fine-tune receipt is not current (an upstream stage re-ran or the fine-tune failed)")
    for stale_file in (ctx.work / "cft_validation_probs.npz", ctx.out / "finetune.json", ctx.out / "finetune_curve.csv"):
        stale_file.unlink(missing_ok=True)  # results.zip bundles everything under outputs/, so nothing stale may remain
    return stale


SUPPORT_COLUMNS = ["system", "observed_frames", "threshold", "f1", "precision", "recall", "iou", "scored_pixels",
                   "reference_rice_pixels", "tp", "fp", "fn", "tn", "undefined"]


def observed_frame_counts(valid: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """(H, W) uint8: in how many of C's selected frames each pixel has a valid observation (0 to len(bins)). Where it
    is below len(bins), C's input there is the filled pretraining mean, with no channel telling the head so."""
    return np.asarray(valid, dtype=np.uint8)[list(bins)].sum(axis=0).astype(np.uint8)


def support_rows(system: str, probabilities: Sequence[np.ndarray], labels: Sequence[np.ndarray],
                 counts: Sequence[np.ndarray], threshold: float, frames: int = QUARTERS) -> list[dict[str, Any]]:
    """Secondary diagnostic (issue #22 S3): pooled metrics of one system split by the number of observed selected
    frames. The groups partition the same scored pixels as the primary table; nothing is removed from it."""
    rows = []
    for k in range(frames + 1):
        masked = [np.where(np.asarray(c) == k, lab, IGNORE).astype(np.uint8) for c, lab in zip(counts, labels, strict=True)]
        if not any((m != IGNORE).any() for m in masked):
            continue
        rows.append({"system": system, "observed_frames": k, "threshold": threshold,
                     **pooled(probabilities, masked, threshold)})
    return rows


def stage_evaluate(ctx: Context) -> None:
    import lightgbm as lgb

    torch = _torch_setup()
    if (ctx.work / "frozen.json").exists():
        raise RuntimeError("validation was already scored in this run directory; scoring happens exactly once")
    drop_stale_finetune(ctx)
    selection_path = ctx.out / "selection.json"
    selection = read_json(selection_path)
    for key in ("B", "C", "activity"):
        if key not in selection:
            raise RuntimeError(f"selection.json has no {key} record; run the development stages first")
    write_json(ctx.work / "frozen.json", {"selection_sha256": sha256_file(selection_path), "frozen_at": time.time()})
    records = ctx.records()
    ids = records["roles"]["validation"]
    chips = [ctx.chip(cid) for cid in ids]
    labels = [c["label"] for c in chips]
    booster = lgb.Booster(model_file=str(ctx.work / "baseline_b.txt"))
    fill = np.load(ctx.work / "baseline_b_fill.npy")
    probs = {
        "A all-non-rice": [np.zeros((CHIP, CHIP)) for _ in ids],
        "B LightGBM": [booster.predict(fill_features(m1_features(c), fill)).reshape(CHIP, CHIP) for c in chips],
    }
    cache = dict(np.load(ctx.work / "features_t4.npz"))
    state = dict(np.load(ctx.work / "head_c.npz"))
    probs["C frozen Prithvi + head"] = list(head_probabilities(torch, _load_head(torch, state), _stack(cache, ids)))
    thresholds = {"A all-non-rice": 0.5, "B LightGBM": selection["B"]["threshold"],
                  "C frozen Prithvi + head": selection["C"]["threshold"]}
    if selection.get("C-FT", {}).get("selected_before_validation"):
        sealed = dict(np.load(ctx.work / "cft_validation_probs.npz"))  # current: drop_stale_finetune ran before the freeze
        probs["C-FT fine-tuned (opt-in)"] = [sealed[cid] for cid in ids]
        thresholds["C-FT fine-tuned (opt-in)"] = selection["C-FT"]["threshold"]
    rows, chip_rows = [], []
    for system, values in probs.items():
        t = thresholds[system]
        rows.append({"system": system, "threshold": None if system.startswith("A") else t, **pooled(values, labels, t)})
        for cid, p, lab in zip(ids, values, labels, strict=True):
            chip_rows.append({"chip": cid, "region": records["chips"][cid]["region"], "system": system,
                              "threshold": None if system.startswith("A") else t, **scores(confusion(p, lab, t))})
    write_csv(ctx.out / "metrics.csv", rows, METRIC_COLUMNS)
    write_csv(ctx.out / "per_chip_metrics.csv", chip_rows, CHIP_COLUMNS)
    np.savez(ctx.work / "validation_probs_c.npz", **dict(zip(ids, probs["C frozen Prithvi + head"], strict=True)))
    np.savez(ctx.work / "validation_probs_b.npz", **dict(zip(ids, probs["B LightGBM"], strict=True)))
    counts = [observed_frame_counts(c["S2_valid"], records["chips"][cid]["frames"]["bins"]) for cid, c in
              zip(ids, chips, strict=True)]
    np.savez(ctx.work / "validation_observed_frames.npz", **dict(zip(ids, counts, strict=True)))
    support = [row for system, values in probs.items() if not system.startswith("A")
               for row in support_rows(system, values, labels, counts, thresholds[system])]
    write_csv(ctx.out / "observation_support.csv", support, SUPPORT_COLUMNS)
    _evaluation_figures(ctx, ids, chips, probs, thresholds)
    ctx.receipt("evaluate", systems=list(probs))


def _evaluation_figures(ctx: Context, ids: Sequence[str], chips: Sequence[Mapping[str, Any]],
                        probs: Mapping[str, Sequence[np.ndarray]], thresholds: Mapping[str, float]) -> None:
    plt = _plt()
    from matplotlib.colors import ListedColormap

    colours = ListedColormap(["#f2f2f2", "#d62728", "#1f77b4", "#2ca02c", "#bdbdbd"])  # TN FP FN TP ignore

    def errors(p, lab, t):
        pred = p >= t
        out = np.zeros(lab.shape, dtype=np.uint8)
        out[(lab == 0) & pred] = 1
        out[(lab == 1) & ~pred] = 2
        out[(lab == 1) & pred] = 3
        out[lab == IGNORE] = 4
        return out

    b_key, c_key = "B LightGBM", "C frozen Prithvi + head"
    fig, axes = plt.subplots(len(ids), 4, figsize=(11, 2.7 * len(ids)))
    for i, (cid, chip) in enumerate(zip(ids, chips, strict=True)):
        lab = chip["label"]
        bins = select_frames(chip["S2_valid"])["bins"]
        axes[i, 0].imshow(_rgb(chip, bins[int(np.argmax([chip["S2_valid"][b].mean() for b in bins]))], (2, 1, 0)))
        axes[i, 0].set_title(f"{cid} true colour", fontsize=8)
        axes[i, 1].imshow(errors(probs[b_key][i], lab, thresholds[b_key]), cmap=colours, vmin=0, vmax=4)
        axes[i, 1].set_title("B errors", fontsize=8)
        axes[i, 2].imshow(errors(probs[c_key][i], lab, thresholds[c_key]), cmap=colours, vmin=0, vmax=4)
        axes[i, 2].set_title("C errors", fontsize=8)
        disagree = (probs[b_key][i] >= thresholds[b_key]) != (probs[c_key][i] >= thresholds[c_key])
        axes[i, 3].imshow(disagree, cmap="Purples", vmin=0, vmax=1)
        axes[i, 3].set_title(f"B vs C disagree ({disagree.mean():.1%})", fontsize=8)
        for ax in axes[i]:
            ax.set_axis_off()
    fig.suptitle("Validation errors: green TP, red FP, blue FN, light TN, grey ignored", y=1.0)
    fig.tight_layout()
    fig.savefig(ctx.figures / "validation_error_maps.png", dpi=90)
    plt.close(fig)

    rice_counts = [int((c["label"] == 1).sum()) for c in chips]
    i = int(np.argmax(rice_counts))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(np.where(chips[i]["label"] == IGNORE, 0.5, chips[i]["label"]), cmap="YlOrBr", vmin=0, vmax=1)
    axes[0].set_title("reference (20 m)")
    axes[1].imshow(probs[c_key][i], cmap="YlOrBr", vmin=0, vmax=1)
    axes[1].set_title("C rice score (from an 8 x 8 grid)")
    axes[2].imshow(probs[b_key][i], cmap="YlOrBr", vmin=0, vmax=1)
    axes[2].set_title("B rice score (per pixel)")
    for ax in axes[:2]:
        for k in range(1, CHIP // PATCH):
            ax.axhline(k * PATCH - 0.5, color="white", lw=0.4)
            ax.axvline(k * PATCH - 0.5, color="white", lw=0.4)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"{ids[i]}: each 16 x 16 px patch = 320 m; the head upsamples a coarse grid")
    fig.tight_layout()
    fig.savefig(ctx.figures / "validation_patch_effect.png", dpi=110)
    plt.close(fig)


def _head_metadata(selection: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        "base_model": pp.MODEL_ID, "base_revision": pp.MODEL_REVISION, "base_converted_sha256": pp.CONVERTED_SHA256,
        "feature_layer_zero_based": FEATURE_LAYER, "frames": "clearest bin per calendar quarter, earliest on ties (T=4)",
        "normalisation": {"scale": pp.REFLECTANCE_SCALE, "means": list(pp.MEANS), "stds": list(pp.STDS),
                          "missing_fill": 0.0},
        "pooling": "temporal mean of patch tokens", "output": HEAD_OUTPUT_ORDER,
        "threshold": selection["C"]["threshold"], "decision": "rice if score >= threshold",
        "classes": {"0": "non-rice", "1": "rice"}, "training": selection["C"]["config"],
        "selected_epoch": selection["C"]["selected_epoch"], "selection_basis": role,
        "contains_base_weights": False, "training_data_embedded": False,
    }


def stage_export(ctx: Context) -> None:
    selection = read_json(ctx.out / "selection.json")
    frozen = read_json(ctx.work / "frozen.json")
    if sha256_file(ctx.out / "selection.json") != frozen["selection_sha256"]:
        raise RuntimeError("selection.json changed after validation scoring")
    state = dict(np.load(ctx.work / "head_c.npz"))
    params = {k: state[k] for k in HEAD_TENSORS}
    save_head(params, ctx.out / "head_adapter", _head_metadata(selection, "development chips (fit/dev protocol)"))
    records = ctx.records()
    probs = dict(np.load(ctx.work / "validation_probs_c.npz"))
    support = dict(np.load(ctx.work / "validation_observed_frames.npz"))
    t = selection["C"]["threshold"]
    for cid, p in probs.items():
        label = ctx.chip(cid)["label"]
        decision = (p >= t).astype(np.uint8)
        info = records["chips"][cid]
        where = {"x_min": info["x_min"], "y_max": info["y_max"]}
        # Scored map: pixels the reference ignores are 255, so it reproduces the evaluation exactly.
        write_geotiff(ctx.out / "predictions" / f"{cid}_C.tif", np.where(label == IGNORE, IGNORE, decision), **where)
        # Usable maps: a decision for every pixel, and the rice scores themselves, so a user can choose a threshold.
        write_geotiff(ctx.out / "predictions" / f"{cid}_C_all_pixels.tif", decision, **where)
        write_geotiff(ctx.out / "predictions" / f"{cid}_C_score.tif", p, nodata=None, dtype=np.float32, **where)
        # Observation support (issue #22 S3): how many of C's four selected frames observed each pixel (0 = all filled).
        write_geotiff(ctx.out / "predictions" / f"{cid}_observed_frames.tif", support[cid], nodata=None, **where)
    ctx.receipt("export", geotiffs=4 * len(probs))


def stage_reload(ctx: Context) -> None:
    """Fresh process: verify the exported head from files, rebuild the base model, recompute validation features from
    the HDF5 chips, and compare with the evaluation's predictions; reopen the GeoTIFFs and recompute the metrics."""
    torch = _torch_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; select a T4 GPU runtime")
    params, manifest = load_head(ctx.out / "head_adapter", expect={"base_revision": pp.MODEL_REVISION,
                                                                   "feature_layer_zero_based": FEATURE_LAYER})
    pipe = load_prithvi(ctx)
    head = _load_head(torch, params)
    records = ctx.records()
    saved = dict(np.load(ctx.work / "validation_probs_c.npz"))
    t = float(manifest["threshold"])
    worst, identical, geotiff_ok, extra_ok, support_ok, placed_ok = 0.0, True, True, True, True, True
    labels, from_tiff = [], []
    for cid in records["roles"]["validation"]:
        chip = ctx.chip(cid)
        x, _ = prithvi_input(chip, select_frames(chip["S2_valid"])["bins"])
        p = head_probabilities(torch, head, encode(pipe, x[None]))[0]
        worst = max(worst, float(np.abs(p - saved[cid]).max()))
        identical &= bool(np.array_equal(p >= t, saved[cid] >= t))
        array, geo = read_geotiff(ctx.out / "predictions" / f"{cid}_C.tif")
        expected = np.where(chip["label"] == IGNORE, IGNORE, (saved[cid] >= t).astype(np.uint8))
        geotiff_ok &= bool(np.array_equal(array, expected)) and geo.get("epsg") == EPSG
        everywhere, geo_all = read_geotiff(ctx.out / "predictions" / f"{cid}_C_all_pixels.tif")
        score, geo_score = read_geotiff(ctx.out / "predictions" / f"{cid}_C_score.tif")
        extra_ok &= (bool(np.array_equal(everywhere, (saved[cid] >= t).astype(np.uint8)))
                     and bool(np.array_equal(score, saved[cid].astype(np.float32)))
                     and geo_all.get("epsg") == EPSG and geo_score.get("epsg") == EPSG)
        observed, geo_obs = read_geotiff(ctx.out / "predictions" / f"{cid}_observed_frames.tif")
        support_ok &= (bool(np.array_equal(observed, observed_frame_counts(chip["S2_valid"], select_frames(
            chip["S2_valid"])["bins"]))) and geo_obs.get("epsg") == EPSG)
        info = records["chips"][cid]
        for suffix in ("_C", "_C_all_pixels", "_C_score", "_observed_frames"):  # m1: origin, size, transform, bounds
            placed_ok &= placement_matches(geotiff_placement(ctx.out / "predictions" / f"{cid}{suffix}.tif"),
                                           info["x_min"], info["y_max"])
        labels.append(chip["label"])
        from_tiff.append(np.where(array == IGNORE, 0, array).astype(np.float64))
    recomputed = pooled(from_tiff, labels, 0.5)
    reported = next(r for r in read_csv(ctx.out / "metrics.csv") if r["system"] == "C frozen Prithvi + head")
    metrics_ok = all(recomputed[k] == reported[k] for k in ("tp", "fp", "fn", "tn"))
    result = {
        "artifact_digests_verified": True, "base_model_strict_load": True,
        "max_abs_score_difference": worst, "tolerance": RELOAD_TOLERANCE,
        "binary_decisions_identical": identical, "geotiffs_match_and_georeferenced": geotiff_ok,
        "metrics_recomputed_from_geotiffs": metrics_ok,
        "score_and_all_pixel_geotiffs_match": extra_ok,
        "observation_support_geotiffs_match": support_ok,
        "geotiff_placement_verified": placed_ok,
        "meaning": "the saved computation is reproducible from files in a fresh process; not evidence of accuracy",
    }
    write_json(ctx.out / "verification.json", result)
    if not (worst <= RELOAD_TOLERANCE and identical and geotiff_ok and metrics_ok and extra_ok and support_ok and placed_ok):
        raise RuntimeError(f"reload parity failed: {result}")
    ctx.receipt("reload")


def stage_report(ctx: Context) -> None:
    selection = read_json(ctx.out / "selection.json")
    metrics = read_csv(ctx.out / "metrics.csv")
    receipts = {p.stem: read_json(p) for p in sorted((ctx.work / "receipts").glob("*.json"))}
    source = read_json(ctx.root / "source.json") if (ctx.root / "source.json").exists() else {}
    (ctx.out / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")

    lines = answer_lines(metrics)
    lines += ["", "# Conclusion scaffold", "", "Complete in your own words:", "",
              "> On the labelled pixels of eight validation chips from two regions, [system] reached a pooled rice F1 of "
              "[value] against [baselines]. On the development chips, under the recipe, epoch count and threshold selected "
              "for four frames, [T=1 vs T=4 result, without claiming a cause]. The error maps show "
              "[pattern], and the 320 m patch grid [effect]. Labels are published map products, the sample is small "
              "and spatially clustered, and Prithvi's pretraining overlap with these places is unknown, so [bounded "
              "claim, naming how the pipelines differ beyond pretraining]. Next we would [specific evidence].", ""]
    (ctx.out / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(ctx.out / "run_summary.json", {
        "notebook": source, "dataset": {"slug": DATASET_SLUG, "version": DATASET_VERSION, "index_sha256": INDEX_SHA256},
        "model": {"id": pp.MODEL_ID, "revision": pp.MODEL_REVISION}, "python": sys.version.split()[0],
        "stages": receipts, "selection_thresholds": {k: v.get("threshold") for k, v in selection.items() if isinstance(v, dict)},
        "evidence_boundary": "teaching scale: 32 training and 8 validation chips; no national or operational claim",
    })
    check = bundle_results(ctx)
    ctx.receipt("report", zip_members=check["members"])
    # The bundle must show its own reporting stage: add the report receipt to the summary and bundle again.
    summary_path = ctx.out / "run_summary.json"
    summary = read_json(summary_path)
    summary["stages"]["report"] = read_json(ctx.work / "receipts" / "report.json")
    write_json(summary_path, summary)
    final = bundle_results(ctx)
    if final["members"] != check["members"]:
        raise RuntimeError(f"results.zip membership changed while finalising: {check} != {final}")


def _fmt(value: Any) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def answer_lines(metrics: Sequence[Mapping[str, Any]]) -> list[str]:
    """This run's own answer to the driving question, written from the validation table it just produced (KND4):
    the pooled metrics, whether C scored above B, and the evidence level that limits the claim."""
    rows = {r["system"].split(" ", 1)[0]: r for r in metrics}
    b, c = rows["B"], rows["C"]
    lines = ["# This run's answer", "", "Validation metrics (8 chips, PH-07/PH-08, pooled over labelled pixels):", ""]
    lines += [f"- {r['system']}: F1 {_fmt(r['f1'])}, IoU {_fmt(r['iou'])}, precision {_fmt(r['precision'])}, "
              f"recall {_fmt(r['recall'])}" for r in metrics]
    better = c["f1"] is not None and (b["f1"] is None or c["f1"] > b["f1"])
    verdict = (f"the frozen-Prithvi pipeline (C, F1 {_fmt(c['f1'])}) "
               f"{'reached a higher pooled rice F1 than' if better else 'did not reach a higher pooled rice F1 than'} "
               f"the spectral-temporal LightGBM pipeline (B, F1 {_fmt(b['f1'])})")
    lines += ["", "**Answer to the driving question, from this run:** on the labelled pixels of these 8 validation "
                  f"chips, {verdict}; calling everything non-rice (A) scored F1 {_fmt(rows['A']['f1'])}."]
    if "C-FT" in rows:
        lines += ["", f"The opt-in fine-tuned encoder (C-FT) scored F1 {_fmt(rows['C-FT']['f1'])}; it was selected on "
                      "development chips before validation was scored."]
    lines += ["", "**What this compares (issue #22 S1):** two whole pipelines. B and C differ in temporal information "
                  "(all 24 bins versus four quarterly frames), missing-data treatment (observed-fraction features and a "
                  "fit-only fill versus mean-filled standardised inputs without a validity channel), class sampling (a "
                  "class-stratified pixel sample versus every labelled pixel of the fit chips) and spatial prediction "
                  "architecture (per-pixel trees versus an 8 x 8 logit grid bilinearly upsampled to 128 x 128), so this "
                  "comparison does not isolate the contribution of foundation-model pretraining. The scores measure "
                  "agreement with a published reference map, not independently established field accuracy."]
    lines += ["", "**Three separate claims (S4):** an F1 above the all-non-rice floor shows that a system finds some "
                  "reference-rice signal; a higher pooled F1 than the other pipeline is a comparison under the declared "
                  "metric; neither shows that a map meets operational requirements, which this capstone does not "
                  "assess."]
    lines += ["", "**Evidence level:** tutorial evidence from one run at teaching scale: 8 validation chips from two "
                  "neighbouring regions, scored once after every choice was frozen on development chips; map-derived "
                  "labels; no confidence intervals. It does not establish which approach is better nationally or "
                  "operationally."]
    return lines


def bundle_results(ctx: Context) -> dict[str, Any]:
    """Build results.zip so that the verification.json inside it is the record the notebook displays: a first build
    measures the bundle, its check is written into verification.json, and the bundle is rebuilt and verified again
    with that final record inside. The two checks must agree."""
    def build() -> dict[str, Any]:
        byod = ctx.out / "byod"
        members = [p for p in ctx.out.rglob("*") if p.is_file() and p.name not in ("results.zip", "byod_results.zip")
                   and byod not in p.parents]  # BYOD results have their own bundle; the canonical one is the sample run
        build_zip(ctx.out / "results.zip", ctx.out, members)
        return verify_zip(ctx.out / "results.zip")

    verification_path = ctx.out / "verification.json"
    verification = read_json(verification_path)
    verification["results_zip"] = build()
    write_json(verification_path, verification)
    final = build()
    if final != verification["results_zip"]:
        raise RuntimeError(f"results.zip changed between builds: {verification['results_zip']} != {final}")
    with zipfile.ZipFile(ctx.out / "results.zip") as archive:
        if json.loads(archive.read("verification.json")) != verification:
            raise RuntimeError("the bundled verification.json differs from the displayed record")
    return final


# ---- BYOD -------------------------------------------------------------------------------------------


class ReloadParityError(RuntimeError):
    """The adapted BYOD head did not reproduce its own scores after being saved and reloaded. A software or environment
    fault, not a problem with the learner's files, so it is deliberately not a ChipError: it is never printed as an
    input refusal (`Refused:`, exit 3) and it stops the stage before any success receipt or result bundle exists."""


HEAD_SERIALIZATION_SCOPE = (
    "head-serialization parity: the adapted head's weights are saved to files, read back and verified in the same "
    "process, then re-applied to the cached encoder features of the test chips and compared with the scores before "
    "saving. It does not rebuild the base model, recompute features or start a fresh process, so it is not the "
    "canonical fresh-process, input-to-prediction reload of Section 10"
)


def head_serialization_parity(before: np.ndarray, after: np.ndarray, threshold: float,
                              tolerance: float = RELOAD_TOLERANCE) -> dict[str, Any]:
    """Compare the adapted head's test scores before saving with the scores of the head read back from its files.
    Passing needs a finite maximum absolute difference within `tolerance` and identical binary decisions at
    `threshold`. Returns the record; `require_head_serialization_parity` raises when it fails."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.size == 0:
        raise ReloadParityError(f"head-serialization parity: score shapes differ ({before.shape} vs {after.shape})")
    with np.errstate(invalid="ignore"):
        difference = float(np.abs(after - before).max())
    finite = bool(np.isfinite(before).all() and np.isfinite(after).all() and math.isfinite(difference))
    identical = bool(np.array_equal(before >= threshold, after >= threshold))
    return {
        "check": HEAD_SERIALIZATION_SCOPE, "max_abs_difference": difference if finite else None,
        "finite": finite, "tolerance": tolerance, "threshold": threshold, "binary_decisions_identical": identical,
        "passed": finite and difference <= tolerance and identical,
    }


def require_head_serialization_parity(record: Mapping[str, Any]) -> None:
    if record["passed"]:
        return
    reasons = []
    if not record["finite"]:
        reasons.append("a score is not finite")
    elif record["max_abs_difference"] > record["tolerance"]:
        reasons.append(f"max |score difference| {record['max_abs_difference']:.3g} exceeds {record['tolerance']}")
    if not record["binary_decisions_identical"]:
        reasons.append(f"rice/non-rice decisions at threshold {record['threshold']} differ")
    raise ReloadParityError(
        "BYOD adaptation failed verification (head-serialization parity): " + "; ".join(reasons) + ". The adapted "
        "head was discarded and no BYOD result bundle was written. This is a software or environment fault, not a "
        "problem with your files: keep the stage log (byod.log in the run directory) and report it; running the "
        "cell again starts from a clean BYOD folder."
    )


def has_labelled_pixels(label: np.ndarray | None) -> bool:
    """True when a label exists and at least one pixel is 0 or 1; an all-255 mask can only be predicted, not adapted on
    (a training batch of ignored targets alone would make the mean cross-entropy NaN)."""
    return label is not None and bool((np.asarray(label) != IGNORE).any())


def byod_group(stem: str, attrs: Mapping[str, Any]) -> str:
    """A chip's site: the HDF5 attribute `group` when present, else the file name without a trailing _YYYY year, so
    the years of one place never straddle fit and test."""
    import re

    return str(attrs["group"]) if "group" in attrs else re.sub(r"_(19|20)\d\d$", "", stem)


def chip_content_digest(chip: Mapping[str, Any]) -> str:
    """SHA-256 of a chip's S2 (NaN-normalised), S2_valid and label arrays: equal for renamed copies of one chip."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.nan_to_num(np.asarray(chip["S2"], dtype=np.float32), nan=-9999.0)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(chip["S2_valid"], dtype=np.uint8)).tobytes())
    if chip.get("label") is not None:
        digest.update(np.ascontiguousarray(np.asarray(chip["label"], dtype=np.uint8)).tobytes())
    return digest.hexdigest()


def byod_grouping_check(ids: Sequence[str], chips: Mapping[str, Mapping[str, Any]],
                        groups: Mapping[str, str]) -> dict[str, Any]:
    """Issue #22 S2: how trustworthy the BYOD site grouping is. Records which site ids came from an HDF5 attribute and
    which fell back to the file name (unverified), finds chips with identical content (renamed duplicates) and
    georeferenced chips whose footprints overlap on the ground although they carry different site ids. Either of the
    last two would let one place sit in fit and test at once, so adaptation is not run when they occur."""
    by_digest: dict[str, list[str]] = {}
    for cid in ids:
        by_digest.setdefault(chip_content_digest(chips[cid]), []).append(cid)
    placed = [cid for cid in ids if "x_min" in chips[cid]["attrs"] and "y_max" in chips[cid]["attrs"]]
    conflicts = []
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if groups[a] != groups[b] and footprint_gap(chips[a]["attrs"], chips[b]["attrs"])[1]:
                conflicts.append(sorted((a, b)))
    return {
        "site_ids_from_attribute": sorted(c for c in ids if "group" in chips[c]["attrs"]),
        "site_ids_from_file_name_unverified": sorted(c for c in ids if "group" not in chips[c]["attrs"]),
        "duplicate_content": sorted(sorted(v) for v in by_digest.values() if len(v) > 1),
        "overlapping_footprints_with_different_sites": sorted(conflicts),
        "georeferenced": len(placed),
    }


def byod_split(ids: Sequence[str], groups: Mapping[str, str], rice: Mapping[str, bool],
               nonrice: Mapping[str, bool]) -> dict[str, list[str]]:
    """Fit / development / test by whole group (SHA-256 order of 'rice-capstone-byod:<k>:<group>', about 60/20/20);
    the first k giving development and test at least one rice pixel and fit both rice and non-rice pixels is used.
    `rice` / `nonrice` say whether a chip has any labelled pixel of that class: the head trains per pixel, so a fit set
    of chips that all contain some rice is fine as long as their labels also hold non-rice."""
    names = sorted({groups[i] for i in ids})
    if len(names) < 3:
        raise ChipError("adaptation needs at least three groups (sites) so that fit, development and test differ")
    for attempt in range(200):
        order = sorted(names, key=lambda g: sha256_bytes(f"rice-capstone-byod:{attempt}:{g}".encode()))
        n_dev = max(1, round(0.2 * len(order)))
        roles = {"development": order[:n_dev], "test": order[n_dev:2 * n_dev], "fit": order[2 * n_dev:]}
        split = {role: sorted(i for i in ids if groups[i] in members) for role, members in roles.items()}
        if (split["fit"] and any(rice[i] for i in split["development"]) and any(rice[i] for i in split["test"])
                and any(rice[i] for i in split["fit"]) and any(nonrice[i] for i in split["fit"])):
            return split
    raise ChipError("no group assignment gives rice in fit, development and test and non-rice in fit; "
                    "add more labelled sites")


def collect_byod(sources: Sequence[Path], target: Path) -> list[Path]:
    """Copy the .h5 chips named by `sources` (folders, .zip archives or .h5 files) into an emptied `target`. The folder
    is emptied first, so even a rejected retry keeps nothing from an earlier upload; unsafe or oversized archives and
    two chips with the same file name are refused before anything is copied."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    plan: dict[str, tuple[Path, str | None]] = {}
    total = 0

    def add(name: str, origin: Path, member: str | None, size: int) -> None:
        nonlocal total
        if name in plan:
            first = plan[name][0] if plan[name][1] is None else f"{plan[name][0]}:{plan[name][1]}"
            second = origin if member is None else f"{origin}:{member}"
            raise ChipError(f"two chips share the file name {name}: {first} and {second}; "
                            "rename one so each chip-year has a unique name")
        plan[name] = (origin, member)
        total += size

    for source in sources:
        if source.is_dir():
            for path in sorted(source.rglob("*.h5")):
                add(path.name, path, None, path.stat().st_size)
        elif source.suffix == ".zip":
            unreadable = ChipError(f"{source.name}: not a readable .zip archive (truncated, corrupted or misnamed); "
                                   "re-create it or send the .h5 files directly")
            if not zipfile.is_zipfile(source):
                raise unreadable
            try:
                with zipfile.ZipFile(source) as archive:
                    members = archive.infolist()
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
                raise unreadable from exc
            for info in members:
                if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                    raise ChipError(f"{source.name}: unsafe member name {info.filename!r} (absolute or '..'); refused")
                if info.filename.endswith(".h5"):
                    add(Path(info.filename).name, source, info.filename, info.file_size)
        elif source.suffix == ".h5":
            add(source.name, source, None, source.stat().st_size)
        else:
            raise ChipError(f"{source}: expected a folder, a .zip or .h5 files")
        if len(plan) > BYOD_MAX_FILES or total > BYOD_MAX_BYTES:
            raise ChipError(f"the input exceeds {BYOD_MAX_FILES} chips or {BYOD_MAX_BYTES // 2**30} GiB; send fewer files")
    for name, (origin, member) in plan.items():
        if member is None:
            shutil.copy2(origin, target / name)
        else:
            try:
                with zipfile.ZipFile(origin) as archive, archive.open(member) as source, \
                        (target / name).open("wb") as sink:
                    copied = 0
                    while chunk := source.read(1 << 20):  # bounded memory; the declared size is not trusted
                        copied += len(chunk)
                        if copied > BYOD_MAX_BYTES:
                            raise ChipError(f"{origin.name}:{member} expands beyond {BYOD_MAX_BYTES // 2**30} GiB; "
                                            "refused")
                        sink.write(chunk)
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
                if isinstance(exc, ChipError):
                    raise
                raise ChipError(f"{origin.name}:{member} could not be extracted ({type(exc).__name__}); "
                                "re-create the archive or send the .h5 files directly") from exc
    return sorted(target.glob("*.h5"))


def stage_byod(ctx: Context, sources: Sequence[Path]) -> None:
    out = ctx.out / "byod"
    # Clear the previous attempt first: a rejected upload or a failed model load must never leave older predictions,
    # an older adapter or an older byod_results.zip looking as if they belonged to this attempt.
    if out.exists():
        shutil.rmtree(out)
    (ctx.out / "byod_results.zip").unlink(missing_ok=True)
    torch = _torch_setup()
    input_dir = ctx.root / "byod_input"
    files = collect_byod(sources, input_dir)
    if not files:
        raise ChipError(f"no .h5 chips found under {input_dir}; see the schema in the BYOD section")
    if len(files) > BYOD_MAX_FILES:
        raise ChipError(f"{len(files)} chips exceed the limit of {BYOD_MAX_FILES}")
    chips, problems = {}, []
    for path in files:
        try:
            chips[path.stem] = read_chip_h5(path, require_label=False)
        except ChipError as exc:
            problems.append(str(exc))
    if problems:
        raise ChipError(f"{len(problems)} of {len(files)} file(s) failed validation:\n  " + "\n  ".join(problems))
    selection = read_json(ctx.out / "selection.json")
    params, manifest = load_head(ctx.out / "head_adapter", expect={"base_revision": pp.MODEL_REVISION})
    pipe = load_prithvi(ctx)
    features, canonical, rows = {}, {}, []
    for cid, chip in chips.items():
        frames = select_frames(chip["S2_valid"])
        x, fill = prithvi_input(chip, frames["bins"])
        features[cid] = encode(pipe, x[None])[0]
        p = head_probabilities(torch, _load_head(torch, params), features[cid][None])[0]
        canonical[cid] = p
        pred = (p >= manifest["threshold"]).astype(np.uint8)
        if has_labelled_pixels(chip["label"]):  # an all-255 mask is prediction-only: keep every predicted pixel
            pred = np.where(chip["label"] == IGNORE, IGNORE, pred).astype(np.uint8)
        write_geotiff(out / "predictions" / f"{cid}_C.tif", pred, x_min=chip["attrs"].get("x_min"),
                      y_max=chip["attrs"].get("y_max"))
        row = {"chip": cid, "bins": " ".join(map(str, frames["bins"])), "filled_fraction": fill["filled_fraction"],
               "predicted_rice_fraction": float((p >= manifest["threshold"]).mean()),
               "georeferenced": "x_min" in chip["attrs"] and "y_max" in chip["attrs"]}
        if not row["georeferenced"]:
            print(f"{cid}: not georeferenced (no x_min/y_max, and no affine that is EPSG:32651, 20 m and north-up); "
                  "its GeoTIFF has no location", flush=True)
        if chip["label"] is not None:
            row.update(scores(confusion(p, chip["label"], manifest["threshold"])))
        rows.append(row)
    write_csv(out / "byod_canonical_head.csv", rows, table_columns(rows))
    labelled = [cid for cid, c in chips.items() if has_labelled_pixels(c["label"])]
    summary: dict[str, Any] = {"chips": len(chips), "labelled": len(labelled), "canonical_threshold": manifest["threshold"],
                               "data_location": "processed inside this runtime only; nothing is uploaded elsewhere"}
    if len(labelled) >= BYOD_MIN_ADAPT_CHIPS:
        groups = {cid: byod_group(cid, chips[cid]["attrs"]) for cid in labelled}
        rice = {cid: bool((chips[cid]["label"] == 1).any()) for cid in labelled}
        nonrice = {cid: bool((chips[cid]["label"] == 0).any()) for cid in labelled}
        grouping = byod_grouping_check(labelled, chips, groups)
        summary["grouping"] = grouping
        if grouping["site_ids_from_file_name_unverified"]:
            print(f"{len(grouping['site_ids_from_file_name_unverified'])} labelled chip(s) have no group_id/group "
                  "attribute: their site is taken from the file name, which is not verified. Add group_id so that "
                  "every year of one place gets the same site.", flush=True)
        blocked = []
        if grouping["duplicate_content"]:
            blocked.append(f"chips with identical content (renamed copies?): {grouping['duplicate_content']}")
        if grouping["overlapping_footprints_with_different_sites"]:
            blocked.append("chips that overlap on the ground carry different site ids: "
                           f"{grouping['overlapping_footprints_with_different_sites']}; give them the same group_id")
        try:
            if blocked:
                raise ChipError("; ".join(blocked))
            split = byod_split(labelled, groups, rice, nonrice)
        except ChipError as exc:
            summary["adaptation"] = f"not run: {exc}"
            print(f"Adaptation not run: {exc}", flush=True)
        else:
            ys = {cid: chips[cid]["label"] for cid in labelled}
            fx = np.stack([features[c] for c in split["fit"]])
            dx = np.stack([features[c] for c in split["development"]])
            state, curve = train_head(torch, fx, np.stack([ys[c] for c in split["fit"]]), dx,
                                      np.stack([ys[c] for c in split["development"]]), HEAD_CONFIG["max_epochs"])
            head = _load_head(torch, state)
            dev_probs = head_probabilities(torch, head, dx)
            sel = select_threshold(list(dev_probs), [ys[c] for c in split["development"]])
            test_probs = head_probabilities(torch, head, np.stack([features[c] for c in split["test"]]))
            test_labels = [ys[c] for c in split["test"]]
            meta = dict(_head_metadata({"C": {**selection["C"], "threshold": sel["threshold"],
                                              "selected_epoch": select_epoch(curve)}}, "BYOD development chips"))
            save_head({k: state[k] for k in HEAD_TENSORS}, out / "byod_head_adapter", meta)
            reloaded, _m = load_head(out / "byod_head_adapter")
            again = head_probabilities(torch, _load_head(torch, reloaded), np.stack([features[c] for c in split["test"]]))
            parity = head_serialization_parity(test_probs, again, sel["threshold"])
            if not parity["passed"]:  # issue #22 M3: enforced before any summary, bundle or receipt is written
                shutil.rmtree(out / "byod_head_adapter")
                write_json(out / "byod_verification_failure.json", parity)
                require_head_serialization_parity(parity)
            summary["adaptation"] = {
                "split": split, "selected_epoch": select_epoch(curve), "threshold": sel["threshold"],
                "spatial_audit": spatial_audit([
                    {"id": c, "site": groups[c], "year": c.rsplit("_", 1)[-1], "role": role, **{k: chips[c]["attrs"][k] for k in
                                                                         ("x_min", "y_max") if k in chips[c]["attrs"]}}
                    for role, ids in split.items() for c in ids]),
                "test_all_non_rice": pooled([np.zeros_like(p) for p in test_probs], test_labels, 0.5),
                "test_canonical_head": pooled([canonical[c] for c in split["test"]], test_labels,
                                              manifest["threshold"]),
                "test_adapted_head": pooled(list(test_probs), test_labels, sel["threshold"]),
                "head_serialization_parity": parity,
            }
    else:
        summary["adaptation"] = f"not run: needs at least {BYOD_MIN_ADAPT_CHIPS} labelled chips"
    write_json(out / "byod_summary.json", summary)
    bundle = ctx.out / "byod_results.zip"
    build_zip(bundle, out, out.rglob("*"))
    verify_zip(bundle)
    ctx.receipt("byod", chips=len(chips))


STAGES: dict[str, Callable[[Context], None]] = {
    "prepare": stage_prepare, "baseline": stage_baseline, "features": stage_features, "train-head": stage_train_head,
    "activity": stage_activity, "finetune": stage_finetune, "evaluate": stage_evaluate, "export": stage_export,
    "reload": stage_reload, "report": stage_report,
}


# Stages whose receipt must exist before a stage may start. Starting a stage deletes its own receipt and the receipt
# of every stage that depends on it, directly or transitively, so a failed or repeated upstream stage can never leave
# downstream work looking current (prepare, the root, therefore clears every receipt).
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "baseline": ("prepare",), "features": ("prepare",), "train-head": ("baseline", "features"),
    "activity": ("train-head",), "finetune": ("train-head",), "evaluate": ("baseline", "train-head", "activity"),
    "export": ("evaluate",), "reload": ("export",), "report": ("reload",), "byod": ("reload",),
}


# Outputs each finishing stage writes under outputs/. They are removed when the stage (or one it depends on) starts,
# so a failed re-run of export, reload or report can never leave an earlier adapter, prediction set, verification
# record or results bundle beside, or mixed with, newly written files.
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "export": ("head_adapter", "predictions"),
    "reload": ("verification.json",),
    "report": ("results.zip", "run_summary.json", "conclusion.md"),
    "byod": ("byod", "byod_results.zip"),
}


def ancestors(stage: str) -> set[str]:
    """Every stage that `stage` needs, directly or through another stage."""
    found: set[str] = set()
    frontier = [stage]
    while frontier:
        for name in PREREQUISITES.get(frontier.pop(), ()):
            if name not in found:
                found.add(name)
                frontier.append(name)
    return found


def dependents(stage: str) -> set[str]:
    """Every stage that needs `stage`, directly or through another stage."""
    found: set[str] = set()
    frontier = [stage]
    while frontier:
        current = frontier.pop()
        for name, needs in PREREQUISITES.items():
            if current in needs and name not in found:
                found.add(name)
                frontier.append(name)
    return found


def start_stage(ctx: Context, stage: str) -> None:
    """Refuse to run `stage` until its prerequisite stages have completed, then clear its own receipt and those of
    all its dependents, which stay ineligible until the chain completes again."""
    receipts = ctx.work / "receipts"
    if stage in {"evaluate", "finetune"} | ancestors("evaluate") and (ctx.work / "frozen.json").exists():
        # Refuse before touching any receipt or output: a restart of scoring or anything upstream of it would leave the
        # previous run's results.zip, metrics, predictions and adapter looking current if it then failed.
        raise RuntimeError(f"{stage}: validation was already scored in this run directory; scoring happens exactly "
                           "once. Start a fresh runtime to change a development choice")
    missing = [name for name in PREREQUISITES.get(stage, ()) if not (receipts / f"{name}.json").exists()]
    if missing:
        raise RuntimeError(f"{stage} needs the {', '.join(missing)} stage(s) to complete first; run the cells in order")
    for name in {stage} | dependents(stage):
        (receipts / f"{name}.json").unlink(missing_ok=True)
        for relative in STAGE_OUTPUTS.get(name, ()):  # a failed re-run must not leave the previous outputs behind
            path = ctx.out / relative
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    verification = ctx.out / "verification.json"
    if "report" in {stage} | dependents(stage) and verification.exists():
        record = read_json(verification)  # reload's checks stay; report's claim about results.zip goes with the zip
        if record.pop("results_zip", None) is not None:
            write_json(verification, record)


REFUSED_EXIT = 3  # an input refused with a learner-facing message (argparse already uses 2 for usage errors)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=[*STAGES, "byod"], required=True)
    parser.add_argument("--input", type=Path, nargs="+", default=None, help="BYOD folders, .zip archives or .h5 files")
    args = parser.parse_args(argv)
    if args.stage == "byod" and args.input is None:
        parser.error("--input is required for --stage byod")  # before start_stage clears any earlier BYOD results
    ctx = Context(args.root)
    start_stage(ctx, args.stage)
    if args.stage == "byod":
        try:
            stage_byod(ctx, args.input)
        except ChipError as exc:  # the learner's input was refused, not a crash: one message, no traceback (RICE-m8)
            print(f"Refused: {exc}", flush=True)
            return REFUSED_EXIT
    else:
        STAGES[args.stage](ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
