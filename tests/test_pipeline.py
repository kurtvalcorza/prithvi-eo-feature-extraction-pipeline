"""Offline tests for the snapshot manifest, staging, the static pickle audit, the chip and stack contracts, the patch
labels, the metrics and the artifact-manifest rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_chip, synthetic_records
from prithvi_eo_feature_extraction_pipeline import (
    BANDS,
    CKPT_ALLOWED_GLOBALS,
    CLASS_NAMES,
    IMAGE_SIZE,
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    NUM_CLASSES,
    PrithviFeaturePipeline,
    audit_pickle,
    dataset_digest,
    majority_baseline,
    patch_labels,
    read_chip,
    read_mask,
    segmentation_metrics,
    stage_missing_files,
    validate_dataset,
    validate_inputs,
    validate_stacks,
    verify_converted,
    verify_snapshot,
)
from prithvi_eo_feature_extraction_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes], *, pin_source: bool = True) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if pin_source and pl.SOURCE_CKPT_NAME not in files:
        entries.append({"path": pl.SOURCE_CKPT_NAME, "bytes": pl.SOURCE_CKPT_BYTES, "sha256": pl.SOURCE_CKPT_SHA256})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": pl.MODEL_KEY,
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and the committed manifest ------------------------------------------------------------------


def test_identity_is_immutable_and_the_manifest_agrees():
    assert len(MODEL_REVISION) == 40
    manifest = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, pl.MODEL_KEY)
    assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    source = [e for e in manifest["files"] if e["path"] == pl.SOURCE_CKPT_NAME]
    assert source and (source[0]["bytes"], source[0]["sha256"]) == (pl.SOURCE_CKPT_BYTES, pl.SOURCE_CKPT_SHA256)
    assert len(pl.CONVERTED_SHA256) == 64 and pl.CONVERTED_BYTES > 0 and len(pl.PICKLE_AUDIT_SHA256) == 64
    assert pl.PARAMETER_COUNT == 330_419_712 and pl.STATE_TENSORS == 398 and pl.ENCODER_TENSORS + pl.DECODER_TENSORS == 398
    assert [e["path"] for e in manifest["files"]][:3] == [pl.SOURCE_CKPT_NAME, "README.md", "config.json"] and len(
        manifest["files"]
    ) == 7


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x", pl.SOURCE_CKPT_NAME: b"ckpt"}, pin_source=False)
    with pytest.raises(ValueError, match="disagrees with the package constant"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).write_bytes(b"ckpt-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x"}, pin_source=False)
    with pytest.raises(ValueError, match="does not list"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_converted_checks_the_pinned_digest(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        verify_converted(tmp_path)
    (tmp_path / pl.CONVERTED_WEIGHTS_NAME).write_bytes(b"x" * pl.CONVERTED_BYTES if pl.CONVERTED_BYTES < 10_000 else b"x")
    with pytest.raises(ValueError, match="size|sha256"):
        verify_converted(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"fetched")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [pl.SOURCE_CKPT_NAME]
    assert calls == [pl.SOURCE_CKPT_NAME]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=downloader)


# --- static pickle audit ----------------------------------------------------------------------------------


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def _torch_like_archive(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


def test_audit_pickle_lists_globals_and_refuses_code(tmp_path, forbid_model_imports):
    benign = tmp_path / "benign.pt"
    benign.write_bytes(_torch_like_archive(pickle.dumps({"state_dict": {"a": 1}, "epoch": 3})))
    report = audit_pickle(benign)
    assert report["torch_archive"] and report["pickles"] == 1 and report["globals"] == [] and report["violations"] == []
    evil = tmp_path / "evil.pt"
    evil.write_bytes(_torch_like_archive(pickle.dumps({"state_dict": _Evil()})))
    with pytest.raises(ValueError, match="pickle audit failed.*os.system|posix.system|nt.system"):
        audit_pickle(evil)
    plain = tmp_path / "plain.pickle"
    plain.write_bytes(pickle.dumps(complex(1, 2)))
    with pytest.raises(ValueError, match="pickle audit failed"):
        audit_pickle(plain)
    assert "collections.OrderedDict" in CKPT_ALLOWED_GLOBALS and len(CKPT_ALLOWED_GLOBALS) == 3
    with pytest.raises(FileNotFoundError):
        audit_pickle(tmp_path / "missing.pt")


# --- chip contract ----------------------------------------------------------------------------------------


def test_validate_dataset_reports_balance_and_digest(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["n_labelled"] == 6 and report["bands"] == list(BANDS)
    assert 0.1 < report["class_pixel_fraction"]["burn scar"] < 0.5 and report["ignored_pixels"] == 6 * 64
    assert report["digest"] == dataset_digest(records) and len(report["digest"]) == 64
    assert INPUT_SCHEMA["image_size"] == IMAGE_SIZE == 512 and INPUT_SCHEMA["classes"] == {"0": "not burned", "1": "burn scar"}


def test_validate_dataset_refusals_name_the_rule(forbid_model_imports):
    records = synthetic_records(6)
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"})
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="missing 'image'"):
        validate_dataset([{"id": "a", "label": records[0]["label"]}, *records[1:]])
    with pytest.raises(ValueError, match="must have shape \\(6, 512, 512\\)"):
        validate_dataset([{**records[0], "image": records[0]["image"][:5]}, *records[1:]])
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset([{**records[0], "image": records[0]["image"] * np.nan}, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records])
    with pytest.raises(ValueError, match="has no label"):
        validate_dataset([{"id": "u", "image": records[0]["image"]}, *records[1:]])
    with pytest.raises(ValueError, match="label values \\[2\\] outside"):
        validate_dataset([{**records[0], "label": records[0]["label"] + 1}, *records[1:]])
    with pytest.raises(ValueError, match="label must have shape"):
        validate_dataset([{**records[0], "label": records[0]["label"][:100]}, *records[1:]])
    with pytest.raises(ValueError, match="no pixel of class 1"):
        validate_dataset([{**r, "label": np.zeros_like(r["label"])} for r in records])
    assert validate_dataset([{"id": r["id"], "image": r["image"]} for r in records], require_labels=False)["n_labelled"] == 0


def test_reflectance_scaling_and_no_data(forbid_model_imports):
    image, label = synthetic_chip()
    scaled = (image * 10_000).astype(np.float32)
    scaled[0, :4, :4] = -9999.0
    report = validate_inputs({"id": "s", "image": scaled, "label": label})
    assert report["reflectance_range"][0] == 0.0 and report["reflectance_range"][1] <= 1.0
    assert report["label_fraction"]["burn scar"] > 0 and report["ignored_pixels"] == 64
    with pytest.raises(ValueError, match="plausible range"):
        validate_inputs({"id": "s", "image": image * 50_000.0})


def test_geotiff_round_trip(tmp_path, forbid_model_imports):
    import tifffile

    image, label = synthetic_chip()
    stack13 = np.zeros((13, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int16)
    for out_index, band_index in enumerate(pl.S2_L1C_BAND_INDICES):
        stack13[band_index] = (image[out_index] * 10_000).astype(np.int16)
    tifffile.imwrite(tmp_path / "chip.tif", stack13, photometric="minisblack", planarconfig="separate")
    tifffile.imwrite(tmp_path / "hwc.tif", np.moveaxis(image, 0, -1), photometric="minisblack", planarconfig="contig")
    tifffile.imwrite(tmp_path / "label.tif", label.astype(np.int16))
    chip = read_chip(tmp_path / "chip.tif")
    assert chip.shape == (6, IMAGE_SIZE, IMAGE_SIZE) and chip.dtype == np.float32
    assert np.allclose(chip, (image * 10_000).astype(np.int16), atol=1)
    assert read_chip(tmp_path / "hwc.tif").shape == (6, IMAGE_SIZE, IMAGE_SIZE)
    mask = read_mask(tmp_path / "label.tif")
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE) and mask.dtype == np.int64 and set(np.unique(mask)) == {-1, 0, 1}
    assert validate_inputs({"id": "p", "image": str(tmp_path / "chip.tif"), "label": str(tmp_path / "label.tif")})["has_label"]
    with pytest.raises(ValueError, match="image file not found"):
        validate_inputs({"id": "p", "image": str(tmp_path / "none.tif")})


# --- metrics ----------------------------------------------------------------------------------------------


def test_segmentation_metrics_and_baseline(forbid_model_imports):
    _, label = synthetic_chip()
    perfect = segmentation_metrics([np.where(label < 0, 0, label)], [label])
    assert perfect["iou"]["burn scar"] == 1.0 and perfect["f1"] == 1.0 and perfect["mean_iou"] == 1.0
    assert perfect["labelled_pixels"] == IMAGE_SIZE * IMAGE_SIZE - 64
    baseline = majority_baseline([label])
    assert baseline["iou"]["burn scar"] == 0.0 and baseline["recall"] == 0.0
    assert baseline["accuracy"] == round(1 - baseline["positive_fraction"], 4)
    assert baseline["note"].startswith("predicts 'not burned'")
    flipped = segmentation_metrics([np.where(label < 0, 0, 1 - label)], [label])
    assert flipped["accuracy"] == 0.0 and flipped["confusion"][0][0] == 0
    with pytest.raises(ValueError, match="equal length"):
        segmentation_metrics([label], [])
    assert len(CLASS_NAMES) == NUM_CLASSES == 2


# --- artifact manifest (static checks, no weights) -------------------------------------------------------


def _good_manifest(root: Path) -> dict:
    (root / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    return {
        "format": pl.ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": pl.MODEL_KEY, "converted_sha256": pl.CONVERTED_SHA256},
        "adapter": {"trainable": "probe", "feature_layer": pl.FEATURE_LAYER},
        "tensors": sorted(pl.PROBE_TENSORS),
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
    }


def test_artifact_manifest_static_checks(tmp_path, forbid_model_imports):
    good = _good_manifest(tmp_path)
    path, mode = PrithviFeaturePipeline.check_artifact_manifest(tmp_path, good)
    assert path == (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).resolve() and mode == "probe"
    cases = {
        "format": ({**good, "format": "other"}, "artifact format"),
        "version": ({**good, "format_version": "2.0"}, "format_version"),
        "base": ({**good, "base_model": {**good["base_model"], "revision": "0" * 40}}, "different base model"),
        "digest": ({**good, "base_model": {**good["base_model"], "converted_sha256": "0" * 64}}, "converted-base digest"),
        "two files": ({**good, "files": good["files"] * 2}, "exactly one weights file"),
        "other name": ({**good, "files": [{**good["files"][0], "path": "weights.safetensors"}]}, "must be named"),
        "mode": ({**good, "adapter": {"trainable": "everything", "feature_layer": 11}}, "adapter.trainable"),
        "layer": ({**good, "adapter": {"trainable": "probe", "feature_layer": 40}}, "feature_layer"),
        "tensors": ({**good, "tensors": ["probe.weight"]}, "tensor list must be exactly"),
    }
    for name, (manifest, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            PrithviFeaturePipeline.check_artifact_manifest(tmp_path, manifest)
        del name


# --- stacks and patch labels ------------------------------------------------------------------------------


def test_validate_stacks_and_patch_labels(forbid_model_imports):
    image, label = synthetic_chip()
    stacks = validate_stacks([{"id": "one", "frames": image}, {"id": "four", "frames": np.stack([image] * 4, axis=1)}])
    assert stacks["n_records"] == 2 and stacks["shapes"] == [(6, 1, 512, 512), (6, 4, 512, 512)]
    assert stacks["records"][0]["frames"].shape == (6, 1, 512, 512)
    with pytest.raises(ValueError, match="T must be in 1..4"):
        validate_stacks([{"id": "five", "frames": np.stack([image] * 5, axis=1)}])
    with pytest.raises(ValueError, match="multiples of 16"):
        validate_stacks([{"id": "odd", "frames": image[:, :, :500]}])
    with pytest.raises(ValueError, match="must have shape \(6, T, H, W\)"):
        validate_stacks([{"id": "bands", "frames": image[:5]}])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_stacks([{"id": "a", "frames": image}, {"id": "a", "frames": image}])
    with pytest.raises(ValueError, match="frame file not found"):
        validate_stacks([{"id": "f", "frames": ["nowhere.tif"]}])
    grid = patch_labels(label)
    assert grid.shape == (32, 32) and set(np.unique(grid).tolist()) <= {-1, 0, 1} and (grid == 1).any()
    assert grid[0, 0] == 0  # the 8 × 8 no-data corner is a quarter of the patch: still labelled
    mostly_ignored = np.full((512, 512), -1, dtype=np.int64)
    mostly_ignored[:, :256] = 1
    assert (patch_labels(mostly_ignored)[:, :16] == 1).all() and (patch_labels(mostly_ignored)[:, 16:] == -1).all()
    with pytest.raises(ValueError, match="multiple of 16"):
        patch_labels(label[:500])
