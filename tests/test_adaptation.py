"""Probe adaptation, evaluation, embedding, reconstruction and artifact tests on a stub encoder (torch required,
no weights, no terratorch): the linear probe learns a separable patch feature, epoch selection, the transactional
guarantee and the artifact round trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import synthetic_records  # noqa: E402
from prithvi_eo_feature_extraction_pipeline import ADAPTATION_MODES, EMBED_DIM, PATCH, PrithviFeaturePipeline  # noqa: E402
from prithvi_eo_feature_extraction_pipeline import pipeline as pl  # noqa: E402


class _StubEncoder(torch.nn.Module):
    """Mimics `PrithviViT.forward_features`: 24 layer outputs of (B, 1 + T·h·w, 1024) in spatial order. Token
    features are the patch's mean standardised reflectance per band (first six dims), a layer marker (dim 6) and
    zeros elsewhere, so a linear probe can separate dark (burned) patches from bright ones."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward_features(self, x, temporal_coords=None, location_coords=None):
        b, c, t, h, w = x.shape
        pooled = x.reshape(b, c, t, h // PATCH, PATCH, w // PATCH, PATCH).mean(dim=(4, 6))  # (B, C, T, h, w)
        tokens = pooled.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
        feats = torch.zeros(b, tokens.shape[1] + 1, EMBED_DIM)
        feats[:, 1:, :c] = tokens * self.scale
        out = []
        for layer in range(24):
            f = feats.clone()
            f[:, :, 6] = float(layer)
            out.append(f)
        return out


class _StubMAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _StubEncoder()

    def patchify(self, x):
        b, c, t, h, w = x.shape
        p = x.reshape(b, c, t, h // PATCH, PATCH, w // PATCH, PATCH).permute(0, 2, 3, 5, 4, 6, 1)
        return p.reshape(b, t * (h // PATCH) * (w // PATCH), PATCH * PATCH * c)

    def forward(self, x, temporal_coords=None, location_coords=None, mask_ratio=0.75):
        b, c, t, h, w = x.shape
        n = t * (h // PATCH) * (w // PATCH)
        keep = torch.rand(b, n) >= mask_ratio
        mask = (~keep).float().reshape(b, t, h // PATCH, w // PATCH)
        mask = mask.repeat_interleave(PATCH, dim=2).repeat_interleave(PATCH, dim=3)
        pred = torch.zeros_like(x)  # predicts the standardised mean (0) everywhere
        target = self.patchify(x)
        loss_per_patch = ((self.patchify(pred) - target) ** 2).mean(-1)
        m = self.patchify(mask[:, None].expand(-1, c, -1, -1, -1)).mean(-1)
        return {"loss": (loss_per_patch * m).sum() / m.sum()}, pred, mask


def _pipeline() -> PrithviFeaturePipeline:
    return PrithviFeaturePipeline(model=_StubMAE(), device="cpu", weights_dir=Path("unused"), source="stub")


def test_embed_and_reconstruct_shapes():
    pipe = _pipeline()
    records = synthetic_records(2, labels=False)
    stacks = [{"id": r["id"], "frames": r["image"]} for r in records] + [
        {"id": "series", "frames": np.stack([records[0]["image"]] * 3, axis=1)}
    ]
    emb = pipe.embed(stacks, keep_tokens=True)
    assert [e["token_grid"] for e in emb["embeddings"]] == [[1, 32, 32], [1, 32, 32], [3, 32, 32]]
    assert emb["embeddings"][0]["cls"].shape == (EMBED_DIM,) and emb["embeddings"][2]["tokens"].shape == (3, 32, 32, EMBED_DIM)
    assert emb["embeddings"][0]["mean"][6] == 23.0  # the final layer's marker
    rec = pipe.reconstruct(stacks[:1], keep_images=True, seed=3)
    r = rec["results"][0]
    assert 0.7 < r["masked_fraction"] < 0.8 and r["masked_mse"] > 0 and r["baseline_mean_fill_mse"] > 0
    assert r["reconstruction"].shape == (6, 1, 512, 512) and r["mask"].shape == (1, 512, 512)
    with pytest.raises(ValueError, match="mask_ratio"):
        pipe.reconstruct(stacks[:1], mask_ratio=1.5)


def test_predict_and_evaluate_before_adaptation_are_the_majority_baseline():
    pipe = _pipeline()
    records = synthetic_records(3)
    result = pipe.predict(records)
    assert len(result["predictions"]) == 3 and result["predictions"][0]["mask"].shape == (32, 32)
    assert (
        result["predictions"][0]["scores"].shape == (2, 32, 32) and result["predictions"][0]["class_fraction"]["burn scar"] == 0.0
    )
    report = pipe.evaluate(records)
    assert report["model"]["iou"]["burn scar"] == 0.0 and report["model"]["accuracy"] == report["baseline_not_burned"]["accuracy"]
    assert report["adapted"] is False and "patch-level" in report["metric"]


def test_adapt_trains_the_probe_and_keeps_the_best_epoch():
    pipe = _pipeline()
    train, val = synthetic_records(6), synthetic_records(2, seed=50)
    seen = []
    result = pipe.adapt(train, val, epochs=5, lr=1e-1, batch_size=512, seed=0, feature_layer=11, progress=seen.append)
    assert (
        [e["epoch"] for e in seen] == [0, 1, 2, 3, 4, 5]
        and seen[0]["note"].startswith("zero head")
        and seen[0]["val"]["f1"] == 0.0
    )
    assert result["best_epoch"] == min(range(6), key=lambda i: result["history"][i]["val_loss"])
    assert result["n_trainable"] == 2 * EMBED_DIM + 2 and result["feature_layer"] == 11 and result["trainable"] == "probe"
    assert result["history"][-1]["val_loss"] < result["history"][0]["val_loss"]
    assert pipe.feature_layer == 11 and pipe.probe is not None and all(not p.requires_grad for p in pipe.probe.parameters())
    adapted = pipe.evaluate(val)["model"]
    assert adapted["f1"] > 0.9 and adapted == result["history"][result["best_epoch"]]["val"]
    assert torch.equal(pipe.model.encoder.scale, torch.ones(1))  # the encoder is never touched
    assert ADAPTATION_MODES == ("probe",)


def test_adapt_is_transactional_when_the_progress_callback_raises():
    pipe = _pipeline()

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipe.adapt(synthetic_records(4), None, epochs=2, lr=1e-1, progress=boom)
    assert pipe.adapter is None and pipe.probe is None and pipe.feature_stats is None and pipe.feature_layer == pl.FEATURE_LAYER


def test_adapt_refusals():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(synthetic_records(4), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(synthetic_records(4), lr=2.0)
    with pytest.raises(ValueError, match="4..2000"):
        pipe.adapt(synthetic_records(3))
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe.adapt(synthetic_records(4), trainable="decoder")
    with pytest.raises(ValueError, match="feature_layer"):
        pipe.adapt(synthetic_records(4), feature_layer=24)
    with pytest.raises(ValueError, match="only one patch class"):
        pipe.adapt([{**r, "label": np.where(r["label"] < 0, -1, 1)} for r in synthetic_records(4)])
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact("unused")


def test_artifact_round_trip_and_refusals(tmp_path):
    pipe = _pipeline()
    records = synthetic_records(4)
    pipe.adapt(records, records[:2], epochs=3, lr=1e-1, batch_size=512, feature_layer=7)
    adapted = pipe.evaluate(records)["model"]
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"tutorial": "test"})
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["tensors"] == sorted(pl.PROBE_TENSORS) and manifest["adapter"]["feature_layer"] == 7
    assert (
        manifest["adapter"]["trainable"] == "probe"
        and manifest["metadata"] == {"tutorial": "test"}
        and manifest["classes"] == ["not burned", "burn scar"]
    )
    fresh = _pipeline()
    assert fresh.evaluate(records)["model"] != adapted
    fresh.load_artifact(out)
    assert (
        fresh.evaluate(records)["model"] == adapted
        and fresh.adapter["best_epoch"] == manifest["adapter"]["best_epoch"]
        and fresh.feature_layer == 7
    )
    before = pipe.predict(records[:1])["predictions"][0]["scores"]
    after = fresh.predict(records[:1])["predictions"][0]["scores"]
    assert float(np.abs(before - after).max()) < 1e-6
    # a manifest that names another layer or scope is refused before deserialising
    other = dict(manifest, adapter={**manifest["adapter"], "feature_layer": 99})
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_layer"):
        _pipeline().load_artifact(out)
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (out / pl.ARTIFACT_WEIGHTS_NAME).write_bytes((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest or size"):
        _pipeline().load_artifact(out)
