"""Model-backed checks on the real converted weights (skipped when the snapshot is not staged or terratorch is
absent): strict load and parameter count, embeddings of the four upstream example tiles as one series and as
single frames, masked reconstruction against the mean-fill baseline, a short probe adaptation on synthetic chips
and exact reload parity — on CUDA when it is visible, else on the CPU."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")

from conftest import synthetic_records  # noqa: E402
from prithvi_eo_feature_extraction_pipeline import (  # noqa: E402
    CONVERTED_WEIGHTS_NAME,
    DEFAULT_WEIGHTS_DIR,
    EMBED_DIM,
    PARAMETER_COUNT,
    PrithviFeaturePipeline,
    verify_converted,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_NAME).is_file(), reason="converted weights not staged locally"
)


def test_embed_reconstruct_adapt_and_reload_on_the_available_device(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verify_converted(DEFAULT_WEIGHTS_DIR)
    pipe = PrithviFeaturePipeline.from_pretrained(device=device, require_source=False)
    assert pipe.device == device and sum(p.numel() for p in pipe.model.parameters()) == PARAMETER_COUNT
    tiles = sorted((DEFAULT_WEIGHTS_DIR / "examples").glob("*.tif"))
    assert len(tiles) == 4
    series = {"id": "mexico", "frames": [str(t) for t in tiles]}
    emb = pipe.embed([series, {"id": "first", "frames": str(tiles[0])}], keep_tokens=True)
    assert emb["embeddings"][0]["token_grid"] == [4, 28, 35] and emb["embeddings"][1]["token_grid"] == [1, 28, 35]
    assert emb["embeddings"][0]["cls"].shape == (EMBED_DIM,) and np.isfinite(emb["embeddings"][0]["tokens"]).all()
    rec = pipe.reconstruct([series], seed=0)["results"][0]
    assert 0.0 < rec["masked_mse"] < rec["baseline_mean_fill_mse"]  # the pretraining task beats mean-fill on the upstream tiles
    records = synthetic_records(4)
    held_out = synthetic_records(1, seed=90)
    result = pipe.adapt(records, held_out, epochs=3, lr=1e-2, batch_size=1024)
    assert result["n_trainable"] == 2 * EMBED_DIM + 2 and result["n_train_patches"] == 4 * 1024
    adapted = pipe.evaluate(held_out)["model"]
    pipe.save_artifact(tmp_path / "adapter")
    reloaded = PrithviFeaturePipeline.from_artifact(tmp_path / "adapter", device=device, require_source=False)
    assert reloaded.evaluate(held_out)["model"] == adapted
    before = pipe.predict(held_out)["predictions"][0]["scores"]
    after = reloaded.predict(held_out)["predictions"][0]["scores"]
    assert float(np.abs(before - after).max()) < 1e-5
