import builtins

import numpy as np
import pytest

MODEL_LIBRARIES = {"torch", "terratorch", "timm", "lightning", "safetensors", "huggingface_hub", "segmentation_models_pytorch"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_chip(*, seed: int = 0, size: int = 512, bands: int = 6, positive_fraction: float = 0.3):
    """A smooth six-band reflectance chip in [0, 1] with a dark blob of the positive class (burn scar) and its label."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    cx, cy, r = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), np.sqrt(positive_fraction / np.pi)
    positive = ((x - cx) ** 2 + (y - cy) ** 2) < r**2
    base = np.stack([0.08 + 0.05 * b + 0.1 * np.sin(6 * x + b) * np.cos(5 * y) for b in range(bands)]).astype(np.float32)
    base = base + rng.normal(0, 0.01, base.shape).astype(np.float32)
    base[:, positive] *= 0.4
    image = np.clip(base, 0.001, 1.0).astype(np.float32)
    label = positive.astype(np.int64)
    label[:8, :8] = -1  # a no-data corner
    return image, label


def synthetic_records(n: int = 6, *, seed: int = 0, labels: bool = True):
    out = []
    for i in range(n):
        image, label = synthetic_chip(seed=seed + i)
        record = {"id": f"chip-{i:03d}", "image": image}
        if labels:
            record["label"] = label
        out.append(record)
    return out
