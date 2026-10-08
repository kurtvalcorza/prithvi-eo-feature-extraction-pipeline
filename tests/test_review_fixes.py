"""Regression tests for the 2026-10-02 Notebook Review Framework v1 review of prithvi_eo_feature_extraction_colab.ipynb
(FX-M2, FX-M3, FX-M4, FX-m1..FX-m6, FX-S1; FX-M1 is covered by tests/test_sweep_fixes.py and the stub test).

Most tests need only CI's dependencies (NumPy, tifffile). The end-to-end test at the bottom executes the notebook's own
Section 4-8 cells on a stub encoder (torch and matplotlib required, skipped otherwise): stand-in evidence is plumbing
evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from conftest import synthetic_chip, synthetic_records
from prithvi_eo_feature_extraction_pipeline import (
    MIN_RECORDS,
    SAMPLE_RECORDS,
    byod_minimum_records,
    load_byod_dataset,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
    write_sample_pair,
)

with contextlib.suppress(ImportError):  # import torch before any NumPy linear algebra (Windows DLL order, FIX_PACKET row 6)
    import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_eo_feature_extraction_colab.ipynb"
STEM = "prithvi_eo_feature_extraction"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _code(notebook: dict) -> list[str]:
    return [_source(c) for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [s for s in _code(notebook) if marker in s]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _write_chips(root: Path, n: int, *, groups: list[str] | None = None) -> list[tuple[str, str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n):
        image, label = synthetic_chip(seed=40 + k)
        tifffile.imwrite(root / f"c{k}.tif", image, photometric="minisblack", planarconfig="separate")
        tifffile.imwrite(root / f"c{k}.mask.tif", label.astype(np.int16), photometric="minisblack")
        rows.append((f"chip{k}", f"c{k}.tif", f"c{k}.mask.tif"))
    header = "id,image,label" + (",group" if groups else "") + "\n"
    body = "".join(f"{a},{b},{c}" + (f",{groups[i]}" if groups else "") + "\n" for i, (a, b, c) in enumerate(rows))
    (root / "pairs.csv").write_text(header + body, encoding="utf-8")
    return rows


# --- FX-m1: the true minimum is derived, stated and accepted; failures name the row, file and fix -----------------


def test_fx_m1_minimum_is_seven_and_is_what_the_notebook_states(notebook):
    assert byod_minimum_records() == 7
    records = synthetic_records(7)
    splits = split_dataset(records, seed=0)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [4, 1, 2]
    with pytest.raises(ValueError, match="bring at least 7 chips"):
        split_dataset(records[:6], seed=0)
    md = _markdown(notebook)
    assert "at least **seven** labelled chips" in md and "at least four chips" not in md
    assert "'minimum_chips': byod_minimum_records()" in _cell(notebook, "if USE_BYOD:")


def test_fx_m1_missing_member_and_unreadable_tiff_name_the_row_and_the_fix(tmp_path):
    folder = tmp_path / "byod"
    _write_chips(folder, 3)
    (folder / "c1.tif").unlink()
    with pytest.raises(ValueError, match=r"row 'chip1': image file 'c1\.tif' is listed but not in the folder; add the file"):
        load_byod_dataset(folder)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pairs.csv", "id,image,label\nchip0,c0.tif,c0.mask.tif\n")
        zf.writestr("c0.tif", b"not a tiff")
        zf.writestr("c0.mask.tif", b"not a tiff either")
    with pytest.raises(ValueError, match=r"row 'chip0': image file 'c0\.tif' is not a readable GeoTIFF .*six-band 512 × 512"):
        load_byod_dataset(archive)
    with pytest.raises(ValueError, match="has no pairs.csv"):
        load_byod_dataset(tmp_path)


def test_fx_m1_refusal_probes_fail_on_their_own_rule_with_a_two_chip_test_split(notebook):
    source = _cell(notebook, "probe_fill = ")
    block = source[source.index("probe_fill = ") : source.index("try:\n    validate_stacks(")]
    records = synthetic_records(9)
    namespace = {
        "np": np, "MIN_RECORDS": MIN_RECORDS, "validate_dataset": validate_dataset,
        "test_records": records[:2], "train_records": records[2:6], "val_records": records[6:7],
    }
    out = []
    namespace["print"] = out.append
    exec(compile(block, "section4-probes", "exec"), namespace)
    rejected = {row["probe"]: row.get("rejected", "") for row in out}
    assert set(rejected) == {"five-band scene", "unknown label class"}
    assert all("records;" not in message for message in rejected.values()), rejected
    assert "got (5, 512, 512)" in rejected["five-band scene"] and "label values" in rejected["unknown label class"]


# --- FX-m2: the sample's train/test tile overlap is printed and stated; BYOD groups stay in one role ----------------


def test_fx_m2_sample_tile_overlap_is_printed_and_stated(notebook):
    roles: dict[str, set[str]] = {}
    for name, role, *_rest in SAMPLE_RECORDS:
        roles.setdefault(role, set()).add(name.split(".")[0])
    assert sorted(roles["test"] & roles["train"]) == ["T10TFQ", "T10TGS"]
    assert not roles["validation"] & roles["train"]
    assert "'test_tiles_also_in_training'" in _cell(notebook, "if USE_BYOD:")
    md = _markdown(notebook)
    assert "(HLS tiles T10TFQ and T10TGS) share a tile with training scenes" in md and "should list T10TFQ and T10TGS" in md


def test_fx_m2_group_column_keeps_each_group_in_one_role(tmp_path):
    groups = [f"fire{k // 2}" for k in range(12)]
    _write_chips(tmp_path / "g", 12, groups=groups)
    records = load_byod_dataset(tmp_path / "g")
    assert [r["group"] for r in records] == groups and records[0]["region"] == "fire0"
    splits = split_dataset(records, seed=0)
    role_of: dict[str, set[str]] = {}
    for role, part in splits.items():
        for record in part:
            role_of.setdefault(record["group"], set()).add(role)
    assert all(len(r) == 1 for r in role_of.values()), role_of
    assert len(splits["train"]) >= MIN_RECORDS
    mixed = [dict(r) for r in records]
    mixed[0].pop("group")
    with pytest.raises(ValueError, match="give every chip a group or none"):
        split_dataset(mixed, seed=0)
    two = [{**r, "group": "a" if i % 2 else "b"} for i, r in enumerate(records)]
    with pytest.raises(ValueError, match="at least 3 groups"):
        split_dataset(two, seed=0)


# --- FX-m3: the exported template folder loads through the BYOD loader as-is --------------------------------------


def test_fx_m3_template_folder_zips_into_a_loadable_byod_upload(tmp_path, notebook):
    record = {**synthetic_records(1)[0], "source_id": "T10SDH.2020248.v1", "region": "T10SDH"}
    template = tmp_path / f"{STEM}_byod_template"
    write_sample_pair(record, template / "sample_chip.tif", template / "sample_label.tif")
    write_dataset_csv([record], template / "pairs.csv", files={record["id"]: ("sample_chip.tif", "sample_label.tif")})
    archive = tmp_path / "template.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in template.iterdir():
            zf.write(path, path.name)
    loaded = load_byod_dataset(archive)
    assert [r["id"] for r in loaded] == [record["id"]] and loaded[0]["group"] == "T10SDH"
    assert np.allclose(loaded[0]["image"], record["image"], atol=1e-6)
    code = _cell(notebook, "if USE_BYOD:")
    assert "files={{" not in code and "files={test_records[0]['id']: ('sample_chip.tif', 'sample_label.tif')}" in code
    assert "_sample_pairs.csv" not in code


# --- FX-M2: BYOD-safe names and no stale exports -------------------------------------------------------------------


def test_fx_m2_no_cell_indexes_a_sample_only_key(notebook):
    tutorial = [s for s in _code(notebook) if "record['source_id']" in s or 'record["source_id"]' in s]
    assert tutorial == []
    section8 = _cell(notebook, "probe_maps = pipe.predict(")
    namespace: dict = {}
    exec(section8[section8.index("import re") : section8.index("probe_maps = ")].replace("import matplotlib.pyplot as plt", ""), namespace)
    assert namespace["scene_name"]({"id": "my chip/01"}) == "my_chip_01"
    assert namespace["scene_name"]({"id": "test-000", "source_id": "T10SDH.2020248.v1"}) == "T10SDH.2020248.v1"


def test_fx_m2_section_4_clears_this_notebooks_earlier_exports(notebook, tmp_path, monkeypatch):
    source = _cell(notebook, "if USE_BYOD:")
    block = source[source.index("os.makedirs('outputs'") : source.index("if USE_BYOD:")]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / f"{STEM}_adapter").mkdir(parents=True)
    (tmp_path / "outputs" / f"{STEM}_adapter" / "adapter.safetensors").write_bytes(b"x")
    (tmp_path / "outputs" / f"{STEM}_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / "someone_else.txt").write_text("keep", encoding="utf-8")
    import shutil

    exec(compile(block, "section4-clear", "exec"), {"os": os, "Path": Path, "shutil": shutil})
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == ["someone_else.txt"]


# --- FX-m4 / FX-m5 / FX-m6 / FX-M4 / FX-S1: figures, decision-rule ownership, sample-vs-BYOD reading, activity --


def test_fx_m4_sections_5_and_8_draw_figures(notebook):
    for marker in ("reconstruction = pipe.reconstruct(", "probe_maps = pipe.predict("):
        code = _cell(notebook, marker)
        assert "import matplotlib.pyplot as plt" in code and "plt.subplots(" in code and "plt.show()" in code


def test_fx_m5_m6_text(notebook):
    md = _markdown(notebook)
    assert "**not calibrated**" in md and "is a deployment decision" in md and "owns it and must calibrate it" in md
    assert "**On the default sample**" in md and "**If you ran your own scenes (BYOD),**" in md and "near 0.8" not in md
    assert "## Optional activity: Predict → Change → Run → Observe → Explain" in md and "**Scope of a re-run:**" in md
    assert "run Sections 6, 7 and 8 in that order" in md
    assert "F1 about 0.77" not in md and "watch precision and recall trade places" not in md
    assert "{id, frames}" in md and "{{id" not in md


def test_fx_s1_declares_notebook_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"


# --- End to end on a stub encoder: default run, then the documented BYOD re-run from Section 4 (FX-M2, FX-M3) ------


def _stub_classes(torch, patch: int, embed_dim: int):
    class StubEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward_features(self, x, temporal_coords=None, location_coords=None):
            b, c, t, h, w = x.shape
            pooled = x.reshape(b, c, t, h // patch, patch, w // patch, patch).mean(dim=(4, 6))
            tokens = pooled.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
            feats = torch.zeros(b, tokens.shape[1] + 1, embed_dim)
            feats[:, 1:, :c] = tokens * self.scale
            out = []
            for layer in range(24):
                f = feats.clone()
                f[:, :, 6] = float(layer)
                out.append(f)
            return out

    class StubMAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = StubEncoder()

        def patchify(self, x):
            b, c, t, h, w = x.shape
            p = x.reshape(b, c, t, h // patch, patch, w // patch, patch).permute(0, 2, 3, 5, 4, 6, 1)
            return p.reshape(b, t * (h // patch) * (w // patch), patch * patch * c)

        def forward(self, x, temporal_coords=None, location_coords=None, mask_ratio=0.75):
            b, c, t, h, w = x.shape
            n = t * (h // patch) * (w // patch)
            keep = torch.rand(b, n) >= mask_ratio
            mask = (~keep).float().reshape(b, t, h // patch, w // patch)
            mask = mask.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)
            pred = torch.zeros_like(x)
            target = self.patchify(x)
            loss_per_patch = ((self.patchify(pred) - target) ** 2).mean(-1)
            m = self.patchify(mask[:, None].expand(-1, c, -1, -1, -1)).mean(-1)
            return {"loss": (loss_per_patch * m).sum() / m.sum()}, pred, mask

    return StubMAE


def test_fx_m2_m3_byod_rerun_from_section_4_after_a_default_run(notebook, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    namespace: dict = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell.get("metadata", {}).get("dimer", {}).get("embedded_module"):
            exec(compile(_source(cell), "carried-module", "exec"), namespace)
    pfp = namespace["PrithviFeaturePipeline"]
    stub_mae = _stub_classes(torch, namespace["PATCH"], namespace["EMBED_DIM"])
    weights = tmp_path / "weights"
    (weights / "examples").mkdir(parents=True)
    for k in range(4):
        image, _ = synthetic_chip(seed=80 + k, size=224)
        tifffile.imwrite(weights / "examples" / f"HLS.S30.T13REM.201802{k}T173609.v2.0_cropped.tif", image, photometric="minisblack", planarconfig="separate")

    def stub_pipe():
        return pfp(model=stub_mae(), device="cpu", weights_dir=weights, source="stub")

    monkeypatch.setattr(pfp, "from_pretrained", classmethod(lambda cls, **kw: stub_pipe()))
    sample: dict[str, list] = {"train": [], "validation": [], "test": []}
    k = 0
    for role, n in (("train", 8), ("validation", 4), ("test", 4)):
        for j in range(n):
            image, label = synthetic_chip(seed=100 + k)
            sample[role].append(namespace["check_record"]({"id": f"{role}-{j:03d}", "source_id": f"T00XX{k % 3}.2020{k:03d}.v1", "region": f"T00XX{k % 3}", "split": role, "image": image, "label": label, "source": "synthetic"}))
            k += 1
    for fake in ("timm", "lightning"):
        monkeypatch.setitem(sys.modules, fake, types.SimpleNamespace(__version__="stand-in"))
    namespace.update(
        pipe=stub_pipe(), WEIGHTS_DIR=weights, torch=torch, timm=sys.modules["timm"], lightning=sys.modules["lightning"],
        fetch_sample_dataset=lambda **kw: {r: list(v) for r, v in sample.items()},
        verify_converted=lambda *a, **k: {"files": [{"path": "stand-in"}]}, MANIFEST={"files": []},
        NOTEBOOK_SOURCE={"repository_revision": "stand-in"},
    )
    markers = ("if USE_BYOD:", "reconstruction = pipe.reconstruct(", "adapt_result = pipe.adapt(", "probe_verdict = ", "probe_maps = pipe.predict(")
    cells = [_cell(notebook, m).replace("importlib.metadata.version('terratorch')", "'stand-in'") for m in markers]
    cells[2] = cells[2].replace("EPOCHS = 30  #", "EPOCHS = 3  #")
    monkeypatch.chdir(tmp_path)
    out: list[str] = []
    namespace["print"] = lambda *a, **k: out.append(" ".join(map(str, a)))

    def run(sources):
        for index, source in enumerate(sources):
            exec(compile(source, f"section-{index + 4}", "exec"), namespace)

    run(cells)  # the default path
    assert any("'test_tiles_also_in_training'" in line for line in out)
    assert any("'rows_reloaded_by_the_byod_loader': 1" in line for line in out)
    assert (tmp_path / "outputs" / f"{STEM}_probe_map_T00XX1.2020013.v1.tif").is_file()
    sample_result = json.loads((tmp_path / "outputs" / f"{STEM}_result.json").read_text(encoding="utf-8"))
    assert sample_result["data_source"] == namespace["SAMPLE_LABEL_SOURCE"]

    byod_dir = tmp_path / "my_scenes"
    _write_chips(byod_dir, 8)
    archive = tmp_path / "byod8.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in byod_dir.iterdir():
            zf.write(path, path.name)
    section4 = cells[0].replace("USE_BYOD = False  #", "USE_BYOD = True  #").replace("BYOD_PATH = ''  #", f"BYOD_PATH = {str(archive)!r}  #")
    out.clear()
    run([section4, *cells[1:]])  # FX-M2: the documented re-run from Section 4 completes (no KeyError 'source_id')

    frozen = namespace["frozen_test"]
    assert frozen["adapted"] is False and namespace["frozen_val"]["adapted"] is False  # FX-M3
    assert frozen["model"]["iou"]["burn scar"] == 0.0 == frozen["baseline_not_burned"]["iou"]["burn scar"]
    assert frozen["model"]["accuracy"] == frozen["baseline_not_burned"]["accuracy"]
    fresh = stub_pipe().evaluate(namespace["test_records"])
    assert namespace["comparison"]["burn_iou"]["frozen_zero_head"] == fresh["model"]["iou"]["burn scar"]

    outputs = tmp_path / "outputs"
    names = sorted(p.name for p in outputs.iterdir())
    assert not [n for n in names if "T00XX" in n], names  # no sample probe map survives the BYOD run
    assert [n for n in names if n.startswith(f"{STEM}_probe_map_chip")], names
    for name in (f"{STEM}_result.json", f"{STEM}_evaluation_report.json"):
        assert json.loads((outputs / name).read_text(encoding="utf-8"))["data_source"] == "BYOD (byod8.zip)"
    manifest = json.loads((outputs / f"{STEM}_adapter" / "manifest.json").read_text(encoding="utf-8"))
    assert "BYOD (byod8.zip)" in json.dumps(manifest)
    assert any("'reload_parity'" in line for line in out)
