"""Philippines rice-mapping capstone (E2E, WORKSHOP; docs/philippines-rice-mapping-capstone-spec.md §13): generator
parity, carried-source integrity, and CPU checks of the carried stage runner on synthetic arrays.

The runner checks test the protocol and bookkeeping contract (frame selection, gap accounting, normalisation, split,
metrics, BYOD refusals, artifact and file round trips), not model skill. No model, torch or real chip is loaded.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "rice_capstone"
NOTEBOOK = ROOT / "tutorials" / "DIMER_Philippines_Rice_Mapping_Capstone.ipynb"
LOCK = ROOT / "tutorials" / "requirements-rice-capstone.lock.txt"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cap = _load("rice_capstone_runner", TOOLS / "capstone.py")
gen = _load("rice_capstone_builder", TOOLS / "build_capstone.py")

# The 32 training rows of the pinned INDEX.jsonl (sha256 e92bbb9f...): chip id, region, stratum.
TRAIN_ROWS = [
    ("ph_92_743_2021", "PH-01", "rice"), ("ph_93_690_2019", "PH-01", "rice"), ("ph_98_740_2021", "PH-01", "rice"),
    ("ph_133_770_2020", "PH-02", "rice"), ("ph_157_764_2020", "PH-02", "nonrice"), ("ph_183_598_2021", "PH-05", "nonrice"),
    ("ph_204_586_2021", "PH-05", "rice"), ("ph_211_574_2019", "PH-05", "rice"), ("ph_214_567_2019", "PH-05", "nonrice"),
    ("ph_217_577_2019", "PH-05", "rice"), ("ph_251_597_2019", "PH-05", "nonrice"), ("ph_197_347_2021", "PH-09", "nonrice"),
    ("ph_211_337_2020", "PH-09", "rice"), ("ph_212_351_2021", "PH-09", "rice"), ("ph_265_306_2019", "PH-12", "rice"),
    ("ph_279_266_2020", "PH-12", "nonrice"), ("ph_283_301_2020", "PH-12", "rice"), ("ph_300_361_2019", "PH-13", "nonrice"),
    ("ph_314_385_2020", "PH-13", "nonrice"), ("ph_320_362_2019", "PH-13", "rice"), ("ph_327_360_2020", "PH-13", "nonrice"),
    ("ph_330_386_2021", "PH-13", "rice"), ("ph_338_348_2019", "PH-13", "nonrice"), ("ph_242_331_2019", "PH-14", "nonrice"),
    ("ph_245_300_2020", "PH-14", "rice"), ("ph_257_292_2020", "PH-14", "rice"), ("ph_259_299_2020", "PH-14", "rice"),
    ("ph_128_748_2021", "PH-15", "rice"), ("ph_127_597_2021", "PH-40", "rice"), ("ph_172_604_2021", "PH-40", "nonrice"),
    ("ph_102_578_2021", "PH-41", "nonrice"), ("ph_122_572_2019", "PH-41", "rice"),
]


def _rows():
    rows = []
    for cid, region, stratum in TRAIN_ROWS:
        group, year = cid.rsplit("_", 1)
        rows.append({"group_id": group, "year": int(year), "region_iso": region, "stratum": stratum, "split": "train"})
    return rows


def _chip(seed: int = 0, size: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    s2 = rng.uniform(0.01, 0.6, (24, 6, size, size)).astype(np.float32)
    valid = np.ones((24, size, size), dtype=np.uint8)
    label = (rng.uniform(size=(size, size)) > 0.7).astype(np.uint8)
    label[:2] = 255
    return {"S2": s2, "S2_valid": valid, "label": label}


# ---------------------------------------------------------------------------------------- frame selection and input


def test_frame_selection_picks_the_clearest_bin_per_quarter_and_earliest_on_ties():
    valid = np.zeros((24, 4, 4), dtype=np.uint8)
    valid[0:2] = 1  # quarter 1: bins 0 and 1 tie at 1.0 -> earliest (0)
    valid[9, :2] = 1  # quarter 2: bin 9 has 0.5, bin 10 has 1.0 -> 10
    valid[10] = 1
    valid[13, 0, 0] = 1  # quarter 3: only bin 13 observed (1/16)
    # quarter 4: nothing observed -> earliest bin of the quarter (18), valid fraction 0
    result = cap.select_frames(valid)
    assert result["bins"] == [0, 10, 13, 18]
    assert result["valid_fractions"] == [1.0, 1.0, 0.0625, 0.0]
    assert cap.select_frames(valid) == result  # deterministic


def test_nan_accounting_records_filled_shares_and_fills_with_the_pretraining_mean():
    chip = _chip(1)
    chip["S2_valid"][6, :64] = 0  # half of bin 6 unobserved
    chip["S2_valid"][18] = 0  # all of bin 18 unobserved
    clean = cap.validate_chip(chip)
    assert np.isnan(clean["S2"][6, :, :64]).all() and np.isfinite(clean["S2"][6, :, 64:]).all()
    x, record = cap.prithvi_input(clean, [0, 6, 12, 18])
    assert x.shape == (6, 4, 128, 128) and np.isfinite(x).all()
    assert record["filled_fraction_per_frame"] == [0.0, 0.5, 0.0, 1.0]
    assert record["filled_fraction"] == pytest.approx((0.5 + 1.0) / 4)
    assert record["pixels_without_any_observation"] == 0  # every pixel is observed in bin 0
    assert (x[:, 1, :64] == 0).all() and (x[:, 3] == 0).all()
    clean["S2_valid"][[0, 6, 12, 18], :, :3] = 0
    clean = cap.validate_chip(clean)
    assert cap.prithvi_input(clean, [0, 6, 12, 18])[1]["pixels_without_any_observation"] == 128 * 3


def test_normalisation_matches_the_pinned_statistics_is_pure_and_revalidation_never_rescales():
    clean = cap.validate_chip(_chip(2))
    before = clean["S2"].copy()
    x1, _ = cap.prithvi_input(clean, [0, 6, 12, 18])
    x2, _ = cap.prithvi_input(clean, [0, 6, 12, 18])
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(clean["S2"], before)  # input untouched
    mean = np.asarray(cap.pp.MEANS, dtype=np.float32)[:, None, None]
    std = np.asarray(cap.pp.STDS, dtype=np.float32)[:, None, None]
    np.testing.assert_allclose(x1[:, 2], (before[12] * 10000.0 - mean) / std, rtol=1e-5, atol=1e-5)
    again = cap.validate_chip(clean)  # a checked chip passes unchanged
    np.testing.assert_array_equal(again["S2"], clean["S2"])
    assert cap.BAND_NAMES == ("B2", "B3", "B4", "B8A", "B11", "B12") and len(cap.pp.BANDS) == 6
    assert cap.FEATURE_LAYER == 11


# ---------------------------------------------------------------------------------------- labels and metrics


def test_label_mapping_ignores_255_and_refuses_other_values():
    label = np.array([[0, 1], [255, 1]], dtype=np.uint8)
    prob = np.array([[0.9, 0.9], [0.9, 0.1]])
    assert cap.confusion(prob, label, 0.5) == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}
    prob_nan_on_ignored = np.array([[0.9, 0.9], [np.nan, 0.1]])
    assert cap.confusion(prob_nan_on_ignored, label, 0.5)["tp"] == 1  # ignored pixels are not scored
    with pytest.raises(ValueError, match="non-finite"):
        cap.confusion(np.array([[np.nan, 0.9], [0.9, 0.1]]), label, 0.5)
    with pytest.raises(ValueError, match="0/1/255"):
        cap.confusion(prob, np.array([[0, 2], [1, 1]]), 0.5)
    assert cap.confusion(np.array([[0.5]]), np.array([[1]]), 0.5)["tp"] == 1  # score >= threshold is rice


@pytest.mark.parametrize("fill", [0, 1, 255])
def test_label_two_probe_is_invalid_even_on_a_chip_without_rice(fill):
    # c36c3d8 hosted run: the first fit chip had no rice, so mapping 1 -> 2 changed nothing and the probe passed.
    chip = _chip()
    chip["label"] = np.full_like(chip["label"], fill)
    bad = cap._with_label_two(chip["label"])
    assert (bad == 2).sum() == 1 and (chip["label"] == fill).all()  # one pixel changed; the original is untouched
    with pytest.raises(cap.ChipError, match="outside 0"):
        cap.validate_chip({**chip, "label": bad}, name="probe")


def test_stages_refuse_to_start_before_their_prerequisites_and_clear_stale_receipts(tmp_path):
    # c36c3d8 hosted run: prepare failed, yet later stages ran on the records it had already written.
    ctx = cap.Context(tmp_path)
    receipts = ctx.work / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="prepare"):
        cap.start_stage(ctx, "baseline")
    for name in ("prepare", "baseline", "features", "train-head"):
        (receipts / f"{name}.json").write_text("{}")
    with pytest.raises(RuntimeError, match="activity"):
        cap.start_stage(ctx, "evaluate")
    (receipts / "activity.json").write_text("{}")
    (receipts / "evaluate.json").write_text("{}")
    cap.start_stage(ctx, "evaluate")  # a re-run first deletes its own stale receipt
    assert not (receipts / "evaluate.json").exists()
    cap.start_stage(ctx, "prepare")  # a fresh prepare invalidates every downstream receipt
    assert not list(receipts.glob("*.json"))
    assert set(cap.PREREQUISITES) == (set(cap.STAGES) - {"prepare"}) | {"byod"}
    assert cap.PREREQUISITES["byod"] == ("reload",)  # BYOD loads the exported head_adapter, verified by reload


def test_stale_fine_tune_results_are_dropped_before_the_validation_freeze(tmp_path):
    # Codex P2 on 42d8f40: rerunning train-head after a fine-tune invalidated its receipt, but the C-FT selection and
    # sealed predictions stayed and would have been scored beside the new canonical systems.
    ctx = cap.Context(tmp_path)
    (ctx.work / "receipts").mkdir(parents=True, exist_ok=True)
    cap.write_json(ctx.out / "selection.json", {"B": {}, "C": {}, "activity": {}, "C-FT": {"selected_before_validation": True}})
    np.savez(ctx.work / "cft_validation_probs.npz", a=np.zeros(2))
    (ctx.work / "receipts" / "finetune.json").write_text("{}")
    assert cap.drop_stale_finetune(ctx) is False  # a current receipt keeps C-FT
    (ctx.work / "receipts" / "finetune.json").unlink()  # as a train-head restart or a failed fine-tune leaves it
    assert cap.drop_stale_finetune(ctx) is True
    assert "C-FT" not in cap.read_json(ctx.out / "selection.json")
    assert not (ctx.work / "cft_validation_probs.npz").exists()


def test_restarting_an_upstream_stage_invalidates_every_dependent_receipt(tmp_path):
    # Agent Relay review R3 on 8fbcaa6: restarting features and failing left train-head/activity receipts, and
    # evaluate (whose direct prerequisites omit features) still accepted the run.
    ctx = cap.Context(tmp_path)
    receipts = ctx.work / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    for name in ("prepare", "baseline", "features", "train-head", "activity"):
        (receipts / f"{name}.json").write_text("{}")
    cap.start_stage(ctx, "features")  # the restart; suppose it then fails before writing its receipt
    assert sorted(p.stem for p in receipts.glob("*.json")) == ["baseline", "prepare"]
    with pytest.raises(RuntimeError, match="train-head"):
        cap.start_stage(ctx, "evaluate")
    assert cap.dependents("features") == {"train-head", "activity", "finetune", "evaluate", "export", "reload",
                                          "report", "byod"}
    assert cap.dependents("report") == set() and "byod" not in cap.dependents("report")


def test_a_second_scoring_is_refused_before_any_receipt_is_cleared(tmp_path):
    # Notebook review RC-m2 at 764291c: re-running Section 8 cleared evaluate/export/reload/report receipts before the
    # "already scored" refusal, stranding Sections 10-12 in that run directory.
    ctx = cap.Context(tmp_path)
    receipts = ctx.work / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    done = ("prepare", "baseline", "features", "train-head", "activity", "evaluate", "export", "reload", "report")
    for name in done:
        (receipts / f"{name}.json").write_text("{}")
    (ctx.work / "frozen.json").write_text("{}")
    for stage in ("evaluate", "finetune"):
        with pytest.raises(RuntimeError, match="already scored"):
            cap.start_stage(ctx, stage)
    assert sorted(p.stem for p in receipts.glob("*.json")) == sorted(done)


def test_byod_collection_starts_clean_and_refuses_duplicates_unsafe_and_oversized_input(tmp_path, monkeypatch):
    import zipfile

    site_a, site_b, target = tmp_path / "a", tmp_path / "b", tmp_path / "byod_input"
    site_a.mkdir()
    site_b.mkdir()
    (site_a / "x_2020.h5").write_bytes(b"a")
    (site_b / "y_2021.h5").write_bytes(b"b")
    target.mkdir()
    (target / "stale_2019.h5").write_bytes(b"old")  # left by an earlier attempt
    assert [p.name for p in cap.collect_byod([site_a, site_b], target)] == ["x_2020.h5", "y_2021.h5"]
    archive = tmp_path / "chips.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/z_2020.h5", b"z")
    assert [p.name for p in cap.collect_byod([archive], target)] == ["z_2020.h5"]  # the re-run replaced, not merged
    (site_b / "x_2020.h5").write_bytes(b"other site")
    with pytest.raises(cap.ChipError, match="share the file name x_2020.h5"):
        cap.collect_byod([site_a, site_b], target)
    # Codex P2 on 4787f32: a rejected retry must not keep the earlier upload either; it copies nothing.
    assert list(target.glob("*.h5")) == []
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as handle:
        handle.writestr("../escape_2020.h5", b"e")
    with pytest.raises(cap.ChipError, match="unsafe member"):
        cap.collect_byod([unsafe], target)
    (site_b / "x_2020.h5").unlink()
    monkeypatch.setattr(cap, "BYOD_MAX_FILES", 1)
    with pytest.raises(cap.ChipError, match="exceeds 1 chips"):
        cap.collect_byod([site_a, site_b], target)
    with pytest.raises(cap.ChipError, match="expected a folder"):
        cap.collect_byod([tmp_path / "notes.txt"], target)
    # Codex P2 on 42d8f40: members are streamed and the expanded size is enforced while copying.
    monkeypatch.setattr(cap, "BYOD_MAX_FILES", 200)
    big = tmp_path / "big.zip"
    with zipfile.ZipFile(big, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("big_2020.h5", bytes(3 << 20))
    monkeypatch.setattr(cap, "BYOD_MAX_BYTES", 4 << 20)
    assert [p.stat().st_size for p in cap.collect_byod([big], target)] == [3 << 20]
    monkeypatch.setattr(cap, "BYOD_MAX_BYTES", 2 << 20)
    with pytest.raises(cap.ChipError, match="exceeds"):
        cap.collect_byod([big], target)


def test_a_rejected_byod_rerun_leaves_no_earlier_outputs_behind(tmp_path, monkeypatch):
    # Codex P2 on 2d43f2e: cleanup ran after validation, so a malformed re-upload left the previous predictions and
    # byod_results.zip in place as though they belonged to the rejected attempt.
    ctx = cap.Context(tmp_path)
    (ctx.out / "byod" / "predictions").mkdir(parents=True)
    (ctx.out / "byod" / "predictions" / "old_C.tif").write_bytes(b"old")
    (ctx.out / "byod_results.zip").write_bytes(b"old")

    def refuse(*_args, **_kwargs):
        raise cap.ChipError("malformed upload")

    monkeypatch.setattr(cap, "_torch_setup", refuse)  # fail as early as possible, before any validation
    with pytest.raises(cap.ChipError, match="malformed"):
        cap.stage_byod(ctx, [tmp_path / "nothing.h5"])
    assert not (ctx.out / "byod").exists() and not (ctx.out / "byod_results.zip").exists()


def test_the_bundled_verification_record_equals_the_displayed_one(tmp_path):
    # Codex P2 on 4787f32: results.zip was built before results_zip was written into verification.json.
    import zipfile

    ctx = cap.Context(tmp_path)
    cap.write_json(ctx.out / "verification.json", {"binary_decisions_identical": True})
    (ctx.out / "metrics.csv").write_text("system,f1\nB,0.75\n")
    check = cap.bundle_results(ctx)
    shown = cap.read_json(ctx.out / "verification.json")
    assert shown["results_zip"] == check and check["verified"] is True
    with zipfile.ZipFile(ctx.out / "results.zip") as archive:
        assert json.loads(archive.read("verification.json")) == shown


def test_round_codex_539b865_findings():
    # 1. the manifest's output description follows the real order: upsample the logits, then softmax.
    assert cap.HEAD_OUTPUT_ORDER.index("upsampled") < cap.HEAD_OUTPUT_ORDER.index("softmax")
    assert "align_corners=False" in cap.HEAD_OUTPUT_ORDER
    # 3. undecodable optional attributes leave the chip ungeoreferenced instead of raising.
    real = {"affine": "[20.0, 0.0, 614400.0, 0.0, -20.0, 1095680.0]", "crs": "EPSG:32651", "group_id": "ph_240_427"}
    assert cap.georeference_attrs({**real, "crs": b"\xff\xfe"}) == {"group": "ph_240_427"}
    assert cap.georeference_attrs({"x_min": 1.0, "y_max": 2.0, "crs": b"\xff"}) == {}
    assert cap.georeference_attrs({**real, "group_id": b"\xff", "group": b"\xfe"}) == {"x_min": 614400.0,
                                                                                        "y_max": 1095680.0}
    assert cap.georeference_attrs({**real, "affine": b"\xff"}) == {"group": "ph_240_427"}
    # 4. an all-ignore label is not labelled for adaptation; one 0 or 1 is enough.
    assert not cap.has_labelled_pixels(None)
    assert not cap.has_labelled_pixels(np.full((4, 4), 255, dtype=np.uint8))
    assert cap.has_labelled_pixels(np.array([[255, 0]], dtype=np.uint8))


def test_dropping_stale_fine_tuning_removes_its_output_records(tmp_path):
    # 2. Codex on 539b865: finetune.json and finetune_curve.csv stayed in outputs/ and were bundled into results.zip.
    ctx = cap.Context(tmp_path)
    (ctx.work / "receipts").mkdir(parents=True, exist_ok=True)
    cap.write_json(ctx.out / "selection.json", {"B": {}, "C": {}, "activity": {}, "C-FT": {"selected_before_validation": True}})
    for path in (ctx.out / "finetune.json", ctx.out / "finetune_curve.csv", ctx.work / "cft_validation_probs.npz"):
        path.write_text("stale")
    assert cap.drop_stale_finetune(ctx) is True
    assert not any(p.exists() for p in (ctx.out / "finetune.json", ctx.out / "finetune_curve.csv",
                                        ctx.work / "cft_validation_probs.npz"))


def test_round_codex_6300e3b_findings(tmp_path):
    import zipfile

    # 4. after the freeze, scoring and every stage upstream of it refuse before touching anything.
    assert cap.ancestors("evaluate") == {"prepare", "baseline", "features", "train-head", "activity"}
    ctx = cap.Context(tmp_path / "run")
    receipts = ctx.work / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    for name in ("prepare", "baseline", "features", "train-head", "activity", "evaluate", "export"):
        (receipts / f"{name}.json").write_text("{}")
    (ctx.work / "frozen.json").write_text("{}")
    for stage in ("prepare", "baseline", "features", "train-head", "activity", "evaluate", "finetune"):
        with pytest.raises(RuntimeError, match="already scored"):
            cap.start_stage(ctx, stage)
    assert len(list(receipts.glob("*.json"))) == 7
    cap.start_stage(ctx, "reload")  # downstream finishing stages may still run after the freeze
    # 2. a corrupt or misnamed zip becomes an actionable refusal naming the archive.
    bad = tmp_path / "chips.zip"
    bad.write_bytes(b"this is not a zip archive")
    with pytest.raises(cap.ChipError, match="chips.zip: not a readable .zip"):
        cap.collect_byod([bad], tmp_path / "byod_input")
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("a_2020.h5", bytes(4096))
    raw = bytearray(good.read_bytes())
    start = raw.index(b"a_2020.h5") + len("a_2020.h5")
    raw[start:start + 40] = bytes(40)  # corrupt the deflated payload but keep the directory readable
    good.write_bytes(bytes(raw))
    with pytest.raises(cap.ChipError, match="could not be extracted|not a readable"):
        cap.collect_byod([good], tmp_path / "byod_input")


def test_issue_17_follow_ups(tmp_path):
    import zipfile

    # 1. the recorded fine-tune configuration names every trained parameter group.
    assert "cls_token" in cap.FT_CONFIG["trainable"]
    # 2. restarting export removes export/reload/report outputs, not only their receipts.
    ctx = cap.Context(tmp_path / "run")
    receipts = ctx.work / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    for name in ("prepare", "baseline", "features", "train-head", "activity", "evaluate", "export", "reload", "report"):
        (receipts / f"{name}.json").write_text("{}")
    (ctx.out / "head_adapter").mkdir()
    (ctx.out / "head_adapter" / "adapter.safetensors").write_bytes(b"old")
    (ctx.out / "predictions").mkdir()
    for name in ("verification.json", "results.zip", "run_summary.json", "conclusion.md", "metrics.csv"):
        (ctx.out / name).write_text("old")
    (ctx.out / "byod").mkdir()
    (ctx.out / "byod_results.zip").write_text("old")
    cap.start_stage(ctx, "export")
    gone = ("head_adapter", "predictions", "verification.json", "results.zip", "run_summary.json", "conclusion.md",
            "byod", "byod_results.zip")  # Codex on 79900c2: BYOD outputs depend on the reloaded head too
    assert not any((ctx.out / name).exists() for name in gone)
    assert (ctx.out / "metrics.csv").exists()  # evaluate's outputs stay: evaluate did not restart
    # 3. the canonical bundle excludes BYOD outputs and the BYOD bundle.
    cap.write_json(ctx.out / "verification.json", {"binary_decisions_identical": True})
    (ctx.out / "byod" / "predictions").mkdir(parents=True)
    (ctx.out / "byod" / "predictions" / "user_C.tif").write_bytes(b"user")
    (ctx.out / "byod_results.zip").write_bytes(b"user bundle")
    cap.bundle_results(ctx)
    with zipfile.ZipFile(ctx.out / "results.zip") as archive:
        names = archive.namelist()
    assert "metrics.csv" in names and not any(n.startswith("byod") for n in names)


def test_metric_edge_cases_report_null_with_reasons():
    no_rice = np.zeros((4, 4), dtype=np.uint8)
    silent = cap.scores(cap.confusion(np.zeros((4, 4)), no_rice, 0.5))
    assert silent["f1"] is None and silent["recall"] is None and silent["precision"] is None and silent["iou"] is None
    assert "no reference rice pixel" in silent["undefined"] and silent["accuracy"] == 1.0
    noisy = cap.scores(cap.confusion(np.full((4, 4), 0.9), no_rice, 0.5))
    assert noisy["f1"] == 0.0 and noisy["recall"] is None and noisy["precision"] == 0.0
    rice = np.ones((4, 4), dtype=np.uint8)
    all_non_rice = cap.pooled([np.zeros((4, 4)), np.zeros((4, 4))], [rice, no_rice], 0.5)
    assert all_non_rice["f1"] == 0.0 and all_non_rice["recall"] == 0.0 and all_non_rice["precision"] is None
    assert all_non_rice["accuracy"] == 0.5
    with pytest.raises(ValueError):
        cap.pooled([], [], 0.5)


def test_threshold_and_epoch_selection_rules():
    label = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    prob = np.array([[0.7, 0.2, 0.2, 0.2]])  # perfect for every threshold in (0.2, 0.7]
    selected = cap.select_threshold([prob], [label])
    assert selected["threshold"] == 0.5 and selected["f1"] == 1.0  # ties -> closest to 0.5
    assert len(selected["sweep"]) == 17 and selected["sweep"][0]["threshold"] == 0.1
    with pytest.raises(ValueError, match="undefined"):
        cap.select_threshold([np.zeros((2, 2))], [np.zeros((2, 2), dtype=np.uint8)])
    curve = [{"epoch": 1, "dev_f1": None}, {"epoch": 2, "dev_f1": 0.6}, {"epoch": 3, "dev_f1": 0.6}, {"epoch": 4, "dev_f1": 0.5}]
    assert cap.select_epoch(curve) == 2


# ---------------------------------------------------------------------------------------- split


def test_fit_dev_split_is_deterministic_disjoint_by_region_and_has_two_rice_dev_chips():
    first = cap.split_fit_dev(_rows())
    assert first == cap.split_fit_dev(list(reversed(_rows())))
    assert first["attempt"] == 0 and first["target_dev_chips"] == 8
    assert first["development_regions"] == ["PH-01", "PH-02", "PH-15", "PH-40"]
    assert len(first["development"]) == 8 and len(first["fit"]) == 24 and first["development_rice_chips"] == 6
    assert not set(first["fit"]) & set(first["development"])
    assert not set(first["development_regions"]) & set(first["fit_regions"])
    cap.check_roles({"fit": first["fit"], "development": first["development"], "validation": ["ph_240_427_2021"]})
    with pytest.raises(ValueError, match="both"):
        cap.check_roles({"fit": ["a_2019"], "development": ["a_2019"]})
    with pytest.raises(ValueError, match="development example"):
        cap.check_roles({"validation": ["ph_96_766_2021"]})


def test_byod_group_keeps_a_sites_years_together_and_split_is_deterministic():
    assert cap.byod_group("site_a_2019", {}) == "site_a" and cap.byod_group("x", {"group": "g1"}) == "g1"
    ids = [f"s{i}_{y}" for i in range(6) for y in (2019, 2020)]
    groups = {i: cap.byod_group(i, {}) for i in ids}
    rice = {i: not i.startswith("s0") for i in ids}
    nonrice = {i: True for i in ids}
    split = cap.byod_split(ids, groups, rice, nonrice)
    assert split == cap.byod_split(ids, groups, rice, nonrice)
    # Codex P1 on 8fbcaa6: every chip holding some rice (and some non-rice) must still split; the head trains per pixel.
    every_chip_has_rice = cap.byod_split(ids, groups, {i: True for i in ids}, nonrice)
    assert all(every_chip_has_rice.values())
    with pytest.raises(cap.ChipError, match="non-rice in fit"):
        cap.byod_split(ids, groups, {i: True for i in ids}, {i: False for i in ids})
    for a, b in (("fit", "development"), ("fit", "test"), ("development", "test")):
        assert not {groups[i] for i in split[a]} & {groups[i] for i in split[b]}
    with pytest.raises(cap.ChipError, match="three groups"):
        cap.byod_split(ids[:4], groups, rice, nonrice)


# ---------------------------------------------------------------------------------------- baseline B recipe


def test_m1_features_match_the_study_recipe_fixture():
    values = np.stack([np.full((6, 4, 4), float(t), dtype=np.float32) + np.arange(6, dtype=np.float32)[:, None, None]
                       for t in range(24)])
    chip = {"S2": values, "S2_valid": np.ones((24, 4, 4), dtype=np.uint8)}
    x = cap.m1_features(chip)
    assert x.shape == (16, 70) and len(cap.M1_NAMES) == 70
    np.testing.assert_allclose(x[0, :7], [11.5, np.std(np.arange(24)), 0, 23, 2.3, 11.5, 20.7], atol=1e-6)
    np.testing.assert_allclose(x[0, 42:48], np.arange(6) + 2.5)
    np.testing.assert_array_equal(x[0, -4:], 1)
    chip["S2"] = np.full_like(values, np.nan)
    chip["S2_valid"] = np.zeros_like(chip["S2_valid"])
    raw = cap.m1_features(chip)
    fill = cap.fit_fill(x)
    filled = cap.fill_features(raw, fill)
    assert np.isnan(raw[:, :66]).all() and (raw[:, -4:] == 0).all()
    np.testing.assert_allclose(filled[0, :66], fill[:66])


def test_pixel_sample_is_seeded_and_class_stratified():
    labels = [np.array([0] * 90 + [1] * 10), np.array([0] * 50 + [255] * 50)]
    a = cap.pixel_sample(labels, 40, 7)
    b = cap.pixel_sample(labels, 40, 7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert sum(len(x) for x in a) == 40
    assert int((labels[0][a[0]] == 1).sum()) == 10  # the small rice stratum is fully used
    assert not (labels[1][a[1]] == 255).any()


# ---------------------------------------------------------------------------------------- BYOD schema


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(S2=c["S2"][:12]), "S2 has shape"),
        (lambda c: c.update(S2=c["S2"].astype(np.int16)), "dtype"),
        (lambda c: c.update(S2_valid=c["S2_valid"] * 2), "only 0 and 1"),
        (lambda c: c.update(S2=c["S2"] * 10000), "multiply by 0.0001"),
        (lambda c: c.update(S2=np.where(np.arange(128) == 3, np.nan, c["S2"]).astype(np.float32)), "non-finite"),
        (lambda c: c.update(label=c["label"][:64]), "label has shape"),
        (lambda c: c.update(label=np.full((128, 128), 3, dtype=np.uint8)), "outside 0"),
        (lambda c: c.pop("label"), "missing key 'label'"),
        (lambda c: c.pop("S2_valid"), "missing key 'S2_valid'"),
    ],
)
def test_byod_schema_validation_refuses_with_an_actionable_message(mutate, message):
    chip = _chip(3)
    mutate(chip)
    with pytest.raises(cap.ChipError, match=re.escape(message)):
        cap.validate_chip(chip, name="user.h5")


def test_unlabelled_chips_are_accepted_for_prediction_only_and_extra_keys_are_ignored():
    chip = _chip(4)
    chip.pop("label")
    chip["label_raw"] = np.full((128, 128), 9)
    assert cap.validate_chip(chip, require_label=False)["label"] is None


def test_georeference_reads_the_datasets_own_affine_crs_and_group_id_attributes():
    # 36c5c54 BYOD run: the dataset's chips store affine/crs/group_id, so every BYOD chip came out ungeoreferenced.
    real = {"affine": "[20.0, 0.0, 614400.0, 0.0, -20.0, 1095680.0]", "crs": "EPSG:32651", "group_id": b"ph_240_427"}
    assert cap.georeference_attrs(real) == {"x_min": 614400.0, "y_max": 1095680.0, "group": "ph_240_427"}
    assert cap.georeference_attrs({"x_min": 1.0, "y_max": 2.0, "group": "g"}) == {"x_min": 1.0, "y_max": 2.0, "group": "g"}
    assert cap.georeference_attrs({**real, "crs": "EPSG:4326"}) == {"group": "ph_240_427"}  # never mislocated
    assert cap.georeference_attrs({**real, "affine": "[10.0, 0.0, 1.0, 0.0, -10.0, 2.0]"}) == {"group": "ph_240_427"}
    assert cap.georeference_attrs({"affine": "not json", "crs": "EPSG:32651"}) == {}
    # Codex P2 on 42d8f40: plain coordinates must be finite and must not come with an incompatible CRS.
    assert cap.georeference_attrs({"x_min": 1.0, "y_max": 2.0, "crs": "EPSG:4326"}) == {}
    assert cap.georeference_attrs({"x_min": float("nan"), "y_max": 2.0}) == {}
    assert cap.georeference_attrs({"x_min": 1.0, "y_max": 2.0, "crs": b"EPSG:32651"}) == {"x_min": 1.0, "y_max": 2.0}
    assert cap.georeference_attrs({**real, "affine": "[20.0, 0.0, NaN, 0.0, -20.0, 2.0]"}) == {"group": "ph_240_427"}
    # Notebook review RC-S3: an affine stored as a numeric HDF5 attribute is read like the JSON form.
    numeric = {**real, "affine": np.array([20.0, 0.0, 614400.0, 0.0, -20.0, 1095680.0])}
    assert cap.georeference_attrs(numeric) == {"x_min": 614400.0, "y_max": 1095680.0, "group": "ph_240_427"}


def test_hdf5_reader_refuses_an_oversized_declared_shape_before_reading_it(tmp_path):
    # Codex P2 on 8fbcaa6: a small sparse file declaring a huge dataset must be refused, not materialised.
    h5py = pytest.importorskip("h5py")
    chip = _chip(1)
    path = tmp_path / "huge_2020.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("S2", shape=(24, 6, 8192, 8192), dtype="f4", chunks=(1, 1, 64, 64))  # ~38 GiB declared
        handle["S2_valid"] = chip["S2_valid"]
        handle["label"] = chip["label"]
    assert path.stat().st_size < 1_000_000
    with pytest.raises(cap.ChipError, match="checked before reading"):
        cap.read_chip_h5(path)


def test_byod_table_keeps_metric_columns_when_the_first_chip_is_unlabelled(tmp_path):
    # Codex P2 on 8fbcaa6: columns came from the first row only, dropping every later chip's metrics.
    rows = [{"chip": "a_2020", "georeferenced": True}, {"chip": "b_2020", "georeferenced": True, "f1": 0.5, "tp": 3}]
    assert cap.table_columns(rows) == ["chip", "georeferenced", "f1", "tp"]
    cap.write_csv(tmp_path / "t.csv", rows, cap.table_columns(rows))
    assert (tmp_path / "t.csv").read_text().splitlines()[0] == "chip,georeferenced,f1,tp"


def test_hdf5_reader_and_zipped_layout_verify_member_hashes(tmp_path):
    h5py = pytest.importorskip("h5py")
    chip = _chip(5)
    path = tmp_path / "samples" / "train" / "c_1_2020.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        for key in ("S2", "S2_valid", "label"):
            handle[key] = chip[key]
        handle["AlphaEarth"] = np.zeros((64, 128, 128), dtype=np.float32)
        handle.attrs["x_min"] = 250880.0
        handle.attrs["y_max"] = 1896960.0
    loaded = cap.read_chip_h5(path)
    assert loaded["attrs"] == {"x_min": 250880.0, "y_max": 1896960.0}
    np.testing.assert_array_equal(loaded["label"], chip["label"])
    data = path.read_bytes()
    row = {"path": "samples/train/c_1_2020.h5", "archive": "data-train.zip", "sha256": hashlib.sha256(data).hexdigest()}
    assert cap.locate_member(tmp_path, row, tmp_path / "cache") == path  # flat expanded layout
    import zipfile

    zipped = tmp_path / "z"
    zipped.mkdir()
    with zipfile.ZipFile(zipped / "data-train.zip", "w") as archive:
        archive.writestr(row["path"], data)
    extracted = cap.locate_member(zipped, row, tmp_path / "cache")
    assert extracted.read_bytes() == data
    with pytest.raises(ValueError, match="SHA-256"):
        cap.locate_member(zipped, {**row, "sha256": "0" * 64}, tmp_path / "cache2")
    (tmp_path / "bad.h5").write_bytes(b"not hdf5")
    with pytest.raises(cap.ChipError, match="not a readable HDF5"):
        cap.read_chip_h5(tmp_path / "bad.h5")


# ---------------------------------------------------------------------------------------- artifact and file round trips


def test_head_artifact_export_and_reload_round_trip_with_a_tiny_synthetic_head(tmp_path):
    pytest.importorskip("safetensors")
    rng = np.random.default_rng(0)
    shapes = {"conv1.weight": (4, 8, 1, 1), "conv1.bias": (4,), "conv2.weight": (2, 4, 1, 1), "conv2.bias": (2,)}
    params = {name: rng.normal(size=shape).astype(np.float32) for name, shape in shapes.items()}
    out = cap.save_head(params, tmp_path / "head", {"base_revision": cap.pp.MODEL_REVISION, "threshold": 0.45,
                                                    "feature_layer_zero_based": 11})
    loaded, manifest = cap.load_head(out, expect={"base_revision": cap.pp.MODEL_REVISION, "feature_layer_zero_based": 11})
    assert manifest["threshold"] == 0.45 and manifest["in_channels"] == 8 and manifest["hidden"] == 4
    features = rng.normal(size=(2, 8, 3, 3)).astype(np.float32)
    np.testing.assert_array_equal(cap.head_forward_numpy(params, features), cap.head_forward_numpy(loaded, features))
    with pytest.raises(ValueError, match="base_revision"):
        cap.load_head(out, expect={"base_revision": "main"})
    weights = out / cap.ARTIFACT_WEIGHTS
    blob = bytearray(weights.read_bytes())
    blob[-1] ^= 1
    weights.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="SHA-256"):
        cap.load_head(out)
    with pytest.raises(ValueError, match="head tensors"):
        cap.save_head({"conv1.weight": params["conv1.weight"]}, tmp_path / "bad", {})


def test_geotiff_round_trip_keeps_values_georeference_and_nodata(tmp_path):
    pytest.importorskip("tifffile")
    pred = np.zeros((128, 128), dtype=np.uint8)
    pred[10:20, 30:40] = 1
    pred[:2] = 255
    cap.write_geotiff(tmp_path / "p.tif", pred, x_min=250880.0, y_max=1896960.0)
    array, geo = cap.read_geotiff(tmp_path / "p.tif")
    np.testing.assert_array_equal(array, pred)
    assert geo == {"nodata": 255, "x_min": 250880.0, "y_max": 1896960.0, "pixel": 20.0, "epsg": 32651}
    cap.write_geotiff(tmp_path / "plain.tif", pred, x_min=None, y_max=None)
    assert cap.read_geotiff(tmp_path / "plain.tif")[1] == {"nodata": 255}


def test_csv_and_zip_round_trips_with_tamper_detection(tmp_path):
    rows = [{"system": "A", "threshold": None, "f1": 0.0, "tp": 0},
            {"system": "B", "threshold": 0.45, "f1": 0.812345678, "tp": 7}]
    cap.write_csv(tmp_path / "out" / "m.csv", rows)
    back = cap.read_csv(tmp_path / "out" / "m.csv")
    assert back[0] == {"system": "A", "threshold": None, "f1": 0.0, "tp": 0}
    assert back[1]["f1"] == pytest.approx(0.812346) and back[1]["tp"] == 7
    with pytest.raises(ValueError, match="empty"):
        cap.write_csv(tmp_path / "e.csv", [])
    cap.write_json(tmp_path / "out" / "r.json", {"x": np.float32(1.5)})
    digests = cap.build_zip(tmp_path / "results.zip", tmp_path / "out", (tmp_path / "out").rglob("*"))
    assert set(digests) == {"m.csv", "r.json"}
    assert cap.verify_zip(tmp_path / "results.zip") == {"members": 2, "verified": True}
    first = (tmp_path / "results.zip").read_bytes()
    cap.build_zip(tmp_path / "again.zip", tmp_path / "out", (tmp_path / "out").rglob("*"))
    assert (tmp_path / "again.zip").read_bytes() == first  # deterministic bytes
    import zipfile

    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as archive:
        archive.writestr("m.csv", "changed")
        archive.writestr("checksums.json", json.dumps(digests))
    with pytest.raises(ValueError, match="integrity"):
        cap.verify_zip(tmp_path / "bad.zip")


# ---------------------------------------------------------------------------------------- notebook, generator, carried source


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _carried(notebook: dict) -> tuple[dict, dict]:
    cell = next(c for c in notebook["cells"] if "CARRIED_FILES = " in _source(c))
    body = ast.parse(_source(cell)).body
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in body if isinstance(n, ast.Assign)}
    return literals["CARRIED_FILES"], literals["CARRIED_HASHES"]


def test_generator_parity_the_committed_notebook_is_byte_identical():
    assert NOTEBOOK.read_text(encoding="utf-8") == gen.build()
    assert gen.main(["--check"]) == 0


def test_notebook_schema_metadata_and_output_free_cells_compile():
    notebook = _notebook()
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    dimer = notebook["metadata"]["dimer"]
    assert (dimer["notebook_profile"], dimer["notebook_mode"], dimer["standalone"]) == ("E2E", "WORKSHOP", True)
    assert dimer["release_status"] == "Candidate" and dimer["requires_dimer_worker"] is False
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile(_source(cell), cell["id"], "exec")
            # A single huge carrier line froze Colab's editor (layoutlm #10); the flood capstone guards the same way.
            assert max(map(len, _source(cell).splitlines())) <= 2000, cell["id"]
            first = _source(cell).splitlines()[0]
            if first.startswith("# @title Infrastructure"):
                assert cell["metadata"]["cellView"] == "form"


def test_carried_files_match_the_repository_and_their_pinned_hashes():
    files, hashes = _carried(_notebook())
    assert set(files) == set(hashes)
    for name, text in files.items():
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == hashes[name], name

    def repo(path: Path) -> str:
        return path.read_bytes().decode("utf-8").replace("\r\n", "\n")

    assert files["capstone.py"] == repo(TOOLS / "capstone.py")
    assert files["prithvi_pipeline.py"] == repo(ROOT / "src" / "prithvi_eo_feature_extraction_pipeline" / "pipeline.py")
    assert files["requirements.txt"] == repo(LOCK)
    manifest = "weights/prithvi-eo-2.0-300m/dimer-base-manifest.json"
    assert files[manifest] == repo(ROOT / manifest)
    data = json.loads(files["data_manifest.json"])
    assert data["index_sha256"] == cap.INDEX_SHA256 and data["slug"] == cap.DATASET_SLUG
    source = json.loads(files["source.json"])
    assert source["carried_sha256"]["capstone.py"] == hashes["capstone.py"]


def test_isolated_environment_bootstrap_is_pinned_and_nothing_installs_into_the_kernel():
    notebook = _notebook()
    code = "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code")
    install = next(_source(c) for c in notebook["cells"] if "UV_URL = " in _source(c))
    assert gen.UV_WHEEL["sha256"] in install and str(gen.UV_WHEEL["bytes"]) in install
    assert "'--managed-python', '--python', '3.12.12'" in install
    assert "'--require-hashes', '--only-binary', ':all:'" in install
    assert "ENV['MPLBACKEND'] = 'Agg'" in install
    assert "pip install" not in code.replace("'pip', 'install'", "")  # no in-kernel install
    assert "RUN_FINETUNE = False" in code and "USE_BYOD = False" in code
    assert "git clone" not in code and "kaggle.json" not in code


def test_lock_is_fully_hashed_and_covers_the_runtime_imports():
    text = LOCK.read_text(encoding="utf-8")
    entries = re.findall(r"^([A-Za-z0-9_.\-]+)==([^ \\\n]+)", text, flags=re.M)
    names = {name.lower() for name, _version in entries}
    for required in ("torch", "terratorch", "numpy", "safetensors", "huggingface-hub", "h5py", "lightgbm", "kagglehub",
                     "tifffile", "matplotlib"):
        assert required in names, required
    blocks = re.split(r"\n(?=[A-Za-z0-9_.\-]+==)", text.split("\n", 2)[2])
    assert all("--hash=sha256:" in block for block in blocks if "==" in block.split("\n")[0])
    pins = (TOOLS / "requirements-rice-capstone.in").read_text(encoding="utf-8")
    for line in pins.splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            assert (name.lower(), version) in {(n.lower(), v) for n, v in entries}, line


def test_learner_text_attribution_disclosure_and_citations():
    notebook = _notebook()
    markdown = "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert "privately staged" not in markdown.lower()
    assert "AI Assistance Disclosure" in markdown
    for component in ("10.5281/zenodo.15124936", "Contains modified Copernicus Sentinel data 2019–2021", "AlphaEarth",
                      "geoBoundaries PHL ADM1", "Dataset compilation"):
        assert component in markdown, component
    assert "0.760" in markdown and "0.928" in markdown
    references = markdown.split("## References", 1)[1]
    body = markdown.split("## References", 1)[0]
    cited = set(re.findall(r"\(([A-Z][A-Za-zÀ-ÿ\-]+)(?: et al\.| & [A-Z][A-Za-z]+)?, (\d{4})\)", body))
    cited |= set(re.findall(r"([A-Z][A-Za-zÀ-ÿ\-]+) et al\. \((\d{4})\)", body))
    assert cited, "expected in-text citations"
    for author, year in cited:
        assert re.search(rf"^{author},.*\({year}\)", references, flags=re.M), (author, year)
    for entry in [e for e in references.split("\n\n") if re.match(r"^[A-Z][^*]+\(\d{4}\)", e)]:
        assert "https://doi.org/" in entry, entry[:60]
        surname = entry.split(",", 1)[0]
        assert surname in markdown.split("## References", 1)[0], surname
