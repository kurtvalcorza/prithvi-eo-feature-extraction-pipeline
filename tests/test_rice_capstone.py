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


def test_pr18_round2_findings(tmp_path):
    # 1. Codex on 7c8796d: --stage byod without --input cleared earlier BYOD results before the argument error.
    ctx = cap.Context(tmp_path / "run")
    (ctx.work / "receipts").mkdir(parents=True, exist_ok=True)
    for name in ("prepare", "baseline", "features", "train-head", "activity", "evaluate", "export", "reload", "byod"):
        (ctx.work / "receipts" / f"{name}.json").write_text("{}")
    (ctx.out / "byod").mkdir()
    (ctx.out / "byod_results.zip").write_text("kept")
    with pytest.raises(SystemExit):
        cap.main(["--root", str(tmp_path / "run"), "--stage", "byod"])
    assert (ctx.out / "byod").exists() and (ctx.out / "byod_results.zip").exists()
    assert (ctx.work / "receipts" / "byod.json").exists()
    # 2. restarting report withdraws the results_zip attestation but keeps reload's checks.
    (ctx.work / "receipts" / "report.json").write_text("{}")
    cap.write_json(ctx.out / "verification.json", {"binary_decisions_identical": True,
                                                   "results_zip": {"members": 33, "verified": True}})
    cap.start_stage(ctx, "report")
    assert cap.read_json(ctx.out / "verification.json") == {"binary_decisions_identical": True}


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
    # RICE-m5: the exported rice scores are float32 GeoTIFFs with no nodata value, so they keep every pixel exactly.
    score = np.random.default_rng(1).uniform(size=(128, 128))
    cap.write_geotiff(tmp_path / "s.tif", score, x_min=250880.0, y_max=1896960.0, nodata=None, dtype=np.float32)
    array, geo = cap.read_geotiff(tmp_path / "s.tif")
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, score.astype(np.float32))
    assert geo == {"nodata": None, "x_min": 250880.0, "y_max": 1896960.0, "pixel": 20.0, "epsg": 32651}


def test_review_rice_m5_export_writes_scored_all_pixel_and_score_maps():
    # RICE-m5 (review 2026-10-11): results.zip bundles outputs/ only, so the scores and an unmasked decision map must be
    # written there beside the label-masked map that reproduces the evaluation; the reload checks all three.
    source = (TOOLS / "capstone.py").read_text(encoding="utf-8")
    export = source[source.index("def stage_export"):source.index("def stage_reload")]
    reload = source[source.index("def stage_reload"):source.index("def stage_report")]
    for name in ("_C.tif", "_C_all_pixels.tif", "_C_score.tif"):
        assert f'ctx.out / "predictions" / f"{{cid}}{name}"' in export, name
        assert name in reload, name
    assert "score_and_all_pixel_geotiffs_match" in reload and "and extra_ok" in reload


def test_review_knd4_the_run_states_its_own_answer_with_its_evidence_level():
    def row(system, f1):
        return {"system": system, "f1": f1, "iou": None if f1 is None else f1 / 2, "precision": None, "recall": 0.5}

    lines = "\n".join(cap.answer_lines([row("A all-non-rice", 0.0), row("B LightGBM", 0.758),
                                         row("C frozen Prithvi + head", 0.601)]))
    assert "did not reach a higher pooled rice F1 than the spectral-temporal LightGBM pipeline (B, F1 0.758)" in lines
    assert "(C, F1 0.601)" in lines and "(A) scored F1 0.000" in lines and "C-FT" not in lines
    assert "**Evidence level:**" in lines and "nationally" in lines and "precision undefined" in lines
    lines = "\n".join(cap.answer_lines([row("A all-non-rice", 0.0), row("B LightGBM", 0.5),
                                         row("C frozen Prithvi + head", 0.7), row("C-FT fine-tuned (opt-in)", 0.6)]))
    assert "(C, F1 0.700) reached a higher pooled rice F1 than" in lines and "(C-FT) scored F1 0.600" in lines
    verdict = next(line for line in cap.answer_lines([row("A all-non-rice", 0.0), row("B LightGBM", None),
                                                       row("C frozen Prithvi + head", 0.1)])
                   if line.startswith("**Answer"))
    assert "did not" not in verdict


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
    references = markdown.split("## References", 1)[1].split("**Data and software sources:**", 1)[0]
    body = markdown.split("## References", 1)[0]
    # NOTEBOOK_SPEC 2.4 DOC14-DOC16: IEEE numbered citations, one to one with entries, each entry with a resolvable link.
    assert not re.search(r"\([A-Z][\w\s.&-]*et al\., \d{4}\)|\([A-Z][a-z]+ & [A-Z][a-z]+, \d{4}\)", markdown)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    entries = dict(re.findall(r"^\[(\d+)\] (.+)$", references, flags=re.M))
    assert cited and cited == {int(n) for n in entries}, (sorted(cited), sorted(entries))
    for number, entry in entries.items():
        assert "Available: https://" in entry, number
        if "doi:" in entry:
            assert "https://doi.org/" in entry, number


def test_review_findings_2026_10_11_learner_text():
    # The rice capstone review of a00ab42 (RICE-m1..m4, m7, S2) and NOTEBOOK_SPEC 2.4.
    notebook = _notebook()
    cells = [(c["cell_type"], _source(c)) for c in notebook["cells"]]
    first_code = next(i for i, (kind, _) in enumerate(cells) if kind == "code")
    opening = "\n".join(text for _, text in cells[:first_code])
    dimer = notebook["metadata"]["dimer"]
    assert (dimer["notebook_spec"], dimer["notebook_kind"]) == ("2.4", "CAPSTONE")  # m7 / KND1
    assert "> **⚠️ Notice.** For research, training and evaluation only." in opening  # m7 / DOC17-DOC21
    assert "Data you bring stays in your runtime" in opening and '"as is", without warranty' in opening
    assert "### Problem statement" in opening and "### Data context" in opening  # DOC1-DOC8 / KND4
    assert re.search(r"about 10 minutes.*Tesla T4.*`33037307`", opening)  # m3 / KND9, UX12
    assert not re.search(r"(?m)^\s*[23]\.\s+\*\*(Build|Train)\*\*", opening)  # m4
    for _kind, text in cells:  # m2: every cell that names lesson L3/L3B links it
        if re.search(r"\bL3B?\b", text):
            assert "kaggle.com/code/kurtvalcorza/rice-from-space-lesson-3" in text, text[:80]
    for _kind, text in cells:  # m1: quoted outside numbers name their source revision in the same cell
        if re.search(r"(?i)not re-run here", text):
            assert re.search(r"`[0-9a-f]{8}`|`[0-9a-f]{8}…`", text), text[:80]
    byod = next(text for kind, text in cells if kind == "markdown" and "USE_BYOD" in text)
    assert "Run this cell" in byod and "do not need to Run all again" in byod  # S2
    ids = [c["id"] for c in notebook["cells"]]
    assert ids[:3] == ["rice-00", "rice-problem", "rice-01"] and ids[-1] == "rice-35"  # earlier cell ids are kept


# ---------------------------------------------------------------------------------------- RICE-m8 (2026-10-12 browser run)


def _byod_ready(tmp_path: Path):
    """A run directory whose BYOD prerequisites are met, holding outputs of an earlier BYOD attempt."""
    ctx = cap.Context(tmp_path)
    (ctx.work / "receipts").mkdir(parents=True, exist_ok=True)
    (ctx.work / "receipts" / "reload.json").write_text("{}")
    (ctx.out / "byod").mkdir()
    (ctx.out / "byod" / "byod_canonical_head.csv").write_text("chip\nold\n")
    (ctx.out / "byod_results.zip").write_bytes(b"old")
    return ctx


def test_review_rice_m8_a_refused_byod_upload_is_one_message_without_a_traceback(tmp_path, monkeypatch, capsys):
    # The browser run of a533d55b refused a reflectance x 10 000 upload correctly, but the cell showed the stage's
    # traceback twice and a red RuntimeError. The refusal is now one message and a distinct exit code.
    h5py = pytest.importorskip("h5py")
    ctx = _byod_ready(tmp_path)
    chip = _chip(3)
    path = tmp_path / "rice_byod_MALFORMED_reflectance_x10000_2019.h5"
    with h5py.File(path, "w") as handle:
        handle["S2"] = chip["S2"] * 10000.0
        handle["S2_valid"] = chip["S2_valid"]
        handle["label"] = chip["label"]
    monkeypatch.setattr(cap, "_torch_setup", lambda *a, **k: None)  # CI has no torch; refusal precedes any model
    code = cap.main(["--root", str(tmp_path), "--stage", "byod", "--input", str(path)])
    out = capsys.readouterr()
    text = out.out + out.err
    assert code == cap.REFUSED_EXIT == 3 and code != 2  # 2 is argparse's usage-error code
    assert text.count("Refused:") == 1 and "Traceback" not in text
    assert text.count(f"{path.name}: S2 maximum") == 1 and "multiply by 0.0001" in text
    assert not (ctx.out / "byod").exists() and not (ctx.out / "byod_results.zip").exists()  # nothing stale to show
    assert not (ctx.work / "receipts" / "byod.json").exists()


def test_review_rice_m8_genuine_byod_failures_still_raise(tmp_path, monkeypatch):
    _byod_ready(tmp_path)

    def broken(*_args, **_kwargs):
        raise KeyError("a bug, not an input refusal")

    monkeypatch.setattr(cap, "stage_byod", broken)
    with pytest.raises(KeyError):
        cap.main(["--root", str(tmp_path), "--stage", "byod", "--input", str(tmp_path / "x.h5")])


def _run_stage(tmp_path: Path, exit_code: int, lines: list[str]):
    """The notebook's own run_stage, driving a stand-in capstone.py that prints `lines` and exits with `exit_code`."""
    import os
    import subprocess

    cell = next(_source(c) for c in _notebook()["cells"] if "def run_stage(" in _source(c))
    tree = ast.parse(cell)
    wanted = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "run_stage")
              or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REFUSED_EXIT")]
    assert len(wanted) == 2
    (tmp_path / "capstone.py").write_text(
        "import sys\n" + "".join(f"print({line!r})\n" for line in lines) + f"sys.exit({exit_code})\n")
    namespace = {"ROOT": tmp_path, "PYTHON": Path(sys.executable), "ENV": dict(os.environ), "subprocess": subprocess}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "run_stage", "exec"), namespace)
    return namespace["run_stage"]


def test_review_rice_m8_run_stage_streams_each_line_once_and_separates_refusals(tmp_path, capsys):
    refusal = ["Refused: 1 of 1 file(s) failed validation:", "  bad_2019.h5: S2 maximum 8608.5 exceeds 2.0"]
    run_stage = _run_stage(tmp_path, 3, refusal)
    assert run_stage("byod", "--input", "bad_2019.h5", refusal_ok=True) is False
    shown = capsys.readouterr().out
    assert shown.count("Refused:") == 1 and shown.count("S2 maximum") == 1 and "Traceback" not in shown
    with pytest.raises(RuntimeError, match="byod failed with exit 3"):  # without refusal_ok a refusal still stops
        run_stage("byod")
    capsys.readouterr()

    crash = ["Traceback (most recent call last):", "KeyError: 'x'", "x" * 2500]
    run_stage = _run_stage(tmp_path, 1, crash)
    with pytest.raises(RuntimeError, match="prepare failed with exit 1"):  # every other failure still stops Run all
        run_stage("prepare", refusal_ok=True)
    shown = capsys.readouterr().out
    assert shown.count("Traceback (most recent call last):") == 1  # streamed once; the log tail is not re-printed
    assert "x" * 2500 not in shown and "1 over-long line(s) not shown" in shown
    assert "x" * 2500 in (tmp_path / "prepare.log").read_text(encoding="utf-8")
    assert _run_stage(tmp_path, 0, ["ok"])("report") is True


def test_review_rice_m8_the_byod_cell_shows_results_only_after_a_completed_stage():
    cells = [(c["cell_type"], _source(c)) for c in _notebook()["cells"]]
    byod = next(text for kind, text in cells if kind == "code" and "USE_BYOD = " in text)
    branch = byod.split("if run_stage('byod'", 1)
    assert len(branch) == 2 and "refusal_ok=True" in branch[1].split("\n", 1)[0]
    accepted, refused = branch[1].split("\n    else:", 1)
    for display_call in ("show_table(", "byod_summary.json", "FileLink("):
        assert display_call in accepted and display_call not in branch[0] and display_call not in refused
    assert "run this cell again (no Run all needed)" in refused
    trouble = next(text for kind, text in cells if kind == "markdown" and text.startswith("## Troubleshooting"))
    assert "`Refused:`" in trouble and "you do not need to Run all again" in trouble


# ---------------------------------------------------------------------------------------- issue #22 (peer review, 2026-10-12)


def _cells() -> list[tuple[str, str]]:
    return [(c["cell_type"], _source(c)) for c in _notebook()["cells"]]


def _markdown_with(anchor: str) -> str:
    return next(text for kind, text in _cells() if kind == "markdown" and anchor in text)


def test_issue_22_m1_the_frame_activity_is_a_fixed_recipe_comparison_without_causal_claims():
    section = _markdown_with("## 6. Change one thing")
    assert "fixed-recipe, development-only comparison" in section and "not an equally tuned comparison" in section
    assert "How far below T = 4" not in section and "match it or exceed it" in section  # neutral prediction prompt
    after = _markdown_with("**Observe and explain:**")
    assert "more robust than its size" not in after and "supports the season-based explanation" not in after
    assert ("Under a fixed training recipe and the epoch count and threshold selected for the four-frame system"
            .lower() in after.lower())
    assert "observation coverage" in after and "settings better suited to the four-frame system" in after
    conclusion = _markdown_with("## 11. Conclude with evidence")
    assert "consistent with rice being a seasonal signal" not in conclusion
    source = (TOOLS / "capstone.py").read_text(encoding="utf-8")
    assert '"comparison": ACTIVITY_SCOPE' in source and "not an equally tuned comparison" in cap.ACTIVITY_SCOPE
    assert "without claiming a cause" in source
    spec = (ROOT / "docs" / "philippines-rice-mapping-capstone-spec.md").read_text(encoding="utf-8")
    assert "Demonstrates that rice is identified by its season" not in spec and "fixed-recipe" in spec


@pytest.mark.parametrize("case, after, reason", [
    ("excessive difference", lambda p: p + 0.01, "exceeds"),
    ("non-finite difference", lambda p: np.where(np.arange(p.size).reshape(p.shape) == 0, np.nan, p), "not finite"),
    ("binary disagreement", lambda p: np.where(np.isclose(p, 0.5), 0.5 - 1e-6, p), "decisions"),
])
def test_issue_22_m3_head_serialization_parity_rejects_each_failure_mode(case, after, reason):
    before = np.linspace(0.0, 1.0, 2 * 4 * 4).reshape(2, 4, 4)
    before[0, 0, 1] = 0.5  # exactly at the threshold, so a 1e-6 change flips the decision but stays within tolerance
    record = cap.head_serialization_parity(before, after(before), 0.5)
    assert record["passed"] is False, case
    with pytest.raises(cap.ReloadParityError, match=reason) as raised:
        cap.require_head_serialization_parity(record)
    assert not isinstance(raised.value, cap.ChipError)  # never a learner "Refused:" input refusal (exit 3)
    assert "not a problem with your files" in str(raised.value) and "no BYOD result bundle" in str(raised.value)


def test_issue_22_m3_head_serialization_parity_accepts_identical_scores_and_names_its_scope():
    before = np.random.default_rng(0).uniform(size=(2, 8, 8))
    record = cap.head_serialization_parity(before, before + 1e-7, 0.5)
    assert record["passed"] and record["finite"] and record["binary_decisions_identical"]
    assert record["max_abs_difference"] == pytest.approx(1e-7) and record["tolerance"] == cap.RELOAD_TOLERANCE
    assert "head-serialization parity" in record["check"] and "not the canonical fresh-process" in record["check"]
    cap.require_head_serialization_parity(record)  # passes silently


def _byod_standins(tmp_path: Path, monkeypatch, *, corrupt_reload=None):
    """CPU stand-ins for the model parts of stage_byod (no torch): random 8-channel 'encoder features', a tiny head
    evaluated with head_forward_numpy, and a 'trained' head drawn from a seeded generator. Real code paths: chip
    validation, the site split, threshold selection, save_head/load_head (safetensors), GeoTIFFs, the bundle."""
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("safetensors")
    pytest.importorskip("tifffile")
    ctx = cap.Context(tmp_path / "run")
    (ctx.work / "receipts").mkdir(parents=True, exist_ok=True)
    (ctx.work / "receipts" / "reload.json").write_text("{}")
    rng = np.random.default_rng(22)
    shapes = cap.head_shapes(8, 4)

    def params(seed):
        g = np.random.default_rng(seed)
        return {k: g.normal(size=s).astype(np.float32) for k, s in shapes.items()}

    cap.save_head(params(1), ctx.out / "head_adapter", {"base_revision": cap.pp.MODEL_REVISION, "threshold": 0.5})
    cap.write_json(ctx.out / "selection.json", {"C": {"threshold": 0.5, "selected_epoch": 3, "config": cap.HEAD_CONFIG}})
    sources = tmp_path / "chips"
    sources.mkdir()
    for i in range(6):
        chip = _chip(40 + i)
        with h5py.File(sources / f"site{i}_2020.h5", "w") as handle:
            handle["S2"], handle["S2_valid"], handle["label"] = chip["S2"], chip["S2_valid"], chip["label"]
            handle.attrs["group_id"] = f"site{i}"
    features = {}

    def encode(_pipe, x):
        key = float(np.nanmean(x))
        features.setdefault(key, rng.normal(size=(1, 8, 8, 8)).astype(np.float32))
        return features[key]

    def probabilities(_torch, head, x, device="cpu"):
        margin = cap.head_forward_numpy(head, np.asarray(x, dtype=np.float32))
        up = np.kron(margin, np.ones((16, 16)))
        return 1.0 / (1.0 + np.exp(-up.astype(np.float64)))

    real_load = cap.load_head

    def load(path, **kwargs):
        loaded, manifest = real_load(path, **kwargs)
        if corrupt_reload and path.name == "byod_head_adapter":
            loaded = corrupt_reload(dict(loaded))
        return loaded, manifest

    def train(*_args, **_kwargs):
        return params(2), [{"epoch": e, "fit_loss": 1.0, "dev_loss": 1.0, "dev_f1": 0.5} for e in (1, 2, 3)]

    monkeypatch.setattr(cap, "_torch_setup", lambda *a, **k: None)
    monkeypatch.setattr(cap, "load_prithvi", lambda _ctx: object())
    monkeypatch.setattr(cap, "encode", encode)
    monkeypatch.setattr(cap, "head_probabilities", probabilities)
    monkeypatch.setattr(cap, "_load_head", lambda _torch, state: {k: np.asarray(v) for k, v in state.items()})
    monkeypatch.setattr(cap, "train_head", train)
    monkeypatch.setattr(cap, "load_head", load)
    return ctx, sources


def test_issue_22_m3_byod_adaptation_succeeds_and_records_head_serialization_parity(tmp_path, monkeypatch):
    ctx, sources = _byod_standins(tmp_path, monkeypatch)
    assert cap.main(["--root", str(ctx.root), "--stage", "byod", "--input", str(sources)]) == 0
    summary = json.loads((ctx.out / "byod" / "byod_summary.json").read_text(encoding="utf-8"))
    parity = summary["adaptation"]["head_serialization_parity"]
    assert parity["passed"] and parity["max_abs_difference"] == 0.0 and "same process" in parity["check"]
    assert "reload_max_abs_difference" not in summary["adaptation"]  # relabelled: never presented as the canonical reload
    assert (ctx.out / "byod_results.zip").is_file() and (ctx.work / "receipts" / "byod.json").is_file()
    assert (ctx.out / "byod" / "byod_head_adapter" / cap.ARTIFACT_WEIGHTS).is_file()


@pytest.mark.parametrize("corrupt", [
    lambda p: {**p, "conv2.bias": p["conv2.bias"] + np.array([0.0, 5.0], np.float32)},  # shifts the rice margin
    lambda p: {**p, "conv2.bias": np.full_like(p["conv2.bias"], np.nan)},
])
def test_issue_22_m3_a_parity_failure_stops_byod_without_a_receipt_bundle_or_refusal(tmp_path, monkeypatch, capsys,
                                                                                        corrupt):
    ctx, sources = _byod_standins(tmp_path, monkeypatch, corrupt_reload=corrupt)
    with pytest.raises(cap.ReloadParityError, match="head-serialization parity"):
        cap.main(["--root", str(ctx.root), "--stage", "byod", "--input", str(sources)])
    assert "Refused:" not in capsys.readouterr().out  # exit 1 with a traceback, not the learner refusal path
    assert not (ctx.out / "byod_results.zip").exists() and not (ctx.work / "receipts" / "byod.json").exists()
    assert not (ctx.out / "byod" / "byod_summary.json").exists()
    assert not (ctx.out / "byod" / "byod_head_adapter").exists()  # the unverified adapter is discarded
    failure = json.loads((ctx.out / "byod" / "byod_verification_failure.json").read_text(encoding="utf-8"))
    assert failure["passed"] is False


def test_issue_22_m3_learner_text_describes_the_check_as_head_serialization_parity():
    byod = _markdown_with("## 12. Optional: bring your own chips")
    assert "head-serialization parity" in byod and "no `byod_results.zip` is written" in byod
    assert "reloaded from files." not in byod and "does not rebuild the base model" in byod
    trouble = _markdown_with("## Troubleshooting")
    assert "head-serialization parity" in trouble and "not a `Refused:` message" in trouble


def test_issue_22_m2_label_lineage_is_explicit_and_the_recipe_is_checked(tmp_path, monkeypatch):
    lineage = gen.DATA_MANIFEST["label_lineage"]
    assert lineage["recipe_sha256"] == cap.LABEL_RECIPE_SHA256
    assert lineage["code_map"] == {"1": 1, "2": 1, "3": 1, "0": 0, "other": "build fails"}
    assert "same year" in lineage["layer"] and "composite" in lineage["layer"] and "nearest" in lineage["resampling"]
    assert "976530ffa324f0d701051d757214a5dca977621d5a3e752b4f11bac0dbf7f032" in lineage["rebuild_kit"]
    labels = _markdown_with("## 2. Inspect the chips")
    for phrase in ("annual paddy-rice occurrence according to a published map", "same year", "nearest neighbour",
                   "water and sea", "each** side of every class edge", "`label ≠ 255`", "7adf68fe"):
        assert phrase in labels, phrase
    markdown = "\n".join(text for kind, text in _cells() if kind == "markdown")
    assert "20 m band along class edges" not in markdown and "where rice is grown" not in markdown
    assert "not every exported pixel" in _markdown_with("### Problem statement")
    licences = _markdown_with("## Data, model, attribution and licences")
    assert "recipe-kit.zip" in licences and "not the four-year composite" in licences

    def index(rows):
        data = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
        (tmp_path / "INDEX.jsonl").write_bytes(data)
        monkeypatch.setattr(cap, "INDEX_SHA256", hashlib.sha256(data).hexdigest())

    row = {"group_id": "ph_1_2", "year": 2020, "recipe_sha256": cap.LABEL_RECIPE_SHA256}
    index([row])
    assert cap.read_index(tmp_path) == [row]
    index([row, {**row, "group_id": "ph_3_4", "recipe_sha256": "0" * 64}])
    with pytest.raises(ValueError, match="label recipes"):
        cap.read_index(tmp_path)


def test_issue_22_s1_s4_s5_whole_pipeline_claims_spatial_detail_and_responsive_answers():
    def row(system, f1):
        return {"system": system, "f1": f1, "iou": None, "precision": None, "recall": None}

    answer = "\n".join(cap.answer_lines([row("A all-non-rice", 0.0), row("B LightGBM", 0.758),
                                         row("C frozen Prithvi + head", 0.601)]))
    assert "two whole pipelines" in answer and "does not isolate the contribution of foundation-model pretraining" in answer
    assert "8 x 8 logit grid bilinearly upsampled" in answer and "not independently established field accuracy" in answer
    assert "operational requirements" in answer and "mapped rice better" not in answer  # S4: three separate claims
    opening = _markdown_with("**Driving question:**")
    assert "whole pipelines" in opening
    problem = _markdown_with("### Problem statement")
    assert "(1) **Signal:**" in problem and "(3) **Practical usefulness:**" in problem
    assert "addresses the first two questions, not the third" in problem
    sec4 = _markdown_with("## 4. Trace how Prithvi's input")
    assert "to hide the mismatch" not in sec4 and "domain shifts" in sec4
    sec5 = _markdown_with("## 5. Train head C")
    assert "**pixel spacing**" in sec5 and "**resolved detail**" in sec5 and "no high-resolution skip connection" in sec5
    assert "not of every Prithvi-based segmentation design" in sec5
    prepare = _markdown_with("**Checkpoint:** name one month range")
    assert "flatter non-rice pixel" not in prepare and "single examples" in prepare and "one peak, none" in prepare
    ids = {c["id"]: _source(c) for c in _notebook()["cells"]}
    assert "show_source('select_frames', 'prithvi_input')" in ids["rice-source-input"]
    assert "show_source('_head_module')" in ids["rice-source-head"]


def test_issue_22_s2_spatial_audit_measures_separation_instead_of_assuming_it():
    def chip(cid, site, role, x, y, year=2020):
        return {"id": cid, "site": site, "year": year, "role": role, "x_min": x, "y_max": y}

    audit = cap.spatial_audit([
        chip("a", "s1", "fit", 0, 0), chip("b", "s2", "development", 2560, 0),  # touching across roles
        chip("c", "s3", "validation", 1000, 0),  # overlaps a
        chip("d", "s1", "development", 0, 0, 2021),  # site s1 in two roles and two years
        chip("e", "s5", "fit", 100_000, 0),
    ])
    assert audit["unique_sites"] == 4 and audit["sites_with_repeated_years"] == 1
    assert audit["sites_in_more_than_one_role"] == ["s1"]
    assert ["a", "c"] in audit["cross_role_overlapping_footprints"]
    assert ["a", "b"] in audit["cross_role_touching_footprints"]
    assert audit["min_cross_role_distance_m"]["development-fit"] == 0.0
    assert cap.footprint_gap({"x_min": 0, "y_max": 0}, {"x_min": 5120, "y_max": 0}) == (2560.0, False)
    source = (TOOLS / "capstone.py").read_text(encoding="utf-8")
    prepare = source[source.index("def stage_prepare"):source.index("def _with_label_two")]
    assert 'ctx.out / "spatial_audit.json"' in prepare
    sec3 = _markdown_with("## 3. Predict, declare the protocol")
    assert "keep neighbouring chips out of both roles" not in sec3 and "does not by itself guarantee distance" in sec3
    baseline = next(text for kind, text in _cells() if kind == "code" and "run_stage('baseline')" in text)
    assert baseline.startswith("show_record('spatial_audit.json')")


def test_issue_22_s2_byod_grouping_flags_unverified_sites_duplicates_and_overlapping_footprints(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    ctx, sources = _byod_standins(tmp_path, monkeypatch)
    (sources / "site0_2020.h5").rename(sources / "copy_a_2020.h5")
    import shutil

    shutil.copy(sources / "copy_a_2020.h5", sources / "copy_b_2021.h5")  # a renamed duplicate
    with h5py.File(sources / "copy_b_2021.h5", "a") as handle:
        handle.attrs["group_id"] = "elsewhere"
    with h5py.File(sources / "site1_2020.h5", "a") as handle:
        del handle.attrs["group_id"]  # falls back to the file name
    assert cap.main(["--root", str(ctx.root), "--stage", "byod", "--input", str(sources)]) == 0
    summary = json.loads((ctx.out / "byod" / "byod_summary.json").read_text(encoding="utf-8"))
    assert summary["grouping"]["duplicate_content"] == [["copy_a_2020", "copy_b_2021"]]
    assert summary["grouping"]["site_ids_from_file_name_unverified"] == ["site1_2020"]
    assert summary["adaptation"].startswith("not run: chips with identical content")
    assert (ctx.out / "byod_results.zip").is_file()  # prediction still completes

    chips = {f"c{i}": {"S2": np.full((1,), i, np.float32), "S2_valid": np.ones(1, np.uint8), "label": None,
                       "attrs": {"x_min": 1000.0 * i, "y_max": 0.0, "group": f"g{i}"}} for i in range(2)}
    check = cap.byod_grouping_check(["c0", "c1"], chips, {"c0": "g0", "c1": "g1"})
    assert check["overlapping_footprints_with_different_sites"] == [["c0", "c1"]] and not check["duplicate_content"]


def test_issue_22_s2_successful_byod_adaptation_records_its_spatial_audit(tmp_path, monkeypatch):
    ctx, sources = _byod_standins(tmp_path, monkeypatch)
    assert cap.main(["--root", str(ctx.root), "--stage", "byod", "--input", str(sources)]) == 0
    summary = json.loads((ctx.out / "byod" / "byod_summary.json").read_text(encoding="utf-8"))
    audit = summary["adaptation"]["spatial_audit"]
    assert audit["chips"] == 6 and audit["unique_sites"] == 6 and audit["georeferenced_chips"] == 0
    assert summary["grouping"]["site_ids_from_file_name_unverified"] == []


def test_issue_22_s3_observation_support_partitions_the_primary_scored_pixels():
    valid = np.ones((24, 4, 4), np.uint8)
    bins = [0, 6, 12, 18]
    valid[0, 0, :] = 0
    valid[6, 0, :2] = 0
    counts = cap.observed_frame_counts(valid, bins)
    assert counts.dtype == np.uint8 and counts[0, 0] == 2 and counts[0, 3] == 3 and counts[1, 1] == 4
    rng = np.random.default_rng(3)
    labels = [rng.integers(0, 2, (4, 4)).astype(np.uint8) for _ in range(2)]
    labels[0][3, 3] = 255
    probs = [rng.uniform(size=(4, 4)) for _ in range(2)]
    rows = cap.support_rows("C", probs, labels, [counts, counts], 0.5)
    primary = cap.pooled(probs, labels, 0.5)
    assert [r["observed_frames"] for r in rows] == [2, 3, 4]  # empty groups are omitted, never reported as zero
    for key in ("tp", "fp", "fn", "tn"):
        assert sum(r[key] for r in rows) == primary[key]


def _evaluation_standins(tmp_path: Path, monkeypatch):
    """CPU stand-ins for evaluate -> export -> reload (no torch, LightGBM or matplotlib): two synthetic validation chips
    with cloud gaps, a fake booster, a tiny numpy head on random 8-channel 'features'. Real code paths: scoring, the
    observation-support diagnostic, the head artifact, the four GeoTIFFs per chip and the reload checks."""
    import types

    h5py = pytest.importorskip("h5py")
    pytest.importorskip("safetensors")
    pytest.importorskip("tifffile")
    ctx = cap.Context(tmp_path / "run")
    (ctx.work / "receipts").mkdir(parents=True, exist_ok=True)
    ids, chips = ["ph_1_1_2020", "ph_9_9_2021"], {}
    rng = np.random.default_rng(7)
    features = {}
    for i, cid in enumerate(ids):
        chip = _chip(60 + i)
        chip["S2_valid"][:, :40, :] = 0  # a cloudy band: fewer observed frames there
        chip["S2_valid"][3, 40:80, :] = 0
        chip["S2"][np.broadcast_to(chip["S2_valid"][:, None] == 0, chip["S2"].shape)] = np.nan
        path = tmp_path / f"{cid}.h5"
        with h5py.File(path, "w") as handle:
            handle["S2"], handle["S2_valid"], handle["label"] = chip["S2"], chip["S2_valid"], chip["label"]
        chips[cid] = {"path": str(path), "role": "validation", "region": "PH-07", "x_min": 614400.0 + 2560 * i,
                      "y_max": 1095680.0, "frames": cap.select_frames(chip["S2_valid"])}
        features[cid] = rng.normal(size=(8, 8, 8)).astype(np.float32)
    cap.write_json(ctx.work / "records.json", {"chips": chips, "roles": {"validation": ids}})
    head = {k: np.random.default_rng(5).normal(size=s).astype(np.float32) for k, s in cap.head_shapes(8, 4).items()}
    np.savez(ctx.work / "head_c.npz", **head)
    np.savez(ctx.work / "features_t4.npz", **features)
    np.save(ctx.work / "baseline_b_fill.npy", np.zeros(len(cap.M1_NAMES)))
    cap.write_json(ctx.out / "selection.json", {"B": {"threshold": 0.5}, "C": {"threshold": 0.45, "selected_epoch": 3,
                                                "config": cap.HEAD_CONFIG}, "activity": {}})
    for name in ("prepare", "baseline", "train-head", "activity"):
        (ctx.work / "receipts" / f"{name}.json").write_text("{}")

    class Booster:
        def __init__(self, **_kwargs):
            pass

        def predict(self, x, **_kwargs):
            return 1.0 / (1.0 + np.exp(-np.nan_to_num(x[:, 0]) * 3 + 0.5))

    def probabilities(_torch, state, x, device="cpu"):
        margin = cap.head_forward_numpy(state, np.asarray(x, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-np.kron(margin, np.ones((16, 16))).astype(np.float64)))

    by_input = {}

    def encode(_pipe, x):  # the reload recomputes C's input from the HDF5 chip; map it back to that chip's features
        key = round(float(np.abs(x).sum()), 3)
        return features[by_input[key]][None]

    for cid in ids:
        chip = cap.read_chip_h5(Path(chips[cid]["path"]))
        by_input[round(float(np.abs(cap.prithvi_input(chip, chips[cid]["frames"]["bins"])[0]).sum()), 3)] = cid
    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(Booster=Booster))
    torch_standin = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(cap, "_torch_setup", lambda *a, **k: torch_standin)
    monkeypatch.setattr(cap, "_load_head", lambda _torch, state: {k: np.asarray(v) for k, v in state.items()})
    monkeypatch.setattr(cap, "head_probabilities", probabilities)
    monkeypatch.setattr(cap, "_evaluation_figures", lambda *a, **k: None)  # matplotlib is not in CI
    monkeypatch.setattr(cap, "load_prithvi", lambda _ctx: object())
    monkeypatch.setattr(cap, "encode", encode)
    return ctx, ids


def test_issue_22_s3_evaluate_export_reload_write_and_verify_observation_support(tmp_path, monkeypatch):
    ctx, ids = _evaluation_standins(tmp_path, monkeypatch)
    for stage in ("evaluate", "export", "reload"):
        assert cap.main(["--root", str(ctx.root), "--stage", stage]) == 0, stage
    metrics = {r["system"]: r for r in cap.read_csv(ctx.out / "metrics.csv")}
    support = cap.read_csv(ctx.out / "observation_support.csv")
    assert {r["system"] for r in support} == {"B LightGBM", "C frozen Prithvi + head"}
    assert {r["observed_frames"] for r in support} == {0, 4}  # never-observed pixels stay scored (count 0)
    for system in ("B LightGBM", "C frozen Prithvi + head"):
        for key in ("tp", "fp", "fn", "tn"):
            assert sum(r[key] for r in support if r["system"] == system) == metrics[system][key], (system, key)
    verification = json.loads((ctx.out / "verification.json").read_text(encoding="utf-8"))
    assert verification["observation_support_geotiffs_match"] is True
    array, geo = cap.read_geotiff(ctx.out / "predictions" / f"{ids[0]}_observed_frames.tif")
    assert array.dtype == np.uint8 and array.max() <= 4 and geo["epsg"] == cap.EPSG
    # a tampered support map fails the reload
    cap.write_geotiff(ctx.out / "predictions" / f"{ids[0]}_observed_frames.tif", np.zeros((128, 128)), nodata=None,
                      x_min=geo["x_min"], y_max=geo["y_max"])
    with pytest.raises(RuntimeError, match="reload parity failed"):
        cap.main(["--root", str(ctx.root), "--stage", "reload"])


def test_issue_22_s3_learner_text_explains_the_asymmetry_and_shows_the_diagnostic():
    sec4 = _markdown_with("## 4. Trace how Prithvi's input")
    assert "no validity channel" in sec4 and "observed fraction" in sec4
    sec9 = _markdown_with("## 9. Interpret the result")
    assert "secondary diagnostic" in sec9 and "primary comparison is unchanged" in sec9
    figures = next(text for kind, text in _cells() if kind == "code" and "show_figures('validation_*')" in text)
    assert "observation_support.csv" in figures


def _geotiff_tags_by_struct(path: Path) -> dict[int, tuple]:
    """An independent little reader of a classic TIFF's first IFD (struct only, no tifffile), returning the values of
    the GeoTIFF tags 33550 (ModelPixelScale), 33922 (ModelTiepoint) and 34735 (GeoKeyDirectory) plus width/length."""
    import struct

    data = path.read_bytes()
    order = {b"II": "<", b"MM": ">"}[data[:2]]
    assert struct.unpack(order + "H", data[2:4])[0] == 42, "not a classic TIFF"
    ifd = struct.unpack(order + "I", data[4:8])[0]
    count = struct.unpack(order + "H", data[ifd:ifd + 2])[0]
    sizes = {3: ("H", 2), 4: ("I", 4), 12: ("d", 8)}
    out = {}
    for n in range(count):
        tag, kind, items, raw = struct.unpack(order + "HHI4s", data[ifd + 2 + 12 * n: ifd + 14 + 12 * n])
        if tag not in (256, 257, 33550, 33922, 34735) or kind not in sizes:
            continue
        code, size = sizes[kind]
        blob = raw if items * size <= 4 else data[struct.unpack(order + "I", raw)[0]:][:items * size]
        out[tag] = struct.unpack(order + code * items, blob[:items * size])
    return out


def test_issue_22_m1_geotiff_placement_is_checked_and_read_independently(tmp_path):
    pytest.importorskip("tifffile")
    x_min, y_max = 614400.0, 1095680.0
    cap.write_geotiff(tmp_path / "p.tif", np.ones((128, 128), np.uint8), x_min=x_min, y_max=y_max)
    placement = cap.geotiff_placement(tmp_path / "p.tif")
    assert placement["transform"] == (20.0, 0.0, x_min, 0.0, -20.0, y_max)
    assert placement["bounds"] == (x_min, y_max - 2560.0, x_min + 2560.0, y_max)
    assert placement["geokeys"] == {1024: 1, 1025: 1, 3072: 32651}
    assert cap.placement_matches(placement, x_min, y_max)
    assert not cap.placement_matches(placement, x_min + 20.0, y_max)  # one pixel off is caught
    cap.write_geotiff(tmp_path / "small.tif", np.ones((64, 64), np.uint8), x_min=x_min, y_max=y_max)
    assert not cap.placement_matches(cap.geotiff_placement(tmp_path / "small.tif"), x_min, y_max)
    cap.write_geotiff(tmp_path / "plain.tif", np.ones((128, 128), np.uint8), x_min=None, y_max=None)
    assert not cap.placement_matches(cap.geotiff_placement(tmp_path / "plain.tif"), x_min, y_max)
    # Independent of tifffile's reader: the raw tags say the same thing a GIS would read.
    tags = _geotiff_tags_by_struct(tmp_path / "p.tif")
    assert (tags[256][0], tags[257][0]) == (128, 128)
    assert tags[33550] == (20.0, 20.0, 0.0) and tags[33922] == (0.0, 0.0, 0.0, x_min, y_max, 0.0)
    keys = tags[34735]
    assert keys[:4] == (1, 1, 0, 3)
    assert {keys[i]: keys[i + 3] for i in range(4, len(keys), 4)} == {1024: 1, 1025: 1, 3072: 32651}


def test_issue_22_m1_reload_fails_when_a_prediction_geotiff_is_misplaced(tmp_path, monkeypatch):
    ctx, ids = _evaluation_standins(tmp_path, monkeypatch)
    for stage in ("evaluate", "export", "reload"):
        assert cap.main(["--root", str(ctx.root), "--stage", stage]) == 0, stage
    assert json.loads((ctx.out / "verification.json").read_text(encoding="utf-8"))["geotiff_placement_verified"] is True
    path = ctx.out / "predictions" / f"{ids[1]}_C_score.tif"
    array, geo = cap.read_geotiff(path)
    cap.write_geotiff(path, array, x_min=geo["x_min"] + 2560.0, y_max=geo["y_max"], nodata=None, dtype=np.float32)
    with pytest.raises(RuntimeError, match="reload parity failed"):
        cap.main(["--root", str(ctx.root), "--stage", "reload"])
    assert json.loads((ctx.out / "verification.json").read_text(encoding="utf-8"))["geotiff_placement_verified"] is False


def test_issue_22_m2_fine_tune_text_names_what_the_curve_records():
    sec7 = _markdown_with("## 7. Optional extension")
    assert "development loss and F1 per epoch" not in sec7 and "fit loss and the development rice F1" in sec7
    source = (TOOLS / "capstone.py").read_text(encoding="utf-8")
    finetune = source[source.index("def stage_finetune"):source.index("def drop_stale_finetune")]
    assert 'curve.append({"epoch": epoch, "fit_loss": float(np.mean(losses)), "dev_f1": f1})' in finetune
    assert "dev_loss" not in finetune  # the text and the record agree: no development loss for C-FT


def test_issue_22_learner_reread_the_byod_cell_shows_the_test_comparison_before_the_long_record(tmp_path, monkeypatch):
    ctx, sources = _byod_standins(tmp_path, monkeypatch)
    assert cap.main(["--root", str(ctx.root), "--stage", "byod", "--input", str(sources)]) == 0
    cell = next(text for kind, text in _cells() if kind == "code" and "USE_BYOD = " in text)
    shown = []
    namespace = {"ROOT": ctx.root, "Path": Path, "json": json, "run_stage": lambda *a, **k: True,
                 "show_table": lambda *a, **k: shown.append(("table", a)), "FileLink": lambda p: p,
                 "display": lambda *a, **k: shown.append(("display", a)),
                 "display_rows": lambda rows, columns: shown.append(("rows", [r["system"] for r in rows]))}
    exec(compile(cell.replace("USE_BYOD = False", "USE_BYOD = True", 1)
                 .replace("BYOD_PATH = ''", f"BYOD_PATH = {str(sources)!r}", 1), "byod-cell", "exec"), namespace)
    assert ("rows", ["all non-rice", "exported canonical head", "adapted head"]) in shown


def test_issue_22_m3_release_wording_matches_the_recorded_evidence():
    title = _markdown_with("**Driving question:**")
    assert "stays Candidate until a browser Run all" not in title
    assert "**Release status and evidence.**" in title and "separate from how much execution evidence exists" in title
    assert "fine-tune checkbox ticked" in title and "`1ed294f0`" in title and "upload dialog" in title
    assert "not evidence for it until it is run again" in title
    assert _notebook()["metadata"]["dimer"]["release_status"] == "Candidate"  # never promoted by this issue
