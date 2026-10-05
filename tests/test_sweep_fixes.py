"""Regression tests for the 2026-10-05 fleet-sweep fixes of prithvi_eo_feature_extraction_colab.ipynb (SWP-R, SWP-G, SWP-A, SWP-F, SWP-B).

Every test needs only CI's dependencies (NumPy-level, no model, no torch): the notebook's own cell sources are executed
with stand-ins where a model would be needed. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_eo_feature_extraction_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in _code_cells(notebook) if marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart guard, environment reuse, idempotent Section 1 ----------------------


def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(_source(c) for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = _source(kernel[0])
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", "_isolated_environment_ready()", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source, needed


def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock([p for p in build._pins(ROOT) if "==" in p], lock_text)
    # the environment folder is keyed on the lock digest, so a second Run all reuses it
    assert "'dimer_isolated_env_' + LOCK_SHA256[:12]" in source


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker, with every variable later cells created, is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("import os; print('value', learner_value, os.environ.get('MPLBACKEND'), os.environ.get('PYTHONSTARTUP'))\n")
        assert "value 42 Agg None" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- SWP-G: the guided layer ----------------------------------------------------------------------------------------


def test_swp_g_guided_layer_is_present_and_infrastructure_is_collapsed(notebook):
    md = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**How to use this notebook.**", "**Roadmap:**", "**Input → Model → Output.**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert heading in md, heading
    assert md.count("**Predict") >= 5
    assert md.count("<details><summary>Check your reasoning</summary>") >= 5
    assert md.index("**How to use this notebook.**") < md.index("## 1. Install the pinned runtime")
    for leftover in ("{{", "}}", "{MODEL_ID}", "@P:", "TODO", "TBD"):
        assert leftover not in md, leftover
    infra = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in _source(c) or c["metadata"].get("dimer", {}).get("embedded_module") or "stage_missing_files(WEIGHTS_DIR, allow_download=True)" in _source(c)]
    assert infra and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert "> **Infrastructure.**" in md


def test_swp_g_checkpoint_answers_quote_the_recorded_run(notebook):
    md = _markdown(notebook)
    for value in ("0.8149", "0.7865", "0.0773", "0.1294", "0.6931", "0.1996", "0.7577"):
        assert value in md, value


# --- SWP-A: the probe-vs-baseline comparison is a recorded verdict, not an assert ------------------------------------


def test_swp_a_no_model_quality_assert_remains(notebook):
    code = "\n".join(_source(c) for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = [line.strip() for line in code.splitlines() if line.strip().startswith("assert ")]
    # only the reload-parity contract check is left
    assert asserts == ["assert parity['positive_iou_diff'] < 1e-3 and parity['max_abs_score_diff'] < 1e-3"]
    assert "> frozen_test['baseline_not_burned']" not in code


def _metrics(burn_iou: float) -> dict:
    return {"iou": {"not burned": 0.9, "burn scar": burn_iou}, "mean_iou": 0.5, "accuracy": 0.8, "precision": 0.0, "recall": 0.0, "f1": 0.0}


@pytest.mark.parametrize(("probe_iou", "verdict"), [(0.0, "no improvement"), (0.42, "improved")])
def test_swp_a_section_7_records_the_verdict_and_writes_the_report(notebook, tmp_path, monkeypatch, probe_iou, verdict):
    """The real Section 7 cell with a stand-in pipeline: a probe that does not beat the baseline is recorded as
    `no improvement` and the report is still written (the old cell raised AssertionError here)."""
    source = _cell(notebook, "probe_verdict = ")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    adapted = {"model": _metrics(probe_iou)}
    pipe = types.SimpleNamespace(evaluate=lambda records: adapted)
    history = [{"epoch": 0, "val_loss": 0.69, "val": {"iou": {"burn scar": 0.0}}}, {"epoch": 1, "val_loss": 0.69, "val": {"iou": {"burn scar": probe_iou}}}]
    namespace = {
        "json": json, "pipe": pipe, "test_records": [], "val_records": [], "CLASS_NAMES": ("not burned", "burn scar"),
        "frozen_test": {"model": _metrics(0.0), "baseline_not_burned": _metrics(0.0)}, "frozen_val": {"model": _metrics(0.0)},
        "adapt_result": {"history": history, "best_epoch": 1, "n_trainable": 2050}, "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k",
        "data_source": "stand-in", "dataset_report": {}, "embeddings": {"embeddings": []}, "cosine": lambda v: types.SimpleNamespace(tolist=list), "frame_mean": [],
        "reconstruction": {"results": []}, "adapt_seconds": 0.0,
    }
    exec(compile(source, "<section 7>", "exec"), namespace)
    assert namespace["comparison"]["verdict"] == {"probe_vs_majority_baseline_burn_iou": verdict}
    report = json.loads((tmp_path / "outputs" / "prithvi_eo_feature_extraction_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"]["probe_vs_majority_baseline_burn_iou"] == verdict


def test_swp_a_contract_checks_still_stop_the_run(notebook, tmp_path, monkeypatch):
    source = _cell(notebook, "probe_verdict = ")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = types.SimpleNamespace(evaluate=lambda records: {"model": _metrics(0.5)})
    history = [{"epoch": 0, "val_loss": 0.5, "val": {"iou": {"burn scar": 0.0}}}, {"epoch": 1, "val_loss": 0.9, "val": {"iou": {"burn scar": 0.5}}}]
    namespace = {
        "json": json, "pipe": pipe, "test_records": [], "val_records": [], "CLASS_NAMES": ("not burned", "burn scar"),
        "frozen_test": {"model": _metrics(0.0), "baseline_not_burned": _metrics(0.0)}, "frozen_val": {"model": _metrics(0.0)},
        "adapt_result": {"history": history, "best_epoch": 1}, "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k",
        "data_source": "stand-in", "dataset_report": {}, "embeddings": {"embeddings": []}, "cosine": lambda v: types.SimpleNamespace(tolist=list), "frame_mean": [],
        "reconstruction": {"results": []}, "adapt_seconds": 0.0,
    }
    with pytest.raises(RuntimeError, match="contract: the kept epoch"):
        exec(compile(source, "<section 7>", "exec"), namespace)


def test_swp_a_result_json_carries_the_verdict(notebook):
    assert "'verdict': comparison['verdict']," in _cell(notebook, "result_payload = {")


# --- SWP-F: a re-run of Section 5 after Section 6 still scores the zero head as "frozen" -----------------------------


def test_swp_f_frozen_evaluation_sets_the_trained_probe_aside_and_restores_it(notebook):
    source = _cell(notebook, "frozen_test = pipe.evaluate(test_records)")
    snippet = source[source.index("_attached_probe = ") : source.index("pipe.probe, pipe.feature_stats, pipe.feature_layer, pipe.adapter = _attached_probe") + len("pipe.probe, pipe.feature_stats, pipe.feature_layer, pipe.adapter = _attached_probe")]
    seen = []
    pipe = types.SimpleNamespace(probe="trained", feature_stats="stats", feature_layer=11, adapter={"best_epoch": 30})
    pipe.evaluate = lambda records: seen.append((pipe.probe, pipe.feature_stats, pipe.adapter)) or {"model": {}}
    namespace = {"pipe": pipe, "test_records": [], "val_records": []}
    exec(compile(snippet, "<section 5 frozen>", "exec"), namespace)
    assert seen == [(None, None, None), (None, None, None)]
    assert (pipe.probe, pipe.feature_stats, pipe.feature_layer, pipe.adapter) == ("trained", "stats", 11, {"best_epoch": 30})
    pipe.evaluate = lambda records: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        exec(compile(snippet, "<section 5 frozen>", "exec"), namespace)
    assert pipe.probe == "trained" and pipe.adapter == {"best_epoch": 30}


# --- SWP-B: BYOD_PATH works off Colab; the upload fallback is guarded ------------------------------------------------


def _byod_block(notebook: dict) -> str:
    source = _cell(notebook, "if USE_BYOD:")
    return source[source.index("if USE_BYOD:") : source.index("else:\n    splits = fetch_sample_dataset")]


def _byod_namespace(path: str) -> dict:
    loaded = []
    return {
        "USE_BYOD": True, "BYOD_PATH": path, "Path": Path, "loaded": loaded,
        "load_byod_dataset": lambda p: loaded.append(Path(p)) or ["r"],
        "split_dataset": lambda records, seed: {"train": records, "validation": records, "test": records},
    }


def test_swp_b_byod_path_reads_a_folder_without_colab(notebook, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    folder = tmp_path / "my_scenes"
    folder.mkdir()
    namespace = _byod_namespace(str(folder))
    exec(compile(_byod_block(notebook), "<byod>", "exec"), namespace)
    assert namespace["loaded"] == [folder] and namespace["data_source"] == "BYOD (my_scenes)"


def test_swp_b_missing_path_and_off_colab_upload_give_clear_messages(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nope.zip' does not exist"):
        exec(compile(_byod_block(notebook), "<byod>", "exec"), _byod_namespace("nope.zip"))
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(compile(_byod_block(notebook), "<byod>", "exec"), _byod_namespace(""))


@pytest.mark.parametrize(("uploaded", "message"), [(None, "Upload exactly one .zip file \\(received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2"), ({"scenes.tar": b""}, "scenes.tar: upload one .zip")])
def test_swp_b_cancelled_or_wrong_upload_is_refused(notebook, tmp_path, monkeypatch, uploaded, message):
    monkeypatch.chdir(tmp_path)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match=message):
        exec(compile(_byod_block(notebook), "<byod>", "exec"), _byod_namespace(""))
