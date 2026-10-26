"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from prithvi_eo_feature_extraction_pipeline import samples as samples_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/prithvi_eo_feature_extraction_colab.ipynb"
BYOD_CELL = "prithvi_eo_feature_extraction-16"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "prithvi_eo_feature_extraction"
EARLIER = {"splits": "old", "test_records": "old", "frozen_test": "old", "adapt_result": "old", "adapted_test": "old", "result_payload": "old"}
WRAP_END = "    refuse_byod(file_name, f'{type(exc).__name__}: {exc}')\n"


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_adapter").mkdir(parents=True)
    for name in ("sample_chip.tif", "evaluation_report.json", "result.json", "predictions.json"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _namespace(**extra) -> dict:
    ns = {k: v for k, v in vars(samples_module).items() if not k.startswith("__")}
    ns.update(EARLIER, os=os, Path=Path, np=__import__("numpy"), pipe="kept", **extra)
    return ns


def _section4(monkeypatch, tmp_path, path: str, *, use_byod: bool = True, source: str | None = None, extra: dict | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    source = source[: source.index(WRAP_END) + len(WRAP_END)]
    if use_byod:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns = _namespace(**(extra or {}))
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "nopairs.zip").write_bytes(_zip({"a.tif": b"x"}))
    (tmp_path / "badcols.zip").write_bytes(_zip({"pairs.csv": "a,b\n1,2\n"}))
    (tmp_path / "unreadable.zip").write_bytes(_zip({"pairs.csv": "id,image,label\nc1,c1.tif,c1_label.tif\n", "c1.tif": b"not a tiff", "c1_label.tif": b"not a tiff"}))
    return {
        "folder without pairs.csv (the scripted /etc run)": ("etc", "pairs.csv"),
        "plain file": ("etc/hostname", "must be a directory or a .zip"),
        "corrupt zip": ("fake.zip", "BadZipFile"),
        "zip without pairs.csv": ("nopairs.zip", "must contain pairs.csv"),
        "wrong columns": ("badcols.zip", "pairs.csv is missing columns"),
        "unreadable chip": ("unreadable.zip", ""),
        "missing": ("nope.zip", "does not exist"),
    }


CASES = ["folder without pairs.csv (the scripted /etc run)", "plain file", "corrupt zip", "zip without pairs.csv", "wrong columns", "unreadable chip", "missing"]


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    relative, rule = _cases(tmp_path)[case]
    file_name = Path(relative).name
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / relative))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line
    assert "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    source = _cell(BYOD_CELL)
    source = source[: source.index(WRAP_END) + len(WRAP_END)].replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'nope.zip'", 1)
    ns = _namespace()
    with pytest.raises(ValueError, match="^nope.zip: BYOD_PATH"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [(None, "(no file)", "received 0"), ({"a.zip": b"1", "b.zip": b"2"}, "a.zip, b.zip", "received 2"), ({"scenes.tar": b""}, "scenes.tar", "upload one .zip")])
def test_cancelled_or_wrong_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "chips").mkdir()

    def broken(_path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "chips"), extra={"load_byod_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)
    # This notebook clears its own outputs/ exports at the top of the cell on every run (FX-M2), so no file check here.


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise ValueError("pinned archive digest mismatch")

    with pytest.raises(ValueError, match="digest mismatch") as caught:
        _section4(monkeypatch, tmp_path, "", use_byod=False, extra={"fetch_sample_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)


@pytest.mark.skipif(not Path("/etc").is_dir() or Path("/etc/pairs.csv").exists(), reason="needs a POSIX /etc without pairs.csv, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_directory(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc': a folder present in every fresh runtime that has no pairs.csv."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc")
    line = _refusal_line(caught.value)
    assert line.startswith("Refused: etc: FileNotFoundError: ") and "pairs.csv" in line
