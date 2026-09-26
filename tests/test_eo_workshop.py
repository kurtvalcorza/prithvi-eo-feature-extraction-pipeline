# ruff: noqa: E501
"""Static contract tests for the supplemental EO and climate workshop notebook (not execution evidence)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_AI_for_Earth_Observation_and_Climate_Applications_Workshop.ipynb"


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _load()["cells"])


def test_generator_parity() -> None:
    subprocess.run([sys.executable, str(REPO / "tools" / "eo_workshop" / "build_workshop.py"), "--check"], cwd=REPO, check=True)


def test_metadata_declares_a_candidate_multi_capability_workshop() -> None:
    meta = _load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "MULTI-CAPABILITY"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["clean_runtime_evidence"] == "pending"


def test_committed_notebook_is_clean() -> None:
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_no_runtime_repo_dependency() -> None:
    body = _source()
    for literal in ["git clone ", "pip install -e", "raw.githubusercontent.com/kurtvalcorza"]:
        assert literal not in body


def test_reflectance_scaling_shares_the_pipelines_ceiling() -> None:
    # The flood / burn-scar pipelines read only values above REFLECTANCE_MAX = 2.0 as reflectance × 10 000; a lower
    # trigger rescales a bright chip that is already reflectance.
    body = _source()
    assert "if float(x.max()) > 2.0:" in body
    assert "if float(x.max()) > 1.0:" not in body


def _kernel_cells() -> list[str]:
    code = [c for c in _load()["cells"] if c["cell_type"] == "code"]
    return ["".join(c["source"]) for c in code]


def _bridge_namespace() -> dict:
    """Execute the routing cell with routing disabled, which defines the bridge without touching IPython."""
    import os

    namespace = {"SKIP_INSTALL": True, "os": os, "sys": sys, "subprocess": subprocess}
    exec(_kernel_cells()[1], namespace)
    return namespace


def test_no_runtime_restart_path() -> None:
    body = _source()
    assert "os.kill(" not in body
    assert "choose Run all again; " not in body
    install, bridge, verify = _kernel_cells()[:3]
    assert "# dimer: kernel cell" in install and "# dimer: kernel cell" in bridge
    assert "# dimer: kernel cell" not in verify
    assert '"--python", str(EO_PYTHON), *PINS' in install
    assert sum("# dimer: kernel cell" in cell for cell in _kernel_cells()) == 2


def test_routing_skips_kernel_cells_and_blank_cells() -> None:
    route = _bridge_namespace()["_route_to_isolated_runtime"]
    assert route(["x = 1\n"]) == ["_DIMER_EO_RUNTIME.run('x = 1\\n')\n"]
    assert route(["# dimer: kernel cell\n", "x = 1\n"]) == ["# dimer: kernel cell\n", "x = 1\n"]
    assert route(["\n"]) == ["\n"]


def test_isolated_runtime_round_trip() -> None:
    namespace = _bridge_namespace()
    shown = []
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda data, raw: shown.append(data))
    try:
        runtime.run("import numpy as np\nX = 41\nprint('hello')")
        runtime.run("X += 1\ndisplay(X)\nnp.arange(3).sum()")
        assert [d["text/plain"] for d in shown] == ["42", "np.int64(3)"]
        try:
            runtime.run("raise ValueError('boom')")
        except namespace["IsolatedCellError"] as exc:
            assert str(exc) == "ValueError: boom"
        else:
            raise AssertionError("a failing cell must raise in the kernel")
        shown.clear()
        runtime.run("display(X)")
        assert [d["text/plain"] for d in shown] == ["42"]
    finally:
        runtime.close()


def test_isolated_runtime_forwards_the_colab_upload(monkeypatch) -> None:
    namespace = _bridge_namespace()
    monkeypatch.setitem(sys.modules, "google.colab", object())  # the bridge only checks that the kernel is Colab
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda data, raw: None)
    runtime._colab_upload = lambda: {"scene.tif": b"bytes"}
    try:
        runtime.run("from google.colab import files\nUPLOADED = files.upload()")
        shown = []
        runtime._display = lambda data, raw: shown.append(data)
        runtime.run("display(UPLOADED)")
        assert shown[0]["text/plain"] == "{'scene.tif': b'bytes'}"
    finally:
        runtime.close()
