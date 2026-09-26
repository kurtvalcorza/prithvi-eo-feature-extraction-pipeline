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
