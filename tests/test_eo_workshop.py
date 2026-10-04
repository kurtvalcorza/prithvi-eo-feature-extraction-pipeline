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
    assert '"--require-hashes", "--only-binary", ":all:"' in install and "*PINS" not in install
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


# ---- review regressions (EO-M1..M4, EO-m1, EO-m2, EO-m4): the notebook's own helper source, executed with NumPy only ----
def _cell(cell_id: str) -> str:
    for cell in _load()["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise AssertionError(f"cell {cell_id} not found")


def _helpers() -> dict:
    """Execute the §4 helper cell (validation, integrity, metrics, output checks) with NumPy and inert stand-ins."""
    import types

    import numpy as np

    colors = types.SimpleNamespace(ListedColormap=lambda values: values)
    namespace = {
        "np": np, "pd": None, "torch": None, "tifffile": None, "hf_hub_download": None, "Path": Path,
        "matplotlib": types.SimpleNamespace(colors=colors), "plt": None, "IGNORE_INDEX": -1,
        "S2_L1C_BAND_INDICES": (1, 2, 3, 8, 11, 12),
    }
    exec(_cell("7fffe217"), namespace)
    return namespace


def test_foundation_units_are_decided_on_valid_pixels_only() -> None:
    import numpy as np

    h = _helpers()
    reflectance = np.full((6, 16, 16), 0.2, np.float32)
    dn = reflectance * 10_000
    for a, b in ((reflectance, dn), (reflectance.copy(), dn.copy())):
        np.testing.assert_allclose(h["validate_foundation_scene"](a, name="r"), h["validate_foundation_scene"](b, name="d"), rtol=1e-6)
    reflectance[:, 0, 0] = -9999
    dn[:, 0, 0] = -9999
    r, d = h["validate_foundation_scene"](reflectance, name="r"), h["validate_foundation_scene"](dn, name="d")
    np.testing.assert_allclose(r, d, rtol=1e-6)
    assert float(r[0, 1, 1]) == 2000.0 and float(r[0, 0, 0]) == -9999.0  # valid pixel converted, no-data kept
    assert h["UNIT_DECISIONS"]["r"]["source_units"] == "reflectance" and h["UNIT_DECISIONS"]["d"]["source_units"] == "dn"


def test_revalidation_never_rescales_converted_data() -> None:
    import numpy as np

    h = _helpers()
    dark = np.full((18, 224, 224), 0.0001, np.float32)
    once = h["validate_crop_scene"](dark, name="dark")
    assert float(once.max()) == 1.0
    np.testing.assert_array_equal(h["validate_crop_scene"](once, name="again", units="dn"), once)
    foundation = h["validate_foundation_scene"](np.full((6, 16, 16), 0.0001, np.float32), name="dark foundation")
    np.testing.assert_array_equal(h["validate_foundation_scene"](foundation, name="again", units="dn"), foundation)
    chip = h["validate_segmentation_chip"](np.full((6, 512, 512), 2000.0, np.float32), name="chip")
    np.testing.assert_array_equal(h["validate_segmentation_chip"](chip, name="chip again"), chip)


def test_unit_edge_cases_are_explicit() -> None:
    import numpy as np
    import pytest

    h = _helpers()
    negative = np.full((18, 224, 224), 0.1, np.float32)
    negative[0, 0, 0] = -0.01  # plausible slightly negative surface reflectance
    assert float(h["validate_crop_scene"](negative, name="negative").max()) == pytest.approx(1000.0)
    with pytest.raises(ValueError, match="ambiguous"):
        h["validate_crop_scene"](np.full((18, 224, 224), 40.0, np.float32), name="ambiguous")
    with pytest.raises(ValueError, match="every pixel is no-data"):
        h["validate_foundation_scene"](np.full((6, 16, 16), -9999, np.float32), name="empty")
    with pytest.raises(ValueError, match="no-data"):
        h["validate_crop_scene"](np.where(negative > 0, 1000.0, -9999).astype(np.float32), name="crop no-data")
    with pytest.raises(ValueError, match="every pixel is no-data"):
        h["validate_segmentation_chip"](np.zeros((6, 512, 512), np.float32), name="empty chip")
    with pytest.raises(ValueError, match="outside the plausible HLS range"):
        h["validate_foundation_scene"](np.full((6, 16, 16), 50_000, np.float32), name="too bright")
    declared = h["validate_crop_scene"](np.full((18, 224, 224), 40.0, np.float32), name="declared", units="dn")
    assert float(declared.max()) == 40.0 and h["UNIT_DECISIONS"]["declared"]["decision"] == "declared"


def test_invalid_model_outputs_are_rejected_before_maps_or_metrics() -> None:
    import numpy as np
    import pytest

    h = _helpers()
    invalid, check = h["InvalidModelOutput"], h["check_class_scores"]
    good = np.zeros((2, 2, 8, 8), np.float32)
    check(good, name="all background", batch=2, num_classes=2, spatial=(8, 8))  # legitimate finite output passes
    for bad in (np.full_like(good, np.nan), np.where(good == 0, np.inf, good)):
        with pytest.raises(invalid, match="NaN or infinite"):
            check(bad, name="bad", batch=2, num_classes=2, spatial=(8, 8))
    for shape in ((2, 3, 8, 8), (3, 2, 8, 8), (2, 2, 4, 8), (2, 8, 8)):
        with pytest.raises(invalid, match="expected class scores"):
            check(np.zeros(shape, np.float32), name="shape", batch=2, num_classes=2, spatial=(8, 8))
    with pytest.raises(invalid):
        h["check_probabilities"](np.full((1, 2, 4, 4), 0.7, np.float32), name="sums")
    with pytest.raises(invalid):
        h["check_class_map"](np.array([[0, 13]]), name="domain", num_classes=13)
    with pytest.raises(invalid):
        h["check_embedding"](np.full(1024, np.nan, np.float32), name="embedding", dim=1024)


def test_fractional_labels_are_rejected() -> None:
    import numpy as np
    import pytest

    h = _helpers()
    with pytest.raises(ValueError, match="whole numbers"):
        h["validate_label"](np.array([[0.9, 1.9, -1.2]], np.float32), name="fractional", shape=(1, 3), num_classes=2)
    ok = h["validate_label"](np.array([[0.0, 1.0, -1.0]], np.float32), name="integral", shape=(1, 3), num_classes=2)
    assert ok.tolist() == [[0, 1, -1]]


def test_failed_archive_verification_is_remembered(tmp_path) -> None:
    import hashlib
    import io
    import tarfile

    import pytest

    h = _helpers()
    data = b"pinned member bytes"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        info = tarfile.TarInfo("nested/chip.tif")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    archive, opens = buffer.getvalue(), []

    class Reader:
        def __init__(self, url, total):
            self.io, self.count, self.sha, self.resumes = io.BytesIO(archive), 0, hashlib.sha256(), 0
            opens.append(url)

        def read(self, n=-1):
            chunk = self.io.read(n)
            self.count += len(chunk)
            self.sha.update(chunk)
            return chunk

        def close(self):
            self.io.close()

    h["_ResumableReader"] = Reader
    wanted = {"nested/chip.tif": (len(data), hashlib.sha256(data).hexdigest())}
    stream = h["stream_pinned_members"]
    for _ in range(2):  # a wrong whole-archive digest fails every time; it is never forgotten on retry
        with pytest.raises(ValueError):
            stream("synthetic://bad", len(archive), "0" * 64, wanted, tmp_path / "bad")
    assert len(opens) == 2 and not (tmp_path / "bad" / "chip.tif").exists()
    good = hashlib.sha256(archive).hexdigest()
    out = stream("synthetic://good", len(archive), good, wanted, tmp_path / "good", label="good")
    assert out["nested/chip.tif"].read_bytes() == data and h["ARCHIVE_STATUS"]["good"]["whole_archive_verified"] is True
    stream("synthetic://good", len(archive), good, wanted, tmp_path / "good", label="good")
    assert len(opens) == 3  # the verified cache is reused
    out["nested/chip.tif"].write_bytes(b"corrupted")
    assert stream("synthetic://good", len(archive), good, wanted, tmp_path / "good")["nested/chip.tif"].read_bytes() == data
    assert len(opens) == 4


def test_byod_destination_is_created_before_any_model_loads() -> None:
    body = _cell("f23033f9")
    first_load = min(body.index(f) for f in ("load_foundation_model()", "load_segmentation_model(", "load_crop_model()"))
    assert body.index("byod_dir.mkdir(") < first_load
    assert 'OUTPUT_ROOT / "predictions" / BYOD_CAPABILITY' not in body and "receipt.json" in body


def test_reconstruction_activity_is_separate_from_the_canonical_result() -> None:
    canonical, activity, export = _cell("f369c51e"), _cell("guided-controlled-change-run"), _cell("33f19388")
    assert "75% hidden" not in canonical and 'mask_ratio=CANONICAL_MASK_RATIO' in canonical
    assert "reconstruction_result =" not in activity and "activities" in activity and "load_foundation_model()" in activity
    assert "RUN_MASK_RATIO_ACTIVITY = False" in activity  # Run all never executes the activity
    assert '"mask_ratio": 0.75' not in export and 'reconstruction_result["mask_ratio"]' in export


def test_report_packages_only_this_runs_files() -> None:
    export = _cell("33f19388")
    assert "make_archive" not in export and "PRODUCED_FILES" in export and "RUN_ID" in export


def test_run_id_survives_rerunning_the_controls_cell() -> None:
    controls = _cell("ba798aab")
    assert 'RUN_ID = globals().get("RUN_ID") or (' in controls


# ----- 2026-10-03 uv isolated environment ---------------------------------------------------------------------------
LOCK = REPO / "tutorials" / "requirements-eo-workshop.lock.txt"
PINS_IN = REPO / "tools" / "eo_workshop" / "requirements-eo-workshop.in"


def _install_namespace() -> dict:
    """Run only the constant definitions of the install cell (everything before SKIP_INSTALL); nothing is installed."""
    import ast

    install = _kernel_cells()[0]
    module = ast.parse(install)
    keep = []
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "SKIP_INSTALL":
                break
            keep.append(node)
    namespace: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "install", "exec"), namespace)
    return namespace


def test_no_kernel_install_or_restart_guard() -> None:
    body = _source()
    for literal in ["pip install", "%pip", "!pip", "-m\", \"pip", "shutil.which(\"uv\")", "uv==0.8", "Restart session", "restart the runtime and"]:
        assert literal not in body, literal
    install = _kernel_cells()[0]
    assert "--python\", sys.executable" not in install  # the env is not built from the kernel's Python


def test_carried_lock_is_the_repository_lock_with_hashes() -> None:
    import hashlib

    namespace = _install_namespace()
    lock_text = LOCK.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert namespace["LOCK_TEXT"] == lock_text
    assert namespace["LOCK_SHA256"] == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    packages = [line for line in lock_text.splitlines() if line[:1].isalnum()]
    assert len(packages) == namespace["LOCKED_PACKAGES"] > 50
    assert all("==" in line and line.endswith(" \\") for line in packages)
    assert lock_text.count("--hash=sha256:") >= len(packages)
    assert "--generate-hashes" in lock_text and "--only-binary :all:" in lock_text and "x86_64-manylinux_2_28" in lock_text


def test_lock_keeps_the_notebooks_pins() -> None:
    namespace = _install_namespace()
    pins = [line.strip() for line in PINS_IN.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    assert namespace["PINS"] == pins
    locked = {line.split(" ")[0] for line in LOCK.read_text(encoding="utf-8").splitlines() if line[:1].isalnum()}
    for pin in pins:
        if "==" in pin:
            assert pin in locked, pin
    for exact in ["torch==2.14.0", "torchvision==0.29.0", "terratorch==1.2.13", "numpy==2.5.3", "tifffile==2026.9.15"]:
        assert exact in locked


def test_install_is_hash_locked_wheels_only_on_a_pinned_uv_and_managed_python() -> None:
    import re

    install = _kernel_cells()[0]
    namespace = _install_namespace()
    assert '"--require-hashes", "--only-binary", ":all:"' in install
    assert '"-r", str(lock_path)' in install
    assert '"venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON' in install
    assert namespace["MANAGED_PYTHON"] == "3.12.12"
    assert re.fullmatch(r"https://files\.pythonhosted\.org/packages/.+/uv-[0-9.]+-py3-none-manylinux_2_17_x86_64\.manylinux2014_x86_64\.whl", namespace["UV_URL"])
    assert re.fullmatch(r"[0-9a-f]{64}", namespace["UV_SHA256"]) and namespace["UV_BYTES"] > 1_000_000
    assert "hashlib.sha256(wheel).hexdigest() != UV_SHA256" in install
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in install
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert name in install


def test_model_cells_run_in_the_venv_python() -> None:
    install, bridge = _kernel_cells()[:2]
    assert 'EO_PYTHON = EO_ENV / "bin" / "python"' in install
    assert "IsolatedRuntime(EO_PYTHON)" in bridge
    assert 'MPLBACKEND="Agg"' in bridge
    assert 'for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):' in bridge


def test_no_cell_line_over_2000_characters() -> None:
    longest = max(len(line) for cell in _load()["cells"] for line in "".join(cell.get("source", [])).splitlines())
    assert longest <= 2000, longest


def test_revision_log_records_the_uv_migration() -> None:
    meta = _load()["metadata"]
    assert meta["workshop_revision"] == "0.2.2-candidate"
    assert meta["dimer"]["revision_log"][-1]["revision"] == "0.2.2-candidate"
