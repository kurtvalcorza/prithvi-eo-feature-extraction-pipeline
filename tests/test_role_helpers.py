"""Offline tests for the public validation-stage helpers and the package surface."""

from __future__ import annotations

import prithvi_eo_feature_extraction_pipeline as pkg
from conftest import synthetic_records
from prithvi_eo_feature_extraction_pipeline import BANDS, IGNORE_INDEX, IMAGE_SIZE, INPUT_SCHEMA, validate_inputs


def test_input_schema_names_the_contract():
    assert INPUT_SCHEMA["bands"] == list(BANDS) and len(BANDS) == 6
    assert INPUT_SCHEMA["image_size"] == IMAGE_SIZE == 512 and INPUT_SCHEMA["ignore_index"] == IGNORE_INDEX == -1
    assert "any six-band 512 × 512 array is embedded and probed" in INPUT_SCHEMA["validation"]


def test_validate_inputs_reports_the_record():
    report = validate_inputs(synthetic_records(1)[0])
    assert report["id"] == "chip-000" and report["shape"] == (6, 512, 512) and report["has_label"]


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
    assert "PrithviFeaturePipeline" in pkg.__all__ and "audit_pickle" in pkg.__all__ and "validate_stacks" in pkg.__all__
