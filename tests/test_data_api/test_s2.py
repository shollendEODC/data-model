"""
Round-trip tests for Sentinel-2 pydantic-zarr integrated models.

These tests verify that Sentinel-2 data can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss

Note: Documentation code examples are tested separately via pytest-examples
from the markdown files in docs/models/sentinel2.md
"""

from eopf_geozarr.data_api.s2 import Sentinel2Root


def test_sentinel2_roundtrip(s2_json_example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss"""
    model1 = Sentinel2Root.model_validate(s2_json_example)
    dumped = model1.model_dump()
    model2 = Sentinel2Root.model_validate(dumped)
    assert model1.model_dump() == model2.model_dump()
