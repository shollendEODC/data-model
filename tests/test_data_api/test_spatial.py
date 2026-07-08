"""Tests for the Spatial Zarr Convention models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eopf_geozarr.data_api.geozarr.spatial import Spatial


class TestSpatial:
    """Test the Spatial model class."""

    def test_minimal_required_fields(self) -> None:
        """Test creation with only required fields."""
        data: dict[str, object] = {"spatial:dimensions": ["y", "x"]}
        spatial = Spatial.model_validate(data)

        assert spatial.dimensions == ["y", "x"]
        assert spatial.bbox is None
        assert spatial.transform_type == "affine"  # Default value
        assert spatial.transform is None
        assert spatial.shape is None
        assert spatial.registration == "pixel"  # Default value

    def test_missing_required_dimensions(self) -> None:
        """Test that missing dimensions field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Spatial()  # type: ignore[call-arg]  # intentionally missing required field

        assert "spatial:dimensions" in str(exc_info.value)

    def test_full_spatial_metadata(self) -> None:
        """Test creation with all fields populated."""
        data: dict[str, object] = {
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": [500000.0, 4900000.0, 600000.0, 5000000.0],
            "spatial:transform_type": "affine",
            "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
            "spatial:shape": [1000, 1000],
            "spatial:registration": "pixel",
        }

        spatial = Spatial.model_validate(data)

        assert spatial.dimensions == ["y", "x"]
        assert spatial.bbox == [500000.0, 4900000.0, 600000.0, 5000000.0]
        assert spatial.transform_type == "affine"
        assert spatial.transform == [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0]
        assert spatial.shape == [1000, 1000]
        assert spatial.registration == "pixel"

    def test_3d_spatial_data(self) -> None:
        """Test spatial model with 3D data."""
        data: dict[str, object] = {
            "spatial:dimensions": ["z", "y", "x"],
            "spatial:bbox": [500000.0, 4900000.0, 0.0, 600000.0, 5000000.0, 100.0],
            "spatial:shape": [10, 1000, 1000],
        }

        spatial = Spatial.model_validate(data)

        assert spatial.dimensions == ["z", "y", "x"]
        assert spatial.bbox == [500000.0, 4900000.0, 0.0, 600000.0, 5000000.0, 100.0]
        assert spatial.shape == [10, 1000, 1000]

    def test_serialization_by_alias(self) -> None:
        """Test that serialization uses aliases (spatial: prefixes)."""
        data: dict[str, object] = {
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": [0.0, 0.0, 100.0, 100.0],
            "spatial:transform": [1.0, 0.0, 0.0, 0.0, -1.0, 100.0],
            "spatial:shape": [100, 100],
        }

        spatial = Spatial.model_validate(data)
        result = spatial.model_dump()

        # Should serialize with spatial: prefixes
        assert "spatial:dimensions" in result
        assert "spatial:bbox" in result
        assert "spatial:transform" in result
        assert "spatial:shape" in result
        assert "spatial:transform_type" in result
        assert "spatial:registration" in result

        # Should not have unprefixed versions
        assert "dimensions" not in result
        assert "bbox" not in result
        assert "transform" not in result
        assert "shape" not in result

    def test_none_fields_excluded(self) -> None:
        """Test that None fields are excluded from serialization."""
        data: dict[str, object] = {"spatial:dimensions": ["y", "x"]}
        spatial = Spatial.model_validate(data)
        result = spatial.model_dump()

        # None fields should be excluded
        assert "spatial:bbox" not in result
        assert "spatial:transform" not in result
        assert "spatial:shape" not in result

        # Default values should be included
        assert result["spatial:transform_type"] == "affine"
        assert result["spatial:registration"] == "pixel"

    def test_node_registration(self) -> None:
        """Test node registration type."""
        data: dict[str, object] = {"spatial:dimensions": ["y", "x"], "spatial:registration": "node"}

        spatial = Spatial.model_validate(data)
        assert spatial.registration == "node"

    def test_non_affine_transform_type(self) -> None:
        """Test non-affine transform types."""
        data: dict[str, object] = {
            "spatial:dimensions": ["y", "x"],
            "spatial:transform_type": "rpc",
        }

        spatial = Spatial.model_validate(data)
        assert spatial.transform_type == "rpc"

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed."""
        data: dict[str, object] = {
            "spatial:dimensions": ["y", "x"],
            "custom_field": "custom_value",
            "spatial:custom": "also_allowed",
        }

        spatial = Spatial.model_validate(data)
        result = spatial.model_dump()

        assert result["custom_field"] == "custom_value"
        assert result["spatial:custom"] == "also_allowed"

    def test_roundtrip_serialization(self) -> None:
        """Test that serialization and deserialization preserves data."""
        original_data: dict[str, object] = {
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": [500000.0, 4900000.0, 600000.0, 5000000.0],
            "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
            "spatial:shape": [1000, 1000],
            "spatial:registration": "node",
            "spatial:transform_type": "affine",
        }

        # Create model, serialize, then recreate
        spatial1 = Spatial.model_validate(original_data)
        serialized = spatial1.model_dump()
        spatial2 = Spatial(**serialized)

        # Should be equivalent
        assert spatial1.dimensions == spatial2.dimensions
        assert spatial1.bbox == spatial2.bbox
        assert spatial1.transform == spatial2.transform
        assert spatial1.shape == spatial2.shape
        assert spatial1.registration == spatial2.registration
        assert spatial1.transform_type == spatial2.transform_type

    def test_invalid_dimensions_none(self) -> None:
        """Test that None dimensions raise ValidationError."""
        data: dict[str, object] = {"spatial:dimensions": None}  # intentionally invalid value
        with pytest.raises(ValidationError):
            Spatial.model_validate(data)

    def test_empty_dimensions_not_allowed(self) -> None:
        """Test that empty dimensions raise ValidationError."""
        empty_data: dict[str, object] = {"spatial:dimensions": []}
        with pytest.raises(ValidationError) as exc_info:
            Spatial.model_validate(empty_data)

        assert "spatial:dimensions must contain at least one dimension" in str(exc_info.value)
        data: dict[str, object] = {
            "spatial:dimensions": ["y", "x"],
            "spatial:transform_type": "affine",
            "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0],  # Only 5 elements
        }

        # Currently this will pass, but in the future we might want validation
        spatial = Spatial.model_validate(data)
        assert spatial.transform is not None
        assert len(spatial.transform) == 5  # Current behavior

        # Future: might want to validate for exactly 6 elements for affine
        # with pytest.raises(ValidationError):
        #     Spatial(**data)
