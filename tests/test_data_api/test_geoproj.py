from __future__ import annotations

import jsondiff
import pytest
from pydantic import ValidationError
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.geozarr.geoproj import GeoProj, Proj
from tests.test_data_api.conftest import view_json_diff


class TestProj:
    """Test the Proj model class."""

    def test_proj_with_epsg_code(self) -> None:
        """Test creation with EPSG code."""
        proj = Proj.model_validate({"proj:code": "EPSG:4326"})

        assert proj.code == "EPSG:4326"
        assert proj.wkt2 is None
        assert proj.projjson is None

    def test_proj_with_wkt2(self) -> None:
        """Test creation with WKT2 string."""
        wkt2_example = 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984"]]'
        proj = Proj.model_validate({"proj:wkt2": wkt2_example})

        assert proj.wkt2 == wkt2_example
        assert proj.code is None
        assert proj.projjson is None

    def test_proj_with_projjson(self) -> None:
        """Test creation with PROJ JSON."""
        projjson_data = {
            "$schema": "https://proj.org/schemas/v0.7/projjson.schema.json",
            "type": "GeographicCRS",
            "name": "WGS 84",
        }
        proj = Proj.model_validate({"proj:projjson": projjson_data})

        assert proj.projjson is not None
        assert proj.code is None
        assert proj.wkt2 is None

    def test_proj_validation_error_no_crs(self) -> None:
        """Test that missing all CRS fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Proj()  # pyright: ignore[reportCallIssue]  # no-args construction tests the validation error

        assert "At least one of proj:code, proj:wkt2, or proj:projjson must be provided" in str(
            exc_info.value
        )

    def test_proj_multiple_crs_fields(self) -> None:
        """Test that multiple CRS fields can be provided."""
        wkt2_example = 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984"]]'
        proj = Proj.model_validate({"proj:code": "EPSG:4326", "proj:wkt2": wkt2_example})

        assert proj.code == "EPSG:4326"
        assert proj.wkt2 == wkt2_example
        assert proj.projjson is None

    def test_proj_serialization_by_alias(self) -> None:
        """Test that serialization uses aliases (proj: prefixes)."""
        proj = Proj.model_validate({"proj:code": "EPSG:32633"})
        result = proj.model_dump()

        # Should serialize with proj: prefix
        assert "proj:code" in result
        assert result["proj:code"] == "EPSG:32633"

        # Should not have unprefixed version
        assert "code" not in result

    def test_proj_none_fields_excluded(self) -> None:
        """Test that None fields are excluded from serialization."""
        proj = Proj.model_validate({"proj:code": "EPSG:4326"})
        result = proj.model_dump()

        # None fields should be excluded
        assert "proj:wkt2" not in result
        assert "proj:projjson" not in result

        # Provided field should be included
        assert result["proj:code"] == "EPSG:4326"

    def test_proj_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed."""
        proj = Proj.model_validate(
            {
                "proj:code": "EPSG:4326",
                "custom_field": "custom_value",
                "proj:custom": "also_allowed",
            }
        )
        result = proj.model_dump()

        assert result["proj:code"] == "EPSG:4326"
        assert result["custom_field"] == "custom_value"
        assert result["proj:custom"] == "also_allowed"

    def test_proj_roundtrip_serialization(self) -> None:
        """Test that serialization and deserialization preserves data."""
        original_data = {"proj:code": "EPSG:32633", "proj:wkt2": 'PROJCRS["WGS 84 / UTM zone 33N"]'}

        # Create model, serialize, then recreate
        proj1 = Proj.model_validate(original_data)
        serialized = proj1.model_dump()
        proj2 = Proj(**serialized)

        # Should be equivalent
        assert proj1.code == proj2.code
        assert proj1.wkt2 == proj2.wkt2
        assert proj1.projjson == proj2.projjson


class TestBackwardsCompatibility:
    """Test backwards compatibility through GeoProj alias."""

    def test_geoproj_is_proj_alias(self) -> None:
        """Test that GeoProj is an alias for Proj."""
        assert GeoProj is Proj

    def test_geoproj_functionality(self) -> None:
        """Test that GeoProj works exactly like Proj."""
        # Create using both classes
        proj_instance = Proj.model_validate({"proj:code": "EPSG:4326"})
        geoproj_instance = GeoProj.model_validate({"proj:code": "EPSG:4326"})

        # Should be instances of the same class
        assert type(proj_instance) is type(geoproj_instance)
        assert isinstance(proj_instance, Proj)
        assert isinstance(geoproj_instance, Proj)

        # Should have same attributes
        assert proj_instance.code == geoproj_instance.code
        assert proj_instance.model_dump() == geoproj_instance.model_dump()


def test_geoproj_roundtrip(geoproj_example: dict[str, object]) -> None:
    """Test roundtrip serialization with existing examples (backwards compatibility)."""
    value_tup = tuplify_json(geoproj_example)
    attrs_json = value_tup["attributes"]
    model = GeoProj(**attrs_json)
    observed = model.model_dump()
    expected = attrs_json
    assert jsondiff.diff(expected, observed) == {}, view_json_diff(expected, observed)


def test_proj_roundtrip(geoproj_example: dict[str, object]) -> None:
    """Test roundtrip serialization with existing examples using new Proj class."""
    value_tup = tuplify_json(geoproj_example)
    attrs_json = value_tup["attributes"]
    model = Proj(**attrs_json)
    observed = model.model_dump()
    expected = attrs_json
    assert jsondiff.diff(expected, observed) == {}, view_json_diff(expected, observed)
