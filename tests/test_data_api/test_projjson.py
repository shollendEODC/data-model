"""
Tests for PROJ JSON Pydantic models

These tests validate the Pydantic models against various PROJ JSON examples
and ensure proper validation and serialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eopf_geozarr.data_api.geozarr.projjson import (
    Axis,
    BBox,
    BoundCRS,
    CompoundCRS,
    CoordinateMetadata,
    CoordinateSystem,
    DatumEnsemble,
    Ellipsoid,
    GeodeticCRS,
    GeodeticReferenceFrame,
    Id,
    PrimeMeridian,
    ProjectedCRS,
    SingleOperation,
    TemporalCRS,
    Unit,
    VerticalCRS,
)


class TestBasicModels:
    """Test basic building block models"""

    def test_id_model(self) -> None:
        """Test Id model validation"""
        # Valid ID with required fields
        id_data: dict[str, object] = {"authority": "EPSG", "code": 4326}
        id_obj: Id = Id.model_validate(id_data)
        assert id_obj.authority == "EPSG"
        assert id_obj.code == 4326

        # ID with string code
        id_data_str: dict[str, object] = {"authority": "EPSG", "code": "4326"}
        id_obj_str: Id = Id.model_validate(id_data_str)
        assert id_obj_str.code == "4326"

        # ID with optional fields
        id_full: dict[str, object] = {
            "authority": "EPSG",
            "code": 4326,
            "version": "10.095",
            "authority_citation": "EPSG Geodetic Parameter Dataset",
            "uri": "urn:ogc:def:crs:EPSG::4326",
        }
        id_obj_full: Id = Id.model_validate(id_full)
        assert id_obj_full.version == "10.095"
        assert id_obj_full.uri == "urn:ogc:def:crs:EPSG::4326"

        # Missing required field should raise ValidationError
        with pytest.raises(ValidationError):
            Id(authority="EPSG")  # type: ignore[call-arg]  # missing code (intentional)

    def test_unit_model(self) -> None:
        """Test Unit model validation"""
        unit_data: dict[str, object] = {
            "type": "Unit",
            "name": "metre",
            "conversion_factor": 1.0,
        }
        unit: Unit = Unit.model_validate(unit_data)
        assert unit.name == "metre"
        assert unit.conversion_factor == 1.0

        # With ID
        unit_with_id: dict[str, object] = {
            "type": "Unit",
            "name": "degree",
            "conversion_factor": 0.017453292519943295,
            "id": {"authority": "EPSG", "code": 9122},
        }
        unit = Unit.model_validate(unit_with_id)
        assert unit.id is not None
        assert unit.id.authority == "EPSG"
        assert unit.id.code == 9122

    def test_bbox_model(self) -> None:
        """Test BBox model validation"""
        bbox_data: dict[str, float] = {
            "east_longitude": 180.0,
            "west_longitude": -180.0,
            "south_latitude": -90.0,
            "north_latitude": 90.0,
        }
        bbox: BBox = BBox(**bbox_data)
        assert bbox.east_longitude == 180.0
        assert bbox.north_latitude == 90.0

        # Missing required field
        with pytest.raises(ValidationError):
            # missing latitude fields (intentional)
            BBox(east_longitude=180.0, west_longitude=-180.0)  # type: ignore[call-arg]

    def test_axis_model(self) -> None:
        """Test Axis model validation"""
        axis_data: dict[str, object] = {
            "type": "Axis",
            "name": "Geodetic latitude",
            "abbreviation": "Lat",
            "direction": "north",
        }
        axis: Axis = Axis.model_validate(axis_data)
        assert axis.name == "Geodetic latitude"
        assert axis.direction == "north"

        # Invalid direction should raise ValidationError
        with pytest.raises(ValidationError):
            Axis(
                type="Axis",
                name="Invalid",
                abbreviation="Inv",
                direction="invalid_direction",  # type: ignore[arg-type]  # invalid direction (intentional)
            )


class TestEllipsoidModel:
    """Test Ellipsoid model variations"""

    def test_ellipsoid_with_semi_axes(self) -> None:
        """Test ellipsoid with semi-major and semi-minor axes"""
        ellipsoid_data: dict[str, object] = {
            "type": "Ellipsoid",
            "name": "WGS 84",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245179,
        }
        ellipsoid: Ellipsoid = Ellipsoid.model_validate(ellipsoid_data)
        assert ellipsoid.name == "WGS 84"
        assert ellipsoid.semi_major_axis == 6378137.0

    def test_ellipsoid_with_inverse_flattening(self) -> None:
        """Test ellipsoid with inverse flattening"""
        ellipsoid_data: dict[str, object] = {
            "type": "Ellipsoid",
            "name": "WGS 84",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        }
        ellipsoid: Ellipsoid = Ellipsoid.model_validate(ellipsoid_data)
        assert ellipsoid.inverse_flattening == 298.257223563

    def test_ellipsoid_sphere(self) -> None:
        """Test spherical ellipsoid (equal radii)"""
        ellipsoid_data: dict[str, object] = {
            "type": "Ellipsoid",
            "name": "Sphere",
            "radius": 6371000.0,
        }
        ellipsoid: Ellipsoid = Ellipsoid.model_validate(ellipsoid_data)
        assert ellipsoid.radius == 6371000.0


class TestCoordinateSystemModel:
    """Test CoordinateSystem model"""

    def test_ellipsoidal_coordinate_system(self) -> None:
        """Test ellipsoidal coordinate system"""
        cs_data: dict[str, object] = {
            "type": "CoordinateSystem",
            "subtype": "ellipsoidal",
            "axis": [
                {
                    "type": "Axis",
                    "name": "Geodetic latitude",
                    "abbreviation": "Lat",
                    "direction": "north",
                    "unit": {
                        "type": "Unit",
                        "name": "degree",
                        "conversion_factor": 0.017453292519943295,
                    },
                },
                {
                    "type": "Axis",
                    "name": "Geodetic longitude",
                    "abbreviation": "Lon",
                    "direction": "east",
                    "unit": {
                        "type": "Unit",
                        "name": "degree",
                        "conversion_factor": 0.017453292519943295,
                    },
                },
            ],
        }
        cs: CoordinateSystem = CoordinateSystem.model_validate(cs_data)
        assert cs.subtype == "ellipsoidal"
        assert len(cs.axis) == 2
        assert cs.axis[0].name == "Geodetic latitude"

    def test_cartesian_coordinate_system(self) -> None:
        """Test Cartesian coordinate system"""
        cs_data: dict[str, object] = {
            "type": "CoordinateSystem",
            "subtype": "Cartesian",
            "axis": [
                {
                    "type": "Axis",
                    "name": "Easting",
                    "abbreviation": "E",
                    "direction": "east",
                    "unit": {"type": "Unit", "name": "metre", "conversion_factor": 1.0},
                },
                {
                    "type": "Axis",
                    "name": "Northing",
                    "abbreviation": "N",
                    "direction": "north",
                    "unit": {"type": "Unit", "name": "metre", "conversion_factor": 1.0},
                },
            ],
        }
        cs: CoordinateSystem = CoordinateSystem.model_validate(cs_data)
        assert cs.subtype == "Cartesian"
        assert cs.axis[0].direction == "east"
        assert cs.axis[1].direction == "north"


class TestCRSModels:
    """Test various CRS model types"""

    def test_geodetic_crs_wgs84(self) -> None:
        """Test WGS 84 geodetic CRS"""
        wgs84_data: dict[str, object] = {
            "type": "GeographicCRS",
            "name": "WGS 84",
            "datum": {
                "type": "GeodeticReferenceFrame",
                "name": "World Geodetic System 1984",
                "ellipsoid": {
                    "type": "Ellipsoid",
                    "name": "WGS 84",
                    "semi_major_axis": 6378137.0,
                    "inverse_flattening": 298.257223563,
                },
            },
            "coordinate_system": {
                "type": "CoordinateSystem",
                "subtype": "ellipsoidal",
                "axis": [
                    {
                        "type": "Axis",
                        "name": "Geodetic latitude",
                        "abbreviation": "Lat",
                        "direction": "north",
                        "unit": {
                            "type": "Unit",
                            "name": "degree",
                            "conversion_factor": 0.017453292519943295,
                        },
                    },
                    {
                        "type": "Axis",
                        "name": "Geodetic longitude",
                        "abbreviation": "Lon",
                        "direction": "east",
                        "unit": {
                            "type": "Unit",
                            "name": "degree",
                            "conversion_factor": 0.017453292519943295,
                        },
                    },
                ],
            },
            "id": {"authority": "EPSG", "code": 4326},
        }
        crs: GeodeticCRS = GeodeticCRS.model_validate(wgs84_data)
        assert crs.name == "WGS 84"
        assert crs.datum is not None
        assert crs.datum.name == "World Geodetic System 1984"
        assert crs.id is not None
        assert crs.id.code == 4326

    def test_projected_crs_utm(self) -> None:
        """Test UTM projected CRS"""
        utm_data: dict[str, object] = {
            "type": "ProjectedCRS",
            "name": "WGS 84 / UTM zone 33N",
            "base_crs": {
                "type": "GeographicCRS",
                "name": "WGS 84",
                "datum": {
                    "type": "GeodeticReferenceFrame",
                    "name": "World Geodetic System 1984",
                    "ellipsoid": {
                        "type": "Ellipsoid",
                        "name": "WGS 84",
                        "semi_major_axis": 6378137.0,
                        "inverse_flattening": 298.257223563,
                    },
                },
            },
            "conversion": {
                "type": "Conversion",
                "name": "UTM zone 33N",
                "method": {"type": "OperationMethod", "name": "Transverse Mercator"},
                "parameters": [
                    {
                        "type": "ParameterValue",
                        "name": "Latitude of natural origin",
                        "value": 0.0,
                        "unit": {
                            "type": "Unit",
                            "name": "degree",
                            "conversion_factor": 0.017453292519943295,
                        },
                    },
                    {
                        "type": "ParameterValue",
                        "name": "Longitude of natural origin",
                        "value": 15.0,
                        "unit": {
                            "type": "Unit",
                            "name": "degree",
                            "conversion_factor": 0.017453292519943295,
                        },
                    },
                ],
            },
            "coordinate_system": {
                "type": "CoordinateSystem",
                "subtype": "Cartesian",
                "axis": [
                    {
                        "type": "Axis",
                        "name": "Easting",
                        "abbreviation": "E",
                        "direction": "east",
                        "unit": {
                            "type": "Unit",
                            "name": "metre",
                            "conversion_factor": 1.0,
                        },
                    },
                    {
                        "type": "Axis",
                        "name": "Northing",
                        "abbreviation": "N",
                        "direction": "north",
                        "unit": {
                            "type": "Unit",
                            "name": "metre",
                            "conversion_factor": 1.0,
                        },
                    },
                ],
            },
        }
        crs: ProjectedCRS = ProjectedCRS.model_validate(utm_data)
        assert crs.name == "WGS 84 / UTM zone 33N"
        assert crs.base_crs.name == "WGS 84"
        assert crs.conversion.name == "UTM zone 33N"

    def test_compound_crs(self) -> None:
        """Test compound CRS with horizontal and vertical components"""
        compound_data: dict[str, object] = {
            "type": "CompoundCRS",
            "name": "WGS 84 + EGM96 height",
            "components": [
                {
                    "type": "GeographicCRS",
                    "name": "WGS 84",
                    "datum": {
                        "type": "GeodeticReferenceFrame",
                        "name": "World Geodetic System 1984",
                        "ellipsoid": {
                            "type": "Ellipsoid",
                            "name": "WGS 84",
                            "semi_major_axis": 6378137.0,
                            "inverse_flattening": 298.257223563,
                        },
                    },
                },
                {
                    "type": "VerticalCRS",
                    "name": "EGM96 height",
                    "datum": {"type": "VerticalReferenceFrame", "name": "EGM96 geoid"},
                },
            ],
        }
        crs: CompoundCRS = CompoundCRS.model_validate(compound_data)
        assert crs.name == "WGS 84 + EGM96 height"
        assert len(crs.components) == 2
        component_0 = crs.components[0]
        component_1 = crs.components[1]
        assert isinstance(component_0, GeodeticCRS)
        assert isinstance(component_1, VerticalCRS)
        assert component_0.name == "WGS 84"
        assert component_1.name == "EGM96 height"


class TestDatumEnsemble:
    """Test DatumEnsemble model"""

    def test_datum_ensemble_creation(self) -> None:
        """Test creation of datum ensemble"""
        ensemble_data: dict[str, object] = {
            "type": "DatumEnsemble",
            "name": "World Geodetic System 1984 ensemble",
            "members": [
                {"name": "World Geodetic System 1984 (Transit)"},
                {"name": "World Geodetic System 1984 (G730)"},
                {"name": "World Geodetic System 1984 (G873)"},
            ],
            "ellipsoid": {
                "type": "Ellipsoid",
                "name": "WGS 84",
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            },
            "accuracy": "2.0",
        }
        ensemble: DatumEnsemble = DatumEnsemble.model_validate(ensemble_data)
        assert ensemble.name == "World Geodetic System 1984 ensemble"
        assert len(ensemble.members) == 3
        assert ensemble.accuracy == "2.0"


class TestOperations:
    """Test operation models"""

    def test_coordinate_metadata(self) -> None:
        """Test coordinate metadata"""
        metadata_data: dict[str, object] = {
            "type": "CoordinateMetadata",
            "crs": {
                "type": "GeographicCRS",
                "name": "WGS 84",
                "datum": {
                    "type": "GeodeticReferenceFrame",
                    "name": "World Geodetic System 1984",
                    "ellipsoid": {
                        "type": "Ellipsoid",
                        "name": "WGS 84",
                        "semi_major_axis": 6378137.0,
                        "inverse_flattening": 298.257223563,
                    },
                },
            },
            "coordinateEpoch": 2020.0,
        }
        metadata: CoordinateMetadata = CoordinateMetadata.model_validate(metadata_data)
        assert metadata.coordinateEpoch == 2020.0
        assert isinstance(metadata.crs, GeodeticCRS)
        assert metadata.crs.name == "WGS 84"

    def test_single_operation(self) -> None:
        """Test single operation (transformation)"""
        operation_data: dict[str, object] = {
            "type": "Transformation",
            "name": "NAD27 to NAD83 (1)",
            "method": {"type": "OperationMethod", "name": "NADCON"},
            "parameters": [
                {
                    "type": "ParameterValue",
                    "name": "Latitude difference file",
                    "value": "conus.las",
                },
                {
                    "type": "ParameterValue",
                    "name": "Longitude difference file",
                    "value": "conus.los",
                },
            ],
            "accuracy": "0.15",
        }
        operation: SingleOperation = SingleOperation.model_validate(operation_data)
        assert operation.name == "NAD27 to NAD83 (1)"
        assert operation.accuracy == "0.15"
        assert operation.parameters is not None
        assert len(operation.parameters) == 2


class TestValidationEdgeCases:
    """Test validation edge cases and error handling"""

    def test_invalid_crs_type(self) -> None:
        """Test invalid CRS type raises ValidationError"""
        invalid_data: dict[str, object] = {
            "type": "InvalidCRS",  # Invalid type
            "name": "Invalid CRS",
        }
        with pytest.raises(ValidationError):
            GeodeticCRS(**invalid_data)  # type: ignore[arg-type]  # building model from dict[str, object] (intentional)

    def test_missing_required_fields(self) -> None:
        """Test missing required fields raise ValidationError"""
        # Missing name for CRS
        with pytest.raises(ValidationError):
            GeodeticCRS(type="GeographicCRS")  # type: ignore[call-arg]  # missing name (intentional)

        # Missing ellipsoid for geodetic reference frame
        with pytest.raises(ValidationError):
            # missing ellipsoid (intentional)
            GeodeticReferenceFrame(type="GeodeticReferenceFrame", name="Test Datum")  # type: ignore[call-arg]

    def test_mutually_exclusive_fields(self) -> None:
        """Test that mutually exclusive fields are properly validated"""
        # Cannot have both id and ids
        invalid_data: dict[str, object] = {
            "type": "Unit",
            "name": "metre",
            "conversion_factor": 1.0,
            "id": {"authority": "EPSG", "code": 9001},
            "ids": [{"authority": "EPSG", "code": 9001}],
        }
        # Note: This specific validation would need to be implemented in the model
        # For now, we'll just ensure the model can be created with either field

        with pytest.raises(ValidationError):
            Unit(**invalid_data)  # type: ignore[arg-type]  # building model from dict[str, object] (intentional)

        # Valid with id only
        valid_with_id: dict[str, object] = {
            "type": "Unit",
            "name": "metre",
            "conversion_factor": 1.0,
            "id": {"authority": "EPSG", "code": 9001},
        }
        unit: Unit = Unit.model_validate(valid_with_id)
        assert unit.id is not None
        assert unit.ids is None

        # Valid with ids only
        valid_with_ids: dict[str, object] = {
            "type": "Unit",
            "name": "metre",
            "conversion_factor": 1.0,
            "ids": [{"authority": "EPSG", "code": 9001}],
        }
        unit = Unit.model_validate(valid_with_ids)
        assert unit.ids is not None
        assert unit.id is None


class TestSerializationDeserialization:
    """Test JSON serialization and deserialization"""

    def test_round_trip_serialization(self) -> None:
        """Test that models can be serialized to JSON and back"""
        # Create a simple CRS
        crs_data: dict[str, object] = {
            "type": "GeographicCRS",
            "name": "WGS 84",
            "datum": {
                "type": "GeodeticReferenceFrame",
                "name": "World Geodetic System 1984",
                "ellipsoid": {
                    "type": "Ellipsoid",
                    "name": "WGS 84",
                    "semi_major_axis": 6378137.0,
                    "inverse_flattening": 298.257223563,
                },
            },
        }

        # Create model instance
        original_crs: GeodeticCRS = GeodeticCRS.model_validate(crs_data)

        # Deserialize back to model
        json_data: dict[str, object] = original_crs.model_dump()
        reconstructed_crs: GeodeticCRS = GeodeticCRS.model_validate(json_data)

        # Verify they're equivalent
        assert reconstructed_crs.name == original_crs.name
        assert reconstructed_crs.datum is not None
        assert original_crs.datum is not None
        assert reconstructed_crs.datum.name == original_crs.datum.name
        assert reconstructed_crs.datum.ellipsoid.name == original_crs.datum.ellipsoid.name

    def test_projjson_union_type(self) -> None:
        """Test that ProjJSON union type works correctly"""
        # Test with different types that should all be valid ProjJSON

        # Ellipsoid
        ellipsoid_data: dict[str, object] = {
            "type": "Ellipsoid",
            "name": "WGS 84",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        }
        ellipsoid: Ellipsoid = Ellipsoid.model_validate(ellipsoid_data)
        assert ellipsoid.name == "WGS 84"

        # CRS
        crs_data: dict[str, object] = {
            "type": "GeographicCRS",
            "name": "WGS 84",
            "datum": {
                "type": "GeodeticReferenceFrame",
                "name": "World Geodetic System 1984",
                "ellipsoid": ellipsoid_data,
            },
        }
        crs: GeodeticCRS = GeodeticCRS.model_validate(crs_data)
        assert crs.name == "WGS 84"


class TestRoundTripSerialization:
    """Test round-trip serialization with real PROJ JSON examples."""

    def test_projected_crs_round_trip(self, projected_crs_json: dict[str, object]) -> None:
        """Test round-trip serialization of projected CRS example."""
        # Parse JSON to Pydantic model
        from eopf_geozarr.data_api.geozarr.projjson import ProjectedCRS

        # Create model from JSON
        original_crs: ProjectedCRS = ProjectedCRS.model_validate(projected_crs_json)

        # Serialize back to dict
        serialized: dict[str, object] = original_crs.model_dump(exclude_none=True)

        # Create model from serialized data
        round_trip_crs: ProjectedCRS = ProjectedCRS.model_validate(serialized)

        # Verify key properties are preserved
        assert round_trip_crs.name == original_crs.name
        assert round_trip_crs.type == original_crs.type
        assert round_trip_crs.base_crs.name == original_crs.base_crs.name
        assert round_trip_crs.conversion.name == original_crs.conversion.name
        if original_crs.id:
            assert round_trip_crs.id is not None
            assert round_trip_crs.id.authority == original_crs.id.authority
            assert round_trip_crs.id.code == original_crs.id.code

    def test_bound_crs_round_trip(self, bound_crs_json: dict[str, object]) -> None:
        """Test round-trip serialization of bound CRS example."""
        from eopf_geozarr.data_api.geozarr.projjson import BoundCRS

        # Create model from JSON
        original_crs: BoundCRS = BoundCRS.model_validate(bound_crs_json)

        # Serialize back to dict
        serialized: dict[str, object] = original_crs.model_dump(exclude_none=True)

        # Create model from serialized data
        round_trip_crs: BoundCRS = BoundCRS.model_validate(serialized)

        # Verify key properties are preserved
        assert round_trip_crs.type == original_crs.type
        # A nested BoundCRS has no "name"; this example's source/target are named CRSs.
        assert not isinstance(round_trip_crs.source_crs, BoundCRS)
        assert not isinstance(round_trip_crs.target_crs, BoundCRS)
        assert not isinstance(original_crs.source_crs, BoundCRS)
        assert not isinstance(original_crs.target_crs, BoundCRS)
        assert round_trip_crs.source_crs.name == original_crs.source_crs.name
        assert round_trip_crs.target_crs.name == original_crs.target_crs.name
        assert round_trip_crs.transformation.name == original_crs.transformation.name

    def test_compound_crs_round_trip(self, compound_crs_json: dict[str, object]) -> None:
        """Test round-trip serialization of compound CRS example."""
        from eopf_geozarr.data_api.geozarr.projjson import CompoundCRS

        # Create model from JSON
        original_crs: CompoundCRS = CompoundCRS.model_validate(compound_crs_json)

        # Serialize back to dict
        serialized: dict[str, object] = original_crs.model_dump(exclude_none=True)

        # Create model from serialized data
        round_trip_crs: CompoundCRS = CompoundCRS.model_validate(serialized)

        # Verify key properties are preserved
        assert round_trip_crs.name == original_crs.name
        assert round_trip_crs.type == original_crs.type
        assert len(round_trip_crs.components) == len(original_crs.components)
        for i, component in enumerate(round_trip_crs.components):
            # Compound components are named CRSs, never a nested BoundCRS.
            assert not isinstance(component, BoundCRS)
            original_component = original_crs.components[i]
            assert not isinstance(original_component, BoundCRS)
            assert component.name == original_component.name

    def test_datum_ensemble_round_trip(self, datum_ensemble_json: dict[str, object]) -> None:
        """Test round-trip serialization of datum ensemble example."""
        from eopf_geozarr.data_api.geozarr.projjson import GeodeticCRS

        # Create model from JSON
        original_crs: GeodeticCRS = GeodeticCRS.model_validate(datum_ensemble_json)

        # Serialize back to dict
        serialized: dict[str, object] = original_crs.model_dump(exclude_none=True)

        # Create model from serialized data
        round_trip_crs: GeodeticCRS = GeodeticCRS.model_validate(serialized)

        # Verify key properties are preserved
        assert round_trip_crs.name == original_crs.name
        assert round_trip_crs.type == original_crs.type
        if original_crs.datum_ensemble:
            assert round_trip_crs.datum_ensemble is not None
            assert round_trip_crs.datum_ensemble.name == original_crs.datum_ensemble.name
            assert len(round_trip_crs.datum_ensemble.members) == len(
                original_crs.datum_ensemble.members
            )

    def test_transformation_round_trip(self, transformation_json: dict[str, object]) -> None:
        """Test round-trip serialization of transformation example."""
        from eopf_geozarr.data_api.geozarr.projjson import SingleOperation

        # Create model from JSON
        original_op: SingleOperation = SingleOperation.model_validate(transformation_json)

        # Serialize back to dict
        serialized: dict[str, object] = original_op.model_dump(exclude_none=True)

        # Create model from serialized data
        round_trip_op: SingleOperation = SingleOperation.model_validate(serialized)

        # Verify key properties are preserved
        assert round_trip_op.name == original_op.name
        assert round_trip_op.type == original_op.type
        assert round_trip_op.method.name == original_op.method.name
        if original_op.parameters:
            assert round_trip_op.parameters is not None
            assert len(round_trip_op.parameters) == len(original_op.parameters)

    def test_all_examples_round_trip(self, projjson_example: dict[str, object]) -> None:
        """Test that all PROJ JSON examples can be round-tripped without error."""

        # Map types to model classes
        type_mapping = {
            "GeographicCRS": GeodeticCRS,
            "GeodeticCRS": GeodeticCRS,
            "ProjectedCRS": ProjectedCRS,
            "BoundCRS": BoundCRS,
            "CompoundCRS": CompoundCRS,
            "VerticalCRS": VerticalCRS,
            "TemporalCRS": TemporalCRS,
            "Transformation": SingleOperation,
            "Conversion": SingleOperation,
            "DatumEnsemble": DatumEnsemble,
            "Ellipsoid": Ellipsoid,
            "PrimeMeridian": PrimeMeridian,
            "CoordinateMetadata": CoordinateMetadata,
        }

        # Get the model class based on type
        obj_type = projjson_example.get("type")
        assert isinstance(obj_type, str)

        model_class = type_mapping[obj_type]

        # Create model from JSON
        original_model = model_class(**projjson_example)

        # Serialize back to dict
        serialized = original_model.model_dump(exclude_none=True)

        # Create model from serialized data
        round_trip_model = model_class(**serialized)

        # Basic verification that the round-trip worked
        assert round_trip_model.type == original_model.type
        if hasattr(original_model, "name") and original_model.name:
            assert round_trip_model.name == original_model.name


if __name__ == "__main__":
    pytest.main([__file__])
