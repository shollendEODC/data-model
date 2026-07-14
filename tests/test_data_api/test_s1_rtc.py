"""
Round-trip and validation tests for Sentinel-1 GRD RTC pydantic-zarr models.

These tests verify that S1 RTC GeoZarr V3 store metadata can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss
4. Rejects invalid structures
"""

from __future__ import annotations

import copy

import pytest

from eopf_geozarr.data_api.s1_rtc import S1RtcRoot


def _nav(node: object, *keys: str) -> dict[str, object]:
    """Walk a path of string keys through a nested JSON dict (test helper).

    Narrows each level to ``dict`` so mutating the raw fixture stays type-safe.
    """
    for key in keys:
        assert isinstance(node, dict), f"expected a dict at {key!r}"
        node = node[key]
    assert isinstance(node, dict), "expected the leaf to be a dict"
    return node


def test_s1_rtc_roundtrip(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss."""
    model1 = S1RtcRoot.model_validate(s1_rtc_json_example)
    dumped = model1.model_dump()
    model2 = S1RtcRoot.model_validate(dumped)
    assert model1.model_dump() == model2.model_dump()


def test_s1_rtc_descending_present(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that the fixture has a descending orbit group."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    assert model.ascending is None


def test_s1_rtc_r10m_has_data_arrays(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that r10m contains vv, vh, border_mask and coordinate arrays."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    r10m = model.descending.r10m
    assert r10m.vv is not None
    assert r10m.vh is not None
    assert r10m.border_mask is not None
    assert "time" in r10m.members
    assert "absolute_orbit" in r10m.members
    assert "relative_orbit" in r10m.members
    assert "platform" in r10m.members


def test_s1_rtc_overview_levels(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that overview levels r20m-r720m exist and have vv/vh/border_mask."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    orbit = model.descending
    assert orbit is not None
    for level in ("r20m", "r60m", "r120m", "r360m", "r720m"):
        group = orbit.get_resolution(level)
        assert group is not None, f"Missing overview level {level}"
        assert "vv" in group.members
        assert "vh" in group.members
        assert "border_mask" in group.members
        # Overview levels should NOT have coordinate arrays
        assert "time" not in group.members


def test_s1_rtc_conditions(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that conditions group has gamma_area per-orbit arrays."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    conditions = model.descending.conditions
    assert conditions is not None
    gamma_keys = [k for k in conditions.members if k.startswith("gamma_area_")]
    assert len(gamma_keys) >= 1


def test_s1_rtc_orbit_attrs(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that orbit group attributes contain required conventions and metadata."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    attrs = model.descending.attributes
    assert len(attrs.zarr_conventions) == 3
    assert attrs.proj_code.startswith("EPSG:")
    assert attrs.spatial_dimensions == ["y", "x"]
    assert len(attrs.spatial_bbox) == 4
    assert len(attrs.multiscales.layout) == 6
    assert attrs.multiscales.layout[0].asset == "r10m"


def test_s1_rtc_rejects_no_orbit(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject a store with no orbit groups."""
    data = copy.deepcopy(s1_rtc_json_example)
    data["members"] = {}
    with pytest.raises(Exception, match="at least one orbit"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_missing_r10m(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject an orbit group that lacks r10m."""
    data = copy.deepcopy(s1_rtc_json_example)
    del _nav(data, "members", "descending", "members")["r10m"]
    with pytest.raises(Exception, match="r10m"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_missing_convention_uuid(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject orbit attrs with missing convention UUIDs."""
    data = copy.deepcopy(s1_rtc_json_example)
    _nav(data, "members", "descending", "attributes")["zarr_conventions"] = [
        {"uuid": "fake-uuid", "name": "fake"}
    ]
    with pytest.raises(Exception, match="Missing required zarr_conventions"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_bad_spatial_dimensions(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject orbit attrs with wrong spatial:dimensions."""
    data = copy.deepcopy(s1_rtc_json_example)
    _nav(data, "members", "descending", "attributes")["spatial:dimensions"] = ["lat", "lon"]
    with pytest.raises(Exception, match="spatial:dimensions"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_conditions_without_gamma_area(
    s1_rtc_json_example: dict[str, object],
) -> None:
    """Reject conditions group with no gamma_area_* arrays."""
    data = copy.deepcopy(s1_rtc_json_example)
    cond_members = _nav(data, "members", "descending", "members", "conditions", "members")
    # Replace all keys with non-gamma_area names
    _nav(data, "members", "descending", "members", "conditions")["members"] = {
        "some_other": next(iter(cond_members.values()))
    }
    with pytest.raises(Exception, match="gamma_area"):
        S1RtcRoot.model_validate(data)
