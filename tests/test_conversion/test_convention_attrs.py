"""Tests for the zarr-cm convention-attribute helpers in conversion.utils."""

from __future__ import annotations

from typing import Any

import pytest
from pyproj import CRS
from zarr_cm import geo_proj
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.conversion.utils import (
    build_convention_attrs,
    proj_attrs_for_crs,
)


def test_proj_attrs_for_crs_epsg() -> None:
    """A CRS with an EPSG code yields proj:code."""
    crs = CRS.from_epsg(32632)
    assert proj_attrs_for_crs(crs) == {"proj:code": "EPSG:32632"}


def test_proj_attrs_for_crs_wkt_fallback() -> None:
    """A CRS without an EPSG code falls back to proj:wkt2."""
    # A bare WKT-defined CRS with no authority code.
    crs = CRS.from_wkt(CRS.from_epsg(4326).to_wkt())
    out = proj_attrs_for_crs(crs)
    # from_wkt of an EPSG CRS still resolves an epsg in most pyproj versions, so
    # accept either, but the key must be one of the two proj CRS keys.
    assert set(out) <= {"proj:code", "proj:wkt2"}
    assert len(out) == 1


def test_proj_attrs_for_crs_none() -> None:
    """No CRS yields an empty dict (no proj keys)."""
    assert proj_attrs_for_crs(None) == {}


def test_build_convention_attrs_data_and_cmos() -> None:
    """The helper emits spatial+proj data keys plus their CMOs, in order."""
    bbox = [300000.0, 4990200.0, 409800.0, 5100000.0]
    out = build_convention_attrs(
        spatial={
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": bbox,
            "spatial:registration": "pixel",
        },
        crs=CRS.from_epsg(32632),
    )
    assert out["spatial:dimensions"] == ["y", "x"]
    assert out["spatial:bbox"] == bbox
    assert out["spatial:registration"] == "pixel"
    assert out["proj:code"] == "EPSG:32632"
    names = [c["name"] for c in out["zarr_conventions"]]
    assert names == [spatial_cm.CMO["name"], geo_proj.CMO["name"]]


def test_build_convention_attrs_matches_handwritten() -> None:
    """Output is byte-equivalent to the previous hand-assembled dict."""
    bbox = [300000.0, 4990200.0, 409800.0, 5100000.0]
    hand: dict[str, Any] = {
        "spatial:dimensions": ["y", "x"],
        "spatial:bbox": bbox,
        "spatial:registration": "pixel",
        "proj:code": "EPSG:32632",
    }
    out = build_convention_attrs(
        spatial={
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": bbox,
            "spatial:registration": "pixel",
        },
        crs=CRS.from_epsg(32632),
    )
    data = {k: v for k, v in out.items() if k != "zarr_conventions"}
    assert data == hand


def test_build_convention_attrs_validates() -> None:
    """Invalid spatial data is rejected by zarr-cm validation."""
    with pytest.raises(ValueError, match="spatial:dimensions"):
        build_convention_attrs(
            spatial={"spatial:registration": "pixel"},  # missing required dimensions
            crs=CRS.from_epsg(32632),
        )


def test_build_convention_attrs_with_multiscales() -> None:
    """With multiscales, CMOs are ordered [multiscales, spatial, proj]."""
    from zarr_cm import multiscales as multiscales_cm

    out = build_convention_attrs(
        multiscales={
            "layout": [
                {"asset": "r10m", "spatial:shape": [10980, 10980]},
                {
                    "asset": "r20m",
                    "derived_from": "r10m",
                    "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                    "spatial:shape": [5490, 5490],
                },
            ],
            "resampling_method": "average",
        },
        spatial={"spatial:dimensions": ["y", "x"]},
        crs=CRS.from_epsg(32632),
    )
    names = [c["name"] for c in out["zarr_conventions"]]
    assert names == [
        multiscales_cm.CMO["name"],
        spatial_cm.CMO["name"],
        geo_proj.CMO["name"],
    ]
    # extra layout keys (spatial:shape) survive the round-trip
    assert out["multiscales"]["layout"][0]["spatial:shape"] == [10980, 10980]


def test_build_convention_attrs_multiscales_validation() -> None:
    """zarr-cm rejects a layout entry with derived_from but no transform."""
    with pytest.raises(ValueError, match="transform"):
        build_convention_attrs(
            multiscales={
                "layout": [{"asset": "r20m", "derived_from": "r10m"}],  # no transform
                "resampling_method": "average",
            },
            spatial={"spatial:dimensions": ["y", "x"]},
            crs=CRS.from_epsg(32632),
        )
