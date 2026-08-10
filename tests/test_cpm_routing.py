"""Tests for eopf_geozarr.cpm.routing (importable without eopf-cpm)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.cpm.routing import (
    looks_like_sentinel2,
    product_type_of,
    select_pipeline,
)


def make_tree(
    product_type: str | None = None,
    with_s2_structure: bool = False,
) -> xr.DataTree:
    """Build a minimal DataTree with optional CPM attrs and S2 group structure."""
    dt = xr.DataTree()
    if product_type is not None:
        dt.attrs = {"stac_discovery": {"properties": {"product:type": product_type}}}
    if with_s2_structure:
        ds = xr.Dataset({"b01": (["y", "x"], np.zeros((2, 2)))})
        for resolution in ("r10m", "r20m", "r60m"):
            dt[f"measurements/reflectance/{resolution}"] = ds
    return dt


@pytest.mark.parametrize(
    ("product_type", "with_s2_structure", "force", "expected"),
    [
        # Declared product type drives auto-detection.
        ("S02MSIL1C", False, None, "s2-optimized"),
        ("S02MSIL2A", False, None, "s2-optimized"),
        ("S01SIWGRH", False, None, "generic"),
        ("S03OLCERR", True, None, "generic"),  # declared type wins over structure
        # Structural fallback when no product type is declared.
        (None, True, None, "s2-optimized"),
        (None, False, None, "generic"),
        # Explicit forcing wins over everything.
        ("S02MSIL1C", True, False, "generic"),
        ("S03OLCERR", False, True, "s2-optimized"),
    ],
)
def test_select_pipeline(
    product_type: str | None,
    with_s2_structure: bool,
    force: bool | None,
    expected: str,
) -> None:
    """select_pipeline routes on declared type, then structure, unless forced."""
    tree = make_tree(product_type, with_s2_structure)
    assert select_pipeline(tree, force_s2_optimized=force) == expected


def test_product_type_of_missing_attrs() -> None:
    """product_type_of returns None for absent or malformed attribute chains."""
    assert product_type_of(xr.DataTree()) is None
    malformed = xr.DataTree()
    malformed.attrs = {"stac_discovery": "not-a-mapping"}
    assert product_type_of(malformed) is None
    no_type = xr.DataTree()
    no_type.attrs = {"stac_discovery": {"properties": {"product:type": 42}}}
    assert product_type_of(no_type) is None


def test_looks_like_sentinel2_requires_all_native_resolutions() -> None:
    """Structural detection requires all of r10m/r20m/r60m under reflectance."""
    partial = xr.DataTree()
    partial["measurements/reflectance/r10m"] = xr.Dataset({"b01": (["y", "x"], np.zeros((2, 2)))})
    assert not looks_like_sentinel2(partial)
