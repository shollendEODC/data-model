"""Tests for olci_multiscale: decimate_swath, reduce_swath, swath_spatial_attrs."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_multiscale import (
    decimate_swath,
    reduce_swath,
    swath_spatial_attrs,
)


def _swath(rows: int = 8, cols: int = 6) -> xr.Dataset:
    """Minimal synthetic swath dataset with one radiance band and two coords."""
    rad = xr.DataArray(
        np.arange(rows * cols, dtype="uint16").reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={
            "scale_factor": 0.5,
            "units": "mW.m-2.sr-1.nm-1",
            "_FillValue": 65535,
        },
    )
    lat = xr.DataArray(
        np.linspace(0, 1, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "latitude"},
    )
    lon = xr.DataArray(
        np.linspace(10, 11, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude"},
    )
    return xr.Dataset(
        {"oa01_radiance": rad},
        coords={"latitude": lat, "longitude": lon},
    )


# ---------------------------------------------------------------------------
# decimate_swath tests
# ---------------------------------------------------------------------------


def test_decimate_halves_each_axis() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].shape == (4, 3)
    assert out["latitude"].shape == (4, 3)
    assert out["longitude"].shape == (4, 3)


def test_decimate_takes_every_other_pixel() -> None:
    ds = _swath(8, 6)
    out = decimate_swath(ds, factor=2)
    # top-left pixel is preserved exactly (no averaging)
    assert int(out["oa01_radiance"].values[0, 0]) == 0
    assert float(out["latitude"].values[0, 0]) == 0.0
    # interior pixel: stride-2 decimation means out[1, 1] comes from original [2, 2]
    assert int(out["oa01_radiance"].values[1, 1]) == int(ds["oa01_radiance"].values[2, 2])


def test_decimate_preserves_attrs() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].attrs["scale_factor"] == 0.5
    assert out["latitude"].attrs["standard_name"] == "latitude"


def test_decimate_factor_1_returns_unchanged() -> None:
    ds = _swath(8, 6)
    out = decimate_swath(ds, factor=1)
    assert out["oa01_radiance"].shape == (8, 6)


def test_decimate_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="factor must be >= 1"):
        decimate_swath(_swath(), factor=0)


# ---------------------------------------------------------------------------
# reduce_swath tests
# ---------------------------------------------------------------------------


def test_reduce_swath_halves_each_axis() -> None:
    """reduce_swath must produce output with halved spatial dims."""
    out = reduce_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].shape == (4, 3)
    assert out["latitude"].shape == (4, 3)
    assert out["longitude"].shape == (4, 3)


def test_reduce_swath_radiance_is_averaged_not_decimated() -> None:
    """Radiance must be block-averaged; top-left output != top-left input (unless accident)."""
    rng = np.random.default_rng(42)
    rad_data = rng.integers(100, 200, (8, 6)).astype("uint16")
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": 65535},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # block [0:2, 0:2] averages to a value; verify it's a rounded mean
    expected_block = int(np.round(rad_data[0:2, 0:2].astype("float64").mean()))
    assert int(out["oa01_radiance"].values[0, 0]) == expected_block


def test_reduce_swath_coordinates_decimated() -> None:
    """Coordinate arrays must be decimated (stride), not averaged."""
    ds = _swath(8, 6)
    out = reduce_swath(ds, factor=2)
    # lat[0,0] in output == lat[0,0] in input
    assert float(out["latitude"].values[0, 0]) == float(ds["latitude"].values[0, 0])
    # lat[1,1] in output == lat[2,2] in input (stride-2)
    assert float(out["latitude"].values[1, 1]) == float(ds["latitude"].values[2, 2])


def test_reduce_swath_fill_value_preserved_in_all_fill_block() -> None:
    """A block where all pixels are fill must produce fill output, not 65535.0 average."""
    fill = 65535
    rad_data = np.ones((4, 4), dtype="uint16") * fill
    # put some non-fill values only in lower-right block
    rad_data[2:4, 2:4] = 100
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": fill},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # top-left block: all fill -> output must be fill
    assert int(out["oa01_radiance"].values[0, 0]) == fill
    # bottom-right block: all 100 -> output must be 100
    assert int(out["oa01_radiance"].values[1, 1]) == 100


def test_reduce_swath_preserves_attrs() -> None:
    """reduce_swath must carry over variable attributes."""
    out = reduce_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].attrs["scale_factor"] == 0.5
    assert out["latitude"].attrs["standard_name"] == "latitude"


def test_reduce_swath_factor_1_returns_unchanged() -> None:
    ds = _swath(8, 6)
    out = reduce_swath(ds, factor=1)
    assert out["oa01_radiance"].shape == (8, 6)


def test_reduce_swath_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="factor must be >= 1"):
        reduce_swath(_swath(), factor=0)


def test_reduce_swath_non_swath_var_passthrough() -> None:
    """Variables that don't span (rows, columns) must pass through unchanged."""
    ds = _swath(8, 6)
    scalar = xr.DataArray(42.0, attrs={"info": "scalar"})
    ds = ds.assign({"extra": scalar})
    out = reduce_swath(ds, factor=2)
    assert "extra" in out
    assert float(out["extra"].values) == 42.0


# ---------------------------------------------------------------------------
# swath_spatial_attrs tests
# ---------------------------------------------------------------------------


def test_swath_spatial_attrs_has_no_transform() -> None:
    attrs = swath_spatial_attrs()
    assert attrs["spatial:dimensions"] == ["rows", "columns"]
    assert attrs.get("spatial:registration") == "pixel"
    assert "spatial:transform" not in attrs
    assert "spatial:bbox" not in attrs
