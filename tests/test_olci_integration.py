"""Integration tests for convert_olci_optimized."""

from __future__ import annotations

import numpy as np
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_converter import convert_olci_optimized


def build_synthetic_olci(rows: int = 512, cols: int = 480) -> xr.DataTree:
    """Minimal synthetic OLCI L1 EFR datatree (measurements only)."""
    rng = np.random.default_rng(0)
    lat = np.linspace(40, 41, rows * cols).reshape(rows, cols)
    lon = np.linspace(10, 11, rows * cols).reshape(rows, cols)
    alt = np.zeros((rows, cols), dtype="int16")
    data: dict[str, xr.DataArray] = {}
    for i in range(1, 22):
        name = f"oa{i:02d}_radiance"
        arr = xr.DataArray(
            rng.integers(0, 6000, (rows, cols)).astype("uint16"),
            dims=("rows", "columns"),
            attrs={
                "scale_factor": 0.0139,
                "add_offset": 0.0,
                "standard_name": "toa_upwelling_spectral_radiance",
                "coordinates": "latitude longitude altitude",
            },
        )
        data[name] = arr
    ds = xr.Dataset(
        data,
        coords={
            "latitude": (("rows", "columns"), lat, {"standard_name": "latitude"}),
            "longitude": (("rows", "columns"), lon, {"standard_name": "longitude"}),
            "altitude": (("rows", "columns"), alt, {"standard_name": "altitude"}),
        },
    )
    return xr.DataTree.from_dict({"/measurements": ds})


def test_convert_olci_writes_measurements(tmp_path: object) -> None:
    """Native-resolution measurements group must contain all 21 radiance bands."""
    import zarr

    dt = build_synthetic_olci()
    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out)

    g = zarr.open_group(out, mode="r")
    # native measurements present
    assert "measurements" in g
    # all 21 bands at native res
    meas = g["measurements"]
    for i in range(1, 22):
        assert f"oa{i:02d}_radiance" in meas


def test_convert_olci_creates_overviews(tmp_path: object) -> None:
    """At least one /2-decimated overview subgroup must be written under measurements."""
    import zarr

    dt = build_synthetic_olci(rows=512, cols=480)
    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out, min_dimension=256)

    g = zarr.open_group(out, mode="r")
    # at least one decimated overview level exists under measurements
    meas_item = g["measurements"]
    assert isinstance(meas_item, zarr.Group)
    subgroups = list(meas_item.group_keys())
    assert len(subgroups) >= 1


def test_convert_olci_returns_datatree(tmp_path: object) -> None:
    """convert_olci_optimized must return an xr.DataTree backed by the output store."""
    dt = build_synthetic_olci(rows=256, cols=256)
    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    result = convert_olci_optimized(dt, output_path=out, min_dimension=256)
    assert isinstance(result, xr.DataTree)
    assert "/measurements" in result.groups


def test_convert_olci_conditions_quality_passthrough(tmp_path: object) -> None:
    """conditions and quality groups, when present, are copied through unchanged."""
    import zarr

    # Build a tree with conditions and quality groups
    rng = np.random.default_rng(1)
    rows, cols = 128, 128
    lat = np.linspace(40, 41, rows * cols).reshape(rows, cols)
    lon = np.linspace(10, 11, rows * cols).reshape(rows, cols)
    alt = np.zeros((rows, cols), dtype="int16")

    meas_data: dict[str, xr.DataArray] = {
        "oa01_radiance": xr.DataArray(
            rng.integers(0, 6000, (rows, cols)).astype("uint16"),
            dims=("rows", "columns"),
        )
    }
    meas_ds = xr.Dataset(
        meas_data,
        coords={
            "latitude": (("rows", "columns"), lat),
            "longitude": (("rows", "columns"), lon),
            "altitude": (("rows", "columns"), alt),
        },
    )
    cond_ds = xr.Dataset(
        {
            "wind_speed": xr.DataArray(
                rng.random((rows, cols)).astype("float32"), dims=("rows", "columns")
            )
        }
    )
    quality_ds = xr.Dataset(
        {
            "flags": xr.DataArray(
                rng.integers(0, 255, (rows, cols)).astype("uint8"), dims=("rows", "columns")
            )
        }
    )
    dt = xr.DataTree.from_dict(
        {
            "/measurements": meas_ds,
            "/conditions": cond_ds,
            "/quality": quality_ds,
        }
    )

    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out, min_dimension=64)

    g = zarr.open_group(out, mode="r")
    assert "conditions" in g
    assert "quality" in g
