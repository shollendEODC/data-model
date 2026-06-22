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
    """Overview subgroups must only be written when BOTH post-decimation dims >= min_dimension.

    Case 1 — 512x480 with min_dimension=256:
        480//2 = 240 < 256, so the guard fires immediately → zero overview levels.

    Case 2 — 1024x1024 with min_dimension=256:
        1024//2=512>=256 → level r2 (512x512)
        512//2=256>=256  → level r4 (256x256)
        256//2=128<256   → stop
        Exactly two levels; smallest must be exactly 256x256.
    """
    import zarr

    # --- Case 1: 512x480, min_dimension=256 → zero overview levels ---
    dt1 = build_synthetic_olci(rows=512, cols=480)
    out1 = str(tmp_path / "olci_case1.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt1, output_path=out1, min_dimension=256)

    g1 = zarr.open_group(out1, mode="r")
    meas1 = g1["measurements"]
    assert isinstance(meas1, zarr.Group)
    subgroups1 = list(meas1.group_keys())
    assert len(subgroups1) == 0, (
        f"Expected 0 overview levels for 512x480 at min_dimension=256, got {subgroups1}"
    )

    # --- Case 2: 1024x1024, min_dimension=256 → exactly two valid levels ---
    dt2 = build_synthetic_olci(rows=1024, cols=1024)
    out2 = str(tmp_path / "olci_case2.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt2, output_path=out2, min_dimension=256)

    g2 = zarr.open_group(out2, mode="r")
    meas2 = g2["measurements"]
    assert isinstance(meas2, zarr.Group)
    subgroups2 = sorted(meas2.group_keys())
    assert len(subgroups2) == 2, (
        f"Expected exactly 2 overview levels for 1024x1024 at min_dimension=256, got {subgroups2}"
    )

    # Every overview level must have BOTH spatial dims >= min_dimension.
    for sg_name in subgroups2:
        sg = meas2[sg_name]
        assert isinstance(sg, zarr.Group)
        band = sg["oa01_radiance"]
        assert isinstance(band, zarr.Array)
        # shape is (rows, columns)
        overview_rows, overview_cols = band.shape[0], band.shape[1]
        assert overview_rows >= 256, f"measurements/{sg_name} rows={overview_rows} < 256"
        assert overview_cols >= 256, f"measurements/{sg_name} cols={overview_cols} < 256"

    # The deepest level (r4 for 1024-input) must be exactly 256x256.
    deepest = meas2[subgroups2[-1]]
    assert isinstance(deepest, zarr.Group)
    deepest_band = deepest["oa01_radiance"]
    assert isinstance(deepest_band, zarr.Array)
    assert deepest_band.shape[0] == 256, (
        f"Expected smallest overview rows=256, got {deepest_band.shape}"
    )
    assert deepest_band.shape[1] == 256, (
        f"Expected smallest overview cols=256, got {deepest_band.shape}"
    )


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
