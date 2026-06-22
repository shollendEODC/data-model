"""Integration tests for convert_olci_optimized."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec

if TYPE_CHECKING:
    import pathlib

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
    # Build a measurements/orphans sub-dataset to verify child subgroup copy.
    orphans_ds = xr.Dataset(
        {
            "removed_count": xr.DataArray(
                rng.integers(0, 10, (rows,)).astype("int32"), dims=("rows",)
            )
        }
    )
    dt = xr.DataTree.from_dict(
        {
            "/measurements": meas_ds,
            "/measurements/orphans": orphans_ds,
            "/conditions": cond_ds,
            "/quality": quality_ds,
        }
    )

    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out, min_dimension=64)

    g = zarr.open_group(out, mode="r")
    assert "conditions" in g
    assert "quality" in g
    # measurements/orphans subgroup must have been copied through.
    assert "orphans" in g["measurements"]


def test_cli_convert_s3_olci_optimized(tmp_path: pathlib.Path) -> None:
    """convert-s3-olci-optimized subcommand must write a GeoZarr measurements group."""
    import zarr

    # materialise a synthetic OLCI product to a zarr v2 store on disk
    dt = build_synthetic_olci(rows=300, cols=300)
    src = tmp_path / "olci_src.zarr"
    dt.to_zarr(src, mode="w", consolidated=False)
    out = tmp_path / "olci_out.zarr"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eopf_geozarr",
            "convert-s3-olci-optimized",
            str(src),
            str(out),
            "--spatial-chunk",
            "256",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    g = zarr.open_group(str(out), mode="r")
    assert "measurements" in g


def _assert_radiance_dtype_and_attrs(
    group: zarr.Group, band_name: str, *, level_label: str
) -> None:
    """Assert that *band_name* in *group* is uint16 with scale_factor and no stale attrs.

    This is a regression guard: the converter must preserve raw integer storage
    and CF scale/offset, and must strip source-only attrs (_eopf_attrs, dtype,
    valid_min, valid_max).
    """
    band = group[band_name]
    assert isinstance(band, zarr.Array), f"{level_label}/{band_name} is not a zarr.Array"
    assert band.dtype == np.dtype("uint16"), (
        f"{level_label}/{band_name}: expected uint16, got {band.dtype}"
    )
    attrs = dict(band.attrs)
    assert "scale_factor" in attrs, (
        f"{level_label}/{band_name}: scale_factor missing from attrs (got {list(attrs)})"
    )
    for stale_key in ("_eopf_attrs", "dtype", "valid_min", "valid_max"):
        assert stale_key not in attrs, (
            f"{level_label}/{band_name}: stale attr '{stale_key}' present in output attrs"
        )


def test_olci_conversion_matches_snapshot(
    s3_olci_group_example: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Snapshot test: converted OLCI structure must match committed golden file.

    The fixture is a Zarr v2 store representing a real OLCI L1 EFR product.
    We open the whole DataTree so that conditions, quality, and
    measurements/orphans subgroups are included in the conversion.
    The fixture uses the real product's dimension names: tie-point grids reuse
    'columns' (at size 4) while measurement grids also use 'columns' (at size 16);
    orphan arrays use removed_pixels=4.

    ``min_dimension=8`` is used so that the 16x16 measurements grid generates
    one overview level (r2 at 8x8).

    To (re)generate the snapshot, uncomment the regeneration block below,
    run the test once, then re-comment before committing.
    """
    dt_in = xr.open_datatree(
        str(s3_olci_group_example),
        engine="zarr",
        consolidated=False,
        chunks={},
        mask_and_scale=False,
    )
    out = str(tmp_path / "out.zarr")
    convert_olci_optimized(dt_in, output_path=out, min_dimension=8)

    observed_group = zarr.open_group(out, use_consolidated=False)
    observed_structure_json = GroupSpec.from_zarr(observed_group).model_dump()

    expected_path = Path("tests/_test_data/optimized_olci_examples") / (
        s3_olci_group_example.stem + ".json"
    )

    # Uncomment this block to (re)generate the snapshot from the observed structure.
    # expected_path.parent.mkdir(parents=True, exist_ok=True)
    # expected_path.write_text(json.dumps(observed_structure_json, indent=2, sort_keys=True))

    observed_structure = GroupSpec(**tuplify_json(observed_structure_json))
    observed_structure_flat = observed_structure.to_flat()
    expected_structure_json = tuplify_json(json.loads(expected_path.read_text()))
    expected_structure = GroupSpec(**expected_structure_json)
    expected_structure_flat = expected_structure.to_flat()

    o_keys = set(observed_structure_flat.keys())
    e_keys = set(expected_structure_flat.keys())
    assert o_keys == e_keys
    assert [k for k in o_keys if observed_structure_flat[k] != expected_structure_flat[k]] == []

    # Dtype/attrs regression guard: radiance must be stored as uint16 with
    # scale_factor preserved and stale source attrs absent.
    meas_g = observed_group["measurements"]
    assert isinstance(meas_g, zarr.Group)
    _assert_radiance_dtype_and_attrs(meas_g, "oa01_radiance", level_label="native")
    if "r2" in meas_g:
        r2_g = meas_g["r2"]
        assert isinstance(r2_g, zarr.Group)
        _assert_radiance_dtype_and_attrs(r2_g, "oa01_radiance", level_label="r2")
