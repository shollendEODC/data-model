import numpy as np
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_multiscale import decimate_swath


def _swath(rows: int = 8, cols: int = 6) -> xr.Dataset:
    rad = xr.DataArray(
        np.arange(rows * cols, dtype="uint16").reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"scale_factor": 0.5, "units": "mW.m-2.sr-1.nm-1"},
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
