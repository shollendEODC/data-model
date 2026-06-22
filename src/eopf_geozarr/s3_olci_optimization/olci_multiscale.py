"""Multiscale (overview) generation for OLCI swath data.

OLCI L1 EFR is a curvilinear swath geolocated by per-pixel 2-D lat/lon arrays,
so overviews are produced by /2 decimation of the (rows, columns) grid: every
2-D variable and its 2-D coordinate arrays are subsampled together, keeping
geolocation exact. (Averaging is intentionally avoided — an averaged lat/lon
would not correspond to a real pixel.)
"""

from __future__ import annotations

import xarray as xr  # noqa: TC002

SWATH_DIMS = ("rows", "columns")


def decimate_swath(ds: xr.Dataset, factor: int = 2) -> xr.Dataset:
    """Return *ds* with every (rows, columns) array subsampled by *factor*.

    Both data variables and coordinate variables that span exactly the swath
    dims are decimated `[::factor, ::factor]`; everything else is passed
    through unchanged. Attributes and encoding are preserved by xarray's isel.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return ds
    indexers = {dim: slice(None, None, factor) for dim in SWATH_DIMS if dim in ds.sizes}
    if not indexers:
        return ds
    return ds.isel(indexers)
