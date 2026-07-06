"""Multiscale (overview) generation for OLCI swath data.

OLCI L1 EFR is a curvilinear swath geolocated by per-pixel 2-D lat/lon arrays.
Two reduction strategies are provided:

* :func:`decimate_swath` — pure stride-based decimation; every (rows, columns)
  variable is subsampled ``[::factor, ::factor]``.  Geolocation (lat/lon) is
  kept exact; intended for coordinate arrays and cases where preserving pixel
  identity matters.

* :func:`reduce_swath` — radiance bands are fill-aware block-averaged
  (mean of ``factor x factor`` blocks) while coordinate arrays are decimated
  with :func:`decimate_swath`; non-swath variables pass through unchanged.
  Intended for producing GeoZarr multiscale overview groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_band_mapping import OLCI_BANDS

if TYPE_CHECKING:
    from zarr_cm import SpatialAttrs

SWATH_DIMS = ("rows", "columns")


def decimate_swath(ds: xr.Dataset, factor: int = 2) -> xr.Dataset:
    """Return *ds* with every (rows, columns) array subsampled by *factor*.

    Both data variables and coordinate variables that span exactly the swath
    dims are decimated ``[::factor, ::factor]``; everything else is passed
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


def reduce_swath(ds: xr.Dataset, factor: int = 2) -> xr.Dataset:
    """Return *ds* with radiance bands block-averaged and 2-D coordinates decimated.

    Overviews are an unweighted index-block mean that ASSUMES locally-uniform
    pixel spacing; intended for visualization, not quantitative analysis at
    reduced resolution.  Coordinates are decimated (real sub-pixels), radiance
    is fill-aware block-averaged.

    Radiance variables (those named in :data:`OLCI_BANDS`) are averaged over
    ``factor x factor`` pixel blocks with fill-value awareness: fill pixels
    are masked before averaging so a single fill pixel does not contaminate
    the entire block.  Blocks where ALL pixels are fill are set back to the
    fill value in the output.

    2-D coordinate variables spanning exactly ``(rows, columns)`` and **not**
    in :data:`OLCI_BANDS` (e.g. latitude, longitude, altitude) are decimated
    ``[::factor, ::factor]`` to preserve real on-ground positions.

    Variables that do not span exactly ``(rows, columns)`` are passed through
    unchanged.

    Parameters
    ----------
    ds:
        Input dataset.  May contain any mix of radiance bands, 2-D coordinate
        arrays, and other variables.
    factor:
        Spatial reduction factor.  Must be >= 1.  Factor 1 returns *ds*
        unchanged.

    Returns
    -------
    xr.Dataset
        Reduced dataset with the same variables but smaller (rows, columns)
        dimensions.

    Raises
    ------
    ValueError
        If *factor* < 1.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return ds

    olci_band_set: frozenset[str] = frozenset(OLCI_BANDS)
    result_vars: dict[str, xr.DataArray] = {}
    result_coords: dict[str, xr.DataArray] = {}

    coord_names: frozenset[str] = frozenset(str(k) for k in ds.coords)

    # Pre-compute the trimmed length for each swath dimension so that both the
    # coarsen path (radiance) and the isel stride path (coordinates) produce
    # exactly floor(N / factor) output elements.  coarsen(boundary="trim")
    # already truncates to a multiple of factor; we match it by stopping the
    # stride at the same trimmed limit:  slice(0, n_trim, factor).
    #
    # Without this, an odd-length dimension N yields:
    #   coarsen → floor(N / factor)   e.g. 4865 → 2432
    #   isel[::factor] → ceil(N / factor)  e.g. 4865 → 2433
    # producing a store where coordinate arrays are longer than the data they
    # describe, which makes xr.open_dataset raise a conflicting-sizes error.
    dim_trim: dict[str, int] = {
        dim: (ds.sizes[dim] // factor) * factor for dim in SWATH_DIMS if dim in ds.sizes
    }

    all_names: list[str] = [str(k) for k in ds.data_vars] + [str(k) for k in ds.coords]
    for name in all_names:
        var: xr.DataArray = ds[name] if name in ds.data_vars else ds.coords[name]
        is_swath_2d: bool = tuple(str(d) for d in var.dims) == SWATH_DIMS

        if name in olci_band_set and is_swath_2d:
            # Fill-aware block averaging for radiance bands.
            fill_value: int | float | None = var.attrs.get("_FillValue")
            if fill_value is None:
                fill_value = var.encoding.get("_FillValue")
            orig_dtype = var.dtype

            float_var = var.astype("float64")
            if fill_value is not None:
                float_var = float_var.where(float_var != float(fill_value))

            # coarsen().mean() is available at runtime; pyright stubs don't expose .mean()
            # on DataArrayCoarsen, so we suppress the type-check on the reduction call.
            coarsened = float_var.coarsen({"rows": factor, "columns": factor}, boundary="trim")
            averaged: xr.DataArray = coarsened.mean()  # type: ignore[attr-defined,assignment]

            if fill_value is not None:
                valid_mask = ~averaged.isnull().values
                fill_da = xr.where(averaged.isnull(), float(fill_value), averaged)
                result_arr = fill_da.values
            else:
                valid_mask = None
                result_arr = averaged.values
            # Round only when packing back into an integer dtype; float
            # radiance (e.g. a CF-decoded source) must not be quantized.
            if np.issubdtype(orig_dtype, np.integer):
                unrounded = result_arr
                result_arr = np.round(result_arr)
                if fill_value is not None and valid_mask is not None:
                    # A valid block whose mean rounds to the fill sentinel
                    # would be recoded as fill in the overview; nudge it one
                    # step back into the valid domain, on the side of the
                    # sentinel the unrounded mean came from (fill may sit at
                    # either end of the dtype range).
                    collision = valid_mask & (result_arr == float(fill_value))
                    if collision.any():
                        nudged = np.where(
                            unrounded <= float(fill_value),
                            float(fill_value) - 1,
                            float(fill_value) + 1,
                        )
                        result_arr = np.where(collision, nudged, result_arr)
            result_val = result_arr.astype(orig_dtype)

            out_var = xr.DataArray(result_val, dims=averaged.dims, attrs=var.attrs)

            if name in ds.data_vars:
                result_vars[name] = out_var
            else:
                result_coords[name] = out_var

        elif is_swath_2d:
            # Coordinate or non-radiance 2-D swath variable: decimate.
            # Use slice(0, n_trim, factor) rather than slice(None, None, factor)
            # so that an odd-length dimension N yields floor(N / factor) elements,
            # matching the output length of coarsen(boundary="trim").mean().
            indexers: dict[str, slice] = {
                dim: slice(0, dim_trim[dim], factor) for dim in SWATH_DIMS if dim in dim_trim
            }
            decimated = var.isel(indexers)
            if name in coord_names:
                result_coords[name] = decimated
            else:
                result_vars[name] = decimated

        elif any(dim in (str(d) for d in var.dims) for dim in SWATH_DIMS):
            # 1-D (or higher) variable sharing a swath dim but not 2-D swath:
            # decimate along whichever swath dims it carries.
            var_dims = {str(d) for d in var.dims}
            idx: dict[str, slice] = {
                dim: slice(0, dim_trim[dim], factor)
                for dim in SWATH_DIMS
                if dim in var_dims and dim in dim_trim
            }
            decimated = var.isel(idx)
            if name in coord_names:
                result_coords[name] = decimated
            else:
                result_vars[name] = decimated

        else:
            # Non-swath variable: pass through unchanged.
            if name in coord_names:
                result_coords[name] = var
            else:
                result_vars[name] = var

    return xr.Dataset(result_vars, coords=result_coords)


def swath_spatial_attrs(
    dims: tuple[str, str] = SWATH_DIMS,
) -> SpatialAttrs:
    """Spatial-convention data for curvilinear swath geometry.

    OLCI has no affine transform; geolocation is carried by 2-D lat/lon
    coordinate arrays, so we declare the spatial dimensions and pixel
    registration but no ``spatial:transform``/``spatial:bbox``.
    """
    return {
        "spatial:dimensions": [dims[0], dims[1]],
        "spatial:registration": "pixel",
    }
