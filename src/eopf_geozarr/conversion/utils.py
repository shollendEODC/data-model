"""Utility functions for GeoZarr conversion."""

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
import rasterio  # noqa: F401  # Import to enable .rio accessor
import structlog
import xarray as xr
import zarr
import zarr_cm
from zarr.codecs import CastValue
from zarr_cm import GeoProjAttrs, MultiConventionAttrs, MultiscalesAttrs, SpatialAttrs
from zarr_cm import geo_proj as geo_proj_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.types import (
    CF_SCALE_OFFSET_KEYS,
    XARRAY_ENCODING_KEYS,
    XarrayDataArrayEncoding,
)

if TYPE_CHECKING:
    from collections.abc import Hashable


log = structlog.get_logger()


# Dimension names that represent a "band-like" axis (polarization) to allow a per-"band" sharding if they extend beyond ram
# purposedly doeStn inlcude the 'band' option to not impleemnt on small enOugh arrays
BAND_LIKE_DIM_NAMES = frozenset({"polarization"})


def _band_like_dim_index(var_data: xr.DataArray) -> int | None:
    """Index of the band-like dimension, if var_data has one. Falls back to
    None (caller should then skip band-axis sharding, not guess)."""
    if len(var_data.dims) <= 2:
        log.info(
            "Not sharding along bandlike dim if its not dim>2", band_like_dim=BAND_LIKE_DIM_NAMES
        )
    else:
        for i, dim in enumerate(var_data.dims):
            if dim in BAND_LIKE_DIM_NAMES:
                return i
    return None


def _rechunk_ds(ds: xr.Dataset, spatial_chunk: int) -> xr.Dataset:
    chunks = {dim: (min(spatial_chunk, size)) for dim, size in ds.sizes.items()}
    return ds.chunk(chunks)


def rechunk_dataset_for_encoding(
    dataset: xr.Dataset, encoding: dict[str, XarrayDataArrayEncoding]
) -> xr.Dataset:
    """
    Rechunk dataset variables to align with sharding dimensions when sharding is enabled.

    When using Zarr v3 sharding, Dask chunks must align with shard dimensions to avoid
    checksum validation errors.
    """
    rechunked_vars: dict[Hashable, xr.DataArray] = {}

    for var_name, var_data in dataset.data_vars.items():
        if str(var_name) in encoding:
            var_encoding = encoding[str(var_name)]

            # If sharding is enabled, rechunk based on shard dimensions
            if "shards" in var_encoding and var_encoding["shards"] is not None:
                target_chunks = var_encoding["shards"]  # Use shard dimensions for rechunking
            elif "chunks" in var_encoding:
                target_chunks = var_encoding["chunks"]  # Fallback to chunk dimensions
            else:
                # No specific chunking needed, use original variable
                rechunked_vars[var_name] = var_data
                continue

            # Create chunk dict using the actual dimensions of the variable
            var_dims = var_data.dims
            chunk_dict = {}
            for i, dim in enumerate(var_dims):
                if i < len(target_chunks):
                    chunk_dict[dim] = target_chunks[i]

            # Rechunk the variable to match the target dimensions
            rechunked_vars[var_name] = var_data.chunk(chunk_dict)
        else:
            # No specific chunking needed, use original variable
            rechunked_vars[var_name] = var_data

    # Create new dataset with rechunked variables, preserving coordinates
    return xr.Dataset(rechunked_vars, coords=dataset.coords, attrs=dataset.attrs)


def get_chunking_for_encoding(
    var_data: xr.DataArray, shard_along_smallest_dimension: bool = False
) -> tuple[int, ...]:
    """
    requires a prior rechunking of the dataset by calling _rechunk_ds() to rechunk non-metadata arrays to spatial_chunk
    get a tuple of maximal chunksize for the dataarray
    -> (spatial_chukn, spatial_chukn) for spatial arrays
    -> (x, y, z, ..) for multidimensional metadata arrays (just to allow sharding later on)

    Args:
        var_data: DataArray to get the chunks from

    """
    if var_data.chunks:
        # get the maximal chunk shape for zarr encoding -> theoretically it wouldnt be necessary to take the max, as non-uniform chukning (1024, 806)
        # has irregular chunksizes trailing, but the syntax and goal of the code is much clearer this way
        max_chunksizes = [max(c) for c in var_data.chunks]

        if shard_along_smallest_dimension:
            band_dim = _band_like_dim_index(var_data)
            if band_dim is not None:
                max_chunksizes[band_dim] = 1

        # consider the occurance of 1dim arrays, provide the encoding chunk ndim times
        return (max_chunksizes[0],) if var_data.ndim == 1 else tuple(max_chunksizes)
    raise ValueError(
        f"Datavariable {var_data.name!r} is not chunked already, cannot derive Zarr encoding chunks -> will lead to unchunked array"
    )


def create_uniform_encoding(
    dataset: xr.Dataset,
    *,
    spatial_chunk: int,
    enable_sharding: bool = True,
    shard_along_smallest_dimension: bool = False,
    keep_scale_offset: bool = True,
    experimental_scale_offset_codec: bool = False,
    compression_level: int = 3,
) -> dict[str, XarrayDataArrayEncoding]:
    """
    Create encoding (compression, chunking, sharding) for a dataset.

    Chunking is taken from the input dataset's existing chunks when present
    (e.g. a group that's already been rechunked/aggregated, such as a
    pyramid level or a group written with `preferred_chunks`). Only when a
    variable has no chunks at all do we compute a chunk grid from
    `spatial_chunk`. Sharding always covers the *entire* array along every
    dimension, sized as the smallest multiple of that dimension's chunk size
    that is >= the array's shape — so a shard always contains a whole number
    of chunks and there is exactly one shard per array. This avoids partial
    edge chunks ending up in their own oddly-sized shard when e.g. shape=1830
    and chunk=1024 (shard becomes 2048, i.e. 2 chunks, not some 1830-based
    value that would clip/overlap the second chunk).
    """
    import math

    from zarr.codecs import BloscCodec

    encoding: dict[str, XarrayDataArrayEncoding] = {}
    compressor = BloscCodec(cname="zstd", clevel=compression_level, shuffle="shuffle", blocksize=0)

    for var_name, var_data in dataset.data_vars.items():
        var_encoding: XarrayDataArrayEncoding = {}

        encoding_chunks = get_chunking_for_encoding(var_data, shard_along_smallest_dimension)

        var_encoding["chunks"] = encoding_chunks
        var_encoding["compressors"] = (compressor,)

        # --- Shards: cover the whole array, one shard per array -----------
        if enable_sharding:
            # select next largest mutliple of chunksize to fit full array
            shards_ = [
                math.ceil(shape / chunk) * chunk
                for shape, chunk in zip(var_data.shape, encoding_chunks, strict=True)
            ]
            if shard_along_smallest_dimension:
                band_dim = _band_like_dim_index(var_data)
                if band_dim is None:
                    log.warning(
                        "shard_along_smallest_dimension=True but %s has no "
                        "recognized band-like dimension (%s); falling back to "
                        "whole-array sharding",
                        var_data.name,
                        list(var_data.dims),
                    )
                else:
                    shards_[band_dim] = encoding_chunks[band_dim]
                # sharding along the smallest dimesnion -> eg polarization (1, 1000, 2000)
                # mnp = var_data.shape.index(min(var_data.shape))
                # shards_ = [
                #     math.ceil(shape / chunk) * chunk
                #     for shape, chunk in zip(var_data.shape, encoding_chunks, strict=True)
                # ]
                # shards_[mnp] = 1
                # shards = tuple(shards_)

                # shards = tuple(min(shard_number * chunk, math.ceil(shape / chunk) * chunk) for shape, chunk in zip(var_data.shape, encoding_chunks))
                # shards = tuple(
                #     min(
                #         math.ceil(shape // (shard_number / 2) / spatial_chunk) * chunk,
                #         math.ceil(shape / chunk) * chunk,
                #     )
                #     for shape, chunk in zip(var_data.shape, encoding_chunks, strict=True)
                # )
            var_encoding["shards"] = tuple(shards_)
        else:
            var_encoding["shards"] = None

        # --- Forward-propagate remaining encoding keys ---------------------
        keep_keys = XARRAY_ENCODING_KEYS - {"compressors", "shards", "chunks"}

        # Whether to inject a CF _FillValue attribute for xarray issue #11345.
        # The injection itself happens after sanitize_array_attrs below, which
        # would otherwise strip it.
        inject_nan_fillvalue = False

        if experimental_scale_offset_codec and not keep_scale_offset:
            # THIS didnt work when previously tested

            # Push CF scale-offset into the zarr codec pipeline instead of
            # decoding to float. The data stays as packed integers on disk,
            # but zarr transparently decodes on read.
            scale_factor = var_data.encoding.get("scale_factor")
            add_offset = var_data.encoding.get("add_offset")
            packed_dtype = var_data.encoding.get("dtype")

            if scale_factor is not None and add_offset is not None and packed_dtype is not None:
                from eopf_geozarr.codecs.scale_offset import scale_offset_from_cf

                so_codec = scale_offset_from_cf(
                    scale_factor=float(scale_factor), add_offset=float(add_offset)
                )
                packed_np_dtype = np.dtype(packed_dtype)
                source_fill = var_data.encoding.get("_FillValue")
                if source_fill is not None:
                    nan_sentinel = int(source_fill)
                else:
                    nan_sentinel = int(np.iinfo(packed_np_dtype).min)
                cv_codec = CastValue(
                    data_type=packed_np_dtype.name,
                    rounding="nearest-even",
                    scalar_map={
                        "encode": [("NaN", nan_sentinel)],
                        "decode": [(nan_sentinel, "NaN")],
                    },
                )
                var_encoding["filters"] = (so_codec, cv_codec)

            keep_keys = keep_keys - CF_SCALE_OFFSET_KEYS - {"_FillValue", "filters"}
            var_encoding["fill_value"] = "NaN"
            inject_nan_fillvalue = True
        elif not keep_scale_offset:
            # When stripping scale/offset, also strip _FillValue since the original
            # _FillValue is in raw integer units and meaningless for decoded float data.
            keep_keys = keep_keys - CF_SCALE_OFFSET_KEYS - {"_FillValue"}
            var_encoding["fill_value"] = "NaN"
            inject_nan_fillvalue = True
        else:
            # Not stripping scale/offset: pick an explicit zarr-level fill_value
            # rather than letting xarray infer one differently across versions.
            keep_keys = keep_keys - {"fill_value"}
            fv = explicit_fill_value(var_data)
            if fv is not UNSET:
                var_encoding["fill_value"] = fv

        for key in keep_keys:
            if key in var_data.encoding:
                var_encoding[key] = var_data.encoding[key]

        if len(set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS) > 0:
            log.warning(
                "Unknown encoding keys in %s: %s",
                var_name,
                set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS,
            )

        # Sanitize source-only attributes (replace dict — ``.update`` cannot
        # remove keys, so stale ``_eopf_attrs`` / ``dtype`` / ``valid_*`` would
        # otherwise leak into the output).
        is_float = np.issubdtype(var_data.dtype, np.floating)
        var_data.attrs = sanitize_array_attrs(var_data.attrs, is_decoded_float=is_float)
        if inject_nan_fillvalue:
            var_data.attrs["_FillValue"] = np.nan

        encoding[str(var_name)] = var_encoding

    for coord_name, coord_data in dataset.coords.items():
        coord_data.attrs = sanitize_array_attrs(coord_data.attrs)
        encoding[str(coord_name)] = {"compressors": []}  # type: ignore[typeddict-item]

    return encoding


@runtime_checkable
class CRSLike(Protocol):
    """A coordinate reference system that can serialize to EPSG/WKT2.

    Both ``pyproj.CRS`` and ``rasterio.crs.CRS`` satisfy this; the conversion
    code accepts either, so we depend on the shared interface rather than a
    concrete class.
    """

    def to_epsg(self) -> int | None: ...

    def to_wkt(self) -> str: ...


def proj_attrs_for_crs(crs: CRSLike | None) -> GeoProjAttrs:
    """Build the ``proj`` convention data keys for a CRS.

    Prefers an EPSG code (``proj:code``) and falls back to WKT2
    (``proj:wkt2``). Returns an empty mapping when *crs* is ``None`` or exposes
    no EPSG code.
    """
    if crs is None:
        return GeoProjAttrs()
    epsg = crs.to_epsg()
    if epsg:
        return GeoProjAttrs({"proj:code": f"EPSG:{epsg}"})
    return GeoProjAttrs({"proj:wkt2": crs.to_wkt()})


def build_convention_attrs(
    *,
    spatial: SpatialAttrs,
    crs: CRSLike | None,
    multiscales: MultiscalesAttrs | None = None,
) -> MultiConventionAttrs:
    """Build validated multiscales + ``spatial`` + ``proj`` convention attributes.

    Delegates to :func:`zarr_cm.create_many`, which validates each convention's
    data and emits the matching convention-metadata objects into a combined
    ``zarr_conventions`` array. The CMOs are ordered multiscales (if present),
    then spatial, then proj. *spatial* holds the ``spatial:*`` keys; the proj
    keys are derived from *crs* via :func:`proj_attrs_for_crs`.

    The proj convention is only included when *crs* yields a usable CRS
    representation; otherwise only the other conventions are emitted (a proj
    convention with no CRS field is invalid).
    """
    conventions: dict[zarr_cm.ConventionName, MultiscalesAttrs | SpatialAttrs | GeoProjAttrs] = {}
    if multiscales is not None:
        conventions["multiscales"] = multiscales
    conventions["spatial"] = spatial
    proj = proj_attrs_for_crs(crs)
    if proj:
        conventions["geo-proj"] = proj
    # create_many validates each convention and emits its CMO. It returns a
    # generic JSON dict; narrow to the combined convention TypedDict.
    result = zarr_cm.create_many(conventions)
    return cast("MultiConventionAttrs", result)


# Sentinel: distinguish "no explicit fill_value" from a legitimate `None`.
UNSET: Any = object()


def explicit_fill_value(var: xr.DataArray) -> Any:
    """Pick a zarr-level `fill_value` for `var` based on its source `_FillValue`.

    Different xarray versions infer different on-disk fill values when the
    encoding dict doesn't pin it: older xarray defaults floats to 0.0; newer
    xarray honours the source `_FillValue`. Setting `fill_value` explicitly
    via this helper removes that degree of freedom so the on-disk metadata is
    stable across xarray versions.

    Returns
    -------
    object
        The value to assign to `encoding["fill_value"]`. The sentinel `UNSET`
        is returned when the source has no `_FillValue` (caller should leave
        the encoding entry alone). For non-finite floats, returns the
        JSON-canonical string form (`"NaN"` / `"Infinity"` / `"-Infinity"`)
        that zarr-python serialises.
    """
    source_fill = var.encoding.get("_FillValue")
    if source_fill is None:
        return UNSET
    fill_arr = np.asarray(source_fill)
    if np.issubdtype(fill_arr.dtype, np.floating) and not np.isfinite(fill_arr):
        if np.isnan(fill_arr):
            return "NaN"
        return "Infinity" if fill_arr > 0 else "-Infinity"
    return source_fill


def sanitize_array_attrs(
    attrs: dict[str, Any],
    *,
    is_decoded_float: bool = False,
) -> dict[str, Any]:
    """Return a copy of *attrs* with source-only and misleading keys removed.

    - ``_eopf_attrs`` and ``_FillValue`` are always removed. ``_FillValue``
      belongs in the variable's *encoding* (where the zarr-level fill value is
      carried), not in its attributes; callers that need a CF ``_FillValue``
      attribute (e.g. the NaN workaround for xarray issue #11345) must re-add
      it after sanitizing.

      .. warning:: The Sentinel-3 OLCI converter has its own deliberately
         divergent sanitizer
         (``s3_olci_optimization.olci_converter._sanitize_olci_array_attrs_keep_fill``)
         that **preserves** ``_FillValue`` for raw (non-mask-scaled) input.
         Edits to the strip-list here do not apply there, and vice versa.
    - For decoded float measurement arrays (*is_decoded_float=True*), also
      removes raw-encoding leftovers ``dtype``, ``fill_value``,
      ``valid_min``, ``valid_max`` and rewrites
      ``units: "digital_counts"`` → ``"1"``.

    - Geo-proj *convention* keys (``proj:code``, ``proj:wkt2``,
      ``proj:projjson``) are always removed: per the minispec they belong on
      (or are inherited from) the enclosing group, and source products carry
      them on arrays without the required ``zarr_conventions`` declaration.
      Legacy external keys such as ``proj:epsg`` are left alone.

    CF keys ``scale_factor`` and ``add_offset`` are always preserved.
    """
    dropped = {"_eopf_attrs", "_FillValue", *geo_proj_cm.CONVENTION_KEYS}
    out = {k: v for k, v in attrs.items() if k not in dropped}
    if is_decoded_float:
        for key in ("dtype", "fill_value", "valid_min", "valid_max"):
            out.pop(key, None)
        if out.get("units") == "digital_counts":
            out["units"] = "1"
    return out


def downsample_2d_array(
    source_data: np.ndarray,
    target_height: int,
    target_width: int,
    nodata_value: float | None = None,
) -> np.ndarray:
    """
    Downsample a 2D array using block averaging with proper nodata handling.

    Parameters
    ----------
    source_data : numpy.ndarray
        Source 2D array
    target_height : int
        Target height
    target_width : int
        Target width
    nodata_value : float, optional
        Value representing nodata/fill areas. If provided, these areas will be
        excluded from averaging and preserved in the output.

    Returns
    -------
    numpy.ndarray
        Downsampled 2D array with nodata values preserved
    """
    source_height, source_width = source_data.shape

    # Calculate block sizes
    block_size_y = source_height // target_height
    block_size_x = source_width // target_width

    if block_size_y > 1 and block_size_x > 1:
        # Block averaging with nodata handling
        reshaped = source_data[: target_height * block_size_y, : target_width * block_size_x]
        reshaped = reshaped.reshape(target_height, block_size_y, target_width, block_size_x)

        if nodata_value is not None and not np.isnan(nodata_value):
            # Create mask for valid data (not nodata)
            valid_mask = reshaped != nodata_value

            # Calculate mean only for valid data
            with np.errstate(invalid="ignore", divide="ignore"):
                # Sum valid values and count valid pixels
                valid_sum = np.where(valid_mask, reshaped, 0).sum(axis=(1, 3))
                valid_count = valid_mask.sum(axis=(1, 3))

                # Calculate mean, preserving nodata where no valid data exists
                downsampled = np.where(valid_count > 0, valid_sum / valid_count, nodata_value)
        elif nodata_value is not None and np.isnan(nodata_value):
            # Handle NaN nodata values
            with np.errstate(invalid="ignore"):
                downsampled = np.nanmean(reshaped, axis=(1, 3))
        else:
            # No nodata handling needed
            downsampled = reshaped.mean(axis=(1, 3))
    else:
        # Simple subsampling
        y_indices = np.linspace(0, source_height - 1, target_height, dtype=int)
        x_indices = np.linspace(0, source_width - 1, target_width, dtype=int)
        downsampled = source_data[np.ix_(y_indices, x_indices)]

    return downsampled


def is_grid_mapping_variable(ds: xr.Dataset, var_name: str) -> bool:
    """
    Check if a variable is a grid_mapping variable by looking for references to it.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check
    var_name : str
        Variable name to check

    Returns
    -------
    bool
        True if this variable is referenced as a grid_mapping
    """
    for data_var in ds.data_vars:
        if (
            data_var != var_name
            and "grid_mapping" in ds[data_var].attrs
            and ds[data_var].attrs["grid_mapping"] == var_name
        ):
            return True
    return False


def calculate_aligned_chunk_size(dimension_size: int, target_chunk_size: int) -> int:
    """
    Calculate a chunk size that divides evenly into the dimension size.

    This ensures that Zarr chunks align properly with the data dimensions,
    preventing chunk overlap issues when writing with Dask.

    Parameters
    ----------
    dimension_size : int
        Size of the dimension to chunk
    target_chunk_size : int
        Desired chunk size

    Returns
    -------
    int
        Aligned chunk size that divides evenly into dimension_size
    """
    if target_chunk_size >= dimension_size:
        return dimension_size

    # Find the largest divisor of dimension_size that is <= target_chunk_size
    for chunk_size in range(target_chunk_size, int(target_chunk_size * 0.51), -1):
        if dimension_size % chunk_size == 0:
            return chunk_size

    # If no divisor is found, return the closest value to target_chunk_size
    return min(target_chunk_size, dimension_size)


def validate_existing_band_data(
    existing_group: xr.Dataset, var_name: str, reference_ds: xr.Dataset
) -> bool:
    """
    Validate that a specific band exists and is complete in the dataset.

    Parameters
    ----------
    existing_group : xarray.Dataset
        Existing dataset to validate
    var_name : str
        Name of the variable to validate
    reference_ds : xarray.Dataset
        Reference dataset structure for comparison

    Returns
    -------
    bool
        True if the variable exists and is valid, False otherwise
    """
    try:
        # Check if the variable exists
        if var_name not in existing_group.data_vars and var_name not in existing_group.coords:
            return False

        # Check shape matches
        if var_name in reference_ds.data_vars:
            expected_shape = reference_ds[var_name].shape
            existing_shape = existing_group[var_name].shape

            if expected_shape != existing_shape:
                return False

        # Check required attributes for data variables
        if var_name in reference_ds.data_vars and not is_grid_mapping_variable(
            reference_ds, var_name
        ):
            required_attrs = ["_ARRAY_DIMENSIONS", "standard_name"]
            for attr in required_attrs:
                if attr not in existing_group[var_name].attrs:
                    return False

        # Check rio CRS
        if existing_group.rio.crs != reference_ds.rio.crs:
            return False

        # Basic data integrity check for data variables
        if var_name in existing_group.data_vars and not is_grid_mapping_variable(
            existing_group, var_name
        ):
            try:
                # Just check if we can access the array metadata without reading data
                array_info = existing_group[var_name]
                if array_info.size == 0:
                    return False
                # read a piece of data to ensure it's valid
                test = array_info.isel(dict.fromkeys(array_info.dims, 0)).values.mean()
                if np.isnan(test):
                    return False
            except Exception as e:
                log.info("Error validating variable", var_name=var_name, error=str(e))
                return False

    except Exception:
        return False
    else:
        return True


def compute_overview_gcps(
    ds_gcp: xr.Dataset, scale_factor: float, width: int, height: int
) -> xr.Dataset:
    """Compute new GCPs for a given overview from the original GCPs.

    Parameters
    ----------
    ds_gcp : xr.Dataset
        the original GCPs
    scale_factor : float
        Overview's scale factor
    width : int
        Overview's width
    height : int
        Overview's height

    Returns
    -------
    ds_gcp_overview : xr.Dataset
        A new dataset where GCPs line and pixel coordinates are updated
        for the overview, and where duplicate line/pixel GCPs are
        merged together by averaging their latitude, longitude and height.

    """
    return (
        # compute the new decimated line/pixel coordinates
        # TODO: trim line values with height and pixel values with width?
        ds_gcp.assign_coords(
            line=np.round(ds_gcp.line / scale_factor).astype(np.int64),
            pixel=np.round(ds_gcp.pixel / scale_factor).astype(np.int64),
        )
        # find duplicate line/pixel GCPs
        # and compute average for latitude, longitude and height
        .pipe(lambda ds: ds.groupby(["line", "pixel"]))
        .mean()
        # re-assign original dimensions
        .rename_dims(line="azimuth_time", pixel="ground_range")
    )


def _as_bbox(value: object) -> tuple[float, float, float, float] | None:
    """Return *value* as a 4-tuple of floats, or ``None`` if it is not one.

    ``spatial:bbox`` is read from stored metadata, so its type is not known
    statically; this verifies the shape at runtime rather than asserting it.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(v, (int, float)) for v in value):
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _crs_from_attrs(attrs: dict[str, Any]) -> Any | None:
    """Resolve a pyproj CRS from a group's ``proj:*`` attributes, else ``None``.

    Tries ``proj:code``, then ``proj:wkt2``, then ``proj:projjson``. Malformed
    values are logged and treated as unresolvable rather than raised, since the
    attributes come from stored metadata.
    """
    from pyproj import CRS as PyprojCRS

    code = attrs.get("proj:code")
    if isinstance(code, str):
        try:
            return PyprojCRS.from_user_input(code)
        except Exception as e:  # malformed stored metadata
            log.warning("Unresolvable proj:code; skipping group bbox", code=code, error=str(e))
            return None
    wkt2 = attrs.get("proj:wkt2")
    if isinstance(wkt2, str):
        try:
            return PyprojCRS.from_wkt(wkt2)
        except Exception as e:
            log.warning("Unresolvable proj:wkt2; skipping group bbox", error=str(e))
            return None
    projjson = attrs.get("proj:projjson")
    if isinstance(projjson, dict):
        try:
            return PyprojCRS.from_json_dict(projjson)
        except Exception as e:
            log.warning("Unresolvable proj:projjson; skipping group bbox", error=str(e))
            return None
    return None


def write_store_root_geo_metadata(
    output_path: str, storage_options: dict[str, Any] | None = None
) -> None:
    """Write the minispec store-root metadata on the root group.

    Walks the zarr store, collects every child-group `spatial:bbox` along with
    its CRS (resolved from ``proj:code`` / ``proj:wkt2`` / ``proj:projjson``),
    reprojects each to EPSG:4326 with edge densification and writes the union
    plus the CRS code and the matching ``zarr_conventions`` declaration on the
    root group. Groups whose CRS cannot be resolved are skipped with a warning
    rather than assumed to be in degrees. The CRS is always declared explicitly
    per the Store Root section of the minispec — there is no implicit default.

    When *storage_options* is ``None``, the store's options are derived from
    *output_path* via :func:`eopf_geozarr.conversion.fs_utils.get_storage_options`
    so remote (e.g. S3) stores honour the configured endpoint and credentials.
    """
    from pyproj import Transformer

    from eopf_geozarr.conversion import fs_utils

    if storage_options is None:
        storage_options = cast("dict[str, Any] | None", fs_utils.get_storage_options(output_path))

    root = zarr.open_group(output_path, mode="r+", storage_options=storage_options)

    bboxes_4326: list[tuple[float, float, float, float]] = []

    def _walk(group: zarr.Group) -> None:
        attrs = dict(group.attrs)
        corners = _as_bbox(attrs.get("spatial:bbox"))
        if corners is not None:
            crs = _crs_from_attrs(attrs)
            if crs is None:
                if any(k in attrs for k in ("proj:code", "proj:wkt2", "proj:projjson")):
                    # warning already logged by _crs_from_attrs
                    pass
                else:
                    log.warning(
                        "Group has spatial:bbox but no proj:* CRS; skipping it "
                        "for the store-root footprint",
                        group=group.path,
                    )
            elif crs.to_epsg() == 4326:
                bboxes_4326.append(corners)
            else:
                try:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    # transform_bounds densifies the edges, which corner-wise
                    # transformation misses (projected edges curve in lon/lat).
                    bboxes_4326.append(transformer.transform_bounds(*corners, densify_pts=21))
                except Exception as e:  # never abort the write for one group
                    log.warning(
                        "Failed to reproject group bbox; skipping it",
                        group=group.path,
                        error=str(e),
                    )
        for child in group.groups():
            _walk(child[1])

    for _, child_group in root.groups():
        _walk(child_group)

    if not bboxes_4326:
        log.warning("No usable child-group spatial:bbox found; skipping store-root metadata")
        return

    if any(b[0] > b[2] for b in bboxes_4326):
        # At least one footprint crosses the antimeridian; a single
        # [xmin, ymin, xmax, ymax] box cannot represent the union faithfully,
        # so fall back to the full longitude range.
        log.warning(
            "A child bbox crosses the antimeridian; store-root bbox uses the full longitude range"
        )
        xmin, xmax = -180.0, 180.0
    else:
        xmin = min(b[0] for b in bboxes_4326)
        xmax = max(b[2] for b in bboxes_4326)
    ymin = min(b[1] for b in bboxes_4326)
    ymax = max(b[3] for b in bboxes_4326)
    root_attrs: dict[str, Any] = {
        "zarr_conventions": [dict(spatial_cm.CMO), dict(geo_proj_cm.CMO)],
        "spatial:bbox": [xmin, ymin, xmax, ymax],
        "proj:code": "EPSG:4326",
    }
    root.attrs.update(root_attrs)
    log.info("Wrote store-root spatial metadata", bbox=[xmin, ymin, xmax, ymax])


def write_store_root_stac_metadata(
    output_path: str,
    root_attrs: dict[str, dict[str, Any]],
    storage_options: dict[str, Any] | None = None,
) -> None:
    """ """
    from eopf_geozarr.conversion import fs_utils

    if storage_options is None:
        storage_options = cast("dict[str, Any] | None", fs_utils.get_storage_options(output_path))

    root = zarr.open_group(output_path, mode="r+", storage_options=storage_options)

    root.attrs.update(root_attrs)
    log.info(
        "Updated root metadata attributes for STAC ingestion", root_attrs=list(root_attrs.keys())
    )
