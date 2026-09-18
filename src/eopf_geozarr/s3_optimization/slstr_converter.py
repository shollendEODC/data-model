"""Top-level Sentinel-3 SLSTR L1 -> GeoZarr conversion."""

from __future__ import annotations

import time

import numpy as np
import rioxarray  # noqa: F401
import structlog
import xarray as xr
import zarr
from pyproj import CRS

from eopf_geozarr.conversion import utils

# from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
# from eopf_geozarr.s3_optimization.olci_band_mapping import OLCI_BANDS
# from eopf_geozarr.s3_optimization.olci_multiscale import (
#     SWATH_DIMS,
#     grid_spatial_attrs,
#     reduce_swath,
#     swath_spatial_attrs,
# )

# from eopf_geozarr.s3_optimization.olci_reproject import GRID_DIMS, reproject_olci


log = structlog.get_logger()


def _sanitize_olci_array_attrs_keep_fill(attrs: dict[str, object]) -> dict[str, object]:
    """Return a copy of *attrs* with stale source-only keys removed.

    Strips ``_eopf_attrs``, ``dtype``, ``valid_min``, and ``valid_max`` (source
    provenance and raw-integer-domain metadata that is misleading in GeoZarr
    output).  Unlike the shared :func:`~eopf_geozarr.conversion.utils.sanitize_array_attrs`,
    this helper intentionally **preserves** ``_FillValue`` because OLCI input is
    opened with ``mask_and_scale=False`` (raw uint16) and downstream code (e.g.
    ``reduce_swath``) needs ``_FillValue`` in ``.attrs`` to identify fill pixels
    without CF decoding.

    CF keys ``scale_factor``, ``add_offset``, ``units``, ``standard_name``,
    ``coordinates``, and ``long_name`` are always preserved.
    """
    _strip = frozenset(("_eopf_attrs", "dtype", "valid_min", "valid_max"))
    return {k: v for k, v in attrs.items() if k not in _strip}


def sanitize_data_vars(ds: xr.Dataset) -> xr.Dataset:
    """Return *ds* with stale source attrs stripped from all data variables.

    Applies :func:`_sanitize_olci_array_attrs_keep_fill` to every data variable in *ds*.
    Coordinate variable attrs are left intact.

    This removes ``_eopf_attrs``, ``dtype``, ``valid_min``, and ``valid_max``
    (source-only / misleading) while preserving CF attrs
    (``scale_factor``, ``add_offset``, ``_FillValue``, ``units``,
    ``standard_name``, ``coordinates``).

    Note: ``xr.DataArray.assign_attrs`` *merges* (update semantics), so we
    copy the DataArray and replace ``.attrs`` in-place to ensure stale keys
    are actually removed rather than retained from the old dict.
    """
    new_vars: dict[str, xr.DataArray] = {}
    for name in ds.data_vars:
        var = ds[name]
        new_var = var.copy(data=var.data)
        new_var.attrs = _sanitize_olci_array_attrs_keep_fill(dict(var.attrs))
        new_vars[str(name)] = new_var
    return ds.assign(new_vars)


def convert_slstr_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool = False,
    spatial_chunk: int = 1024,
    compression_level: int = 3,
    min_dimension: int = 256,
    keep_scale_offset: bool = False,
    output_grid: str = "native",
    chunk_and_shard_coords: bool = False,
) -> xr.DataTree:
    """Convert an EOPF OLCI L1 EFR DataTree to a GeoZarr multiscale store.

    Writes the measurements pyramid to ``measurements/r0`` (+ ``r2``, ``r4``,
    … siblings). By default the instrument grid is preserved; pass
    ``output_grid=<CRS>`` to warp once onto a regular grid with per-level CRS
    metadata.

    Parameters
    ----------
    dt_input:
        Input OLCI L1 EFR DataTree (must contain a ``/measurements`` node).
    output_path:
        Filesystem path for the output Zarr v3 store.
    enable_sharding:
        Enable Zarr v3 sharding on measurement arrays.
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    spatial_chunk:
        Target spatial chunk size (pixels per side).
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    compression_level:
        Blosc/zstd compression level.
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    min_dimension:
        Stop generating overview levels once either spatial dimension would
        drop below this value after /2 decimation.
    keep_scale_offset:
        When ``True``, preserve CF ``scale_factor``/``add_offset`` in the
        output encoding rather than decoding to float32.
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    output_grid:
        ``"native"`` (default) preserves the instrument swath geometry:
        no warp, 2-D lat/lon geolocation, per-row ``time_stamp`` kept,
        and no CRS metadata.  Any other value is parsed as a CRS
        (``"EPSG:4326"``, WKT, …) and the swath is warped once onto a
        regular grid in that CRS before the pyramid builds.

    Returns
    -------
    xr.DataTree
        The opened output DataTree (lazy; backed by the written Zarr store).
        Opened with ``mask_and_scale=False``, mirroring the raw store and the
        converter's input: radiance is packed ``uint16`` with its CF
        ``scale_factor``/``_FillValue`` attrs intact, not decoded floats.
        Native-resolution arrays live at ``measurements/r0`` with overview
        levels (``r2``, ``r4``, …) as sibling groups, all on the instrument
        grid (default) or a regular ``output_grid`` grid with 1-D ``y``/``x``
        coordinates and a declared CRS; ``measurements`` itself holds only
        the multiscales/spatial convention metadata, so the whole store
        opens cleanly with ``xr.open_datatree``.

    Notes
    -----
    Parameters ``enable_sharding``, ``spatial_chunk``, ``compression_level``,
    and ``keep_scale_offset`` are accepted but not yet applied to the on-disk
    encoding.  Wiring them through the existing ``conversion`` helpers
    (``create_measurements_encoding``, sharding codec, etc.) is left for a
    follow-up task so as not to block the integration test.  A warning is
    logged when a non-default value is passed for any of them, so callers
    aren't silently handed default-encoded output.
    """
    # Fail fast before any store mutation: _overview_levels floor-halves the
    # dimensions, and min(r, c) // 2 >= min_dimension never becomes false for
    # min_dimension <= 0 once the sizes decay to zero (infinite loop).
    if min_dimension < 1:
        raise ValueError(f"min_dimension must be >= 1; got {min_dimension}")

    # arr = dt_input['/measurements/boblique']
    # lat = arr['latitude'].values
    # x = arr['x'].values

    start_time = time.time()

    rechunked_dt: xr.DataTree = xr.DataTree()
    ouput_group = zarr.open_group(output_path)
    processed_groups = {}
    crs = CRS.from_epsg(4326)
    measurement_group_path: str | None = None

    # Truncate any pre-existing store first: the writes below are per-group
    # (mode="w" scoped to measurements/r0, mode="a" for overviews/ancillary),
    # so a prior run with more overview levels or extra ancillary groups would
    # otherwise leave stale sibling groups behind, and the returned DataTree
    # (built by re-scanning the store) would surface them.
    zarr.open_group(output_path, mode="w", zarr_format=3)

    ### apply pre-rechunking ###
    for group_path in dt_input.groups:
        # empty root node
        if group_path == "/":
            rechunked_dt.attrs = dt_input.attrs
            continue

        group_node = dt_input[group_path]

        base_dataset = group_node.to_dataset()

        # Skip empty groups
        if not base_dataset.data_vars and base_dataset.attrs == {}:
            log.info("Skipping empty group: ", group_path=group_path)
            continue

        log.info(
            "Applying rechunking to original group",
            spatial_chunk=spatial_chunk,
            group_path=group_path,
        )

        dataset = utils._rechunk_ds(base_dataset, spatial_chunk)

        # gEt the encoding
        encoding = utils.create_uniform_encoding(
            dataset,
            spatial_chunk=spatial_chunk,
            enable_sharding=enable_sharding,
            keep_scale_offset=keep_scale_offset,
            compression_level=compression_level,
            chunk_and_shard_coords=chunk_and_shard_coords,
        )

        # convert float64 arrays to float32. `xr.DataArray.astype` clears
        # encoding, so we capture and restore it — downstream pyramid
        # levels are coarsened from this dataset and rely on the encoding
        # to drive CF packing / codec filter generation.
        for data_var in dataset.data_vars:
            if dataset[data_var].dtype in (np.dtype("<f8"), np.dtype(">f8")):
                var_encoding = dataset[data_var].encoding
                dataset[data_var] = dataset[data_var].astype("float32")
                dataset[data_var].encoding = var_encoding
        # Clear _FillValue from the DataArray's own encoding to prevent
        # xarray from raising "Zarr does not support _FillValue in encoding".
        if not keep_scale_offset:
            for data_var in dataset.data_vars:
                dataset[data_var].encoding.pop("_FillValue", None)

        # reprojection with gcps
        if "/measurements" in group_path:
            # Add the geo metadata before writing for geozarr
            utils.write_geo_metadata(dataset, crs=crs, input_is_image_array=False)

            # add overview base path here
            measurement_group_path = group_path
            # instead of this
            # measurement_group_path = f"{group_path}/r0"

            measurements = utils.stream_write_dataset(
                dataset,
                path=measurement_group_path,
                group=ouput_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                chunk_and_shard_coords=chunk_and_shard_coords,
                # crs=crs,
            )

            processed_groups[measurement_group_path] = measurements
        else:
            # Write dataset -> adds geo metadata
            ds_out = utils.stream_write_dataset(
                dataset,
                path=group_path,
                group=ouput_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                chunk_and_shard_coords=chunk_and_shard_coords,
                # crs=crs,
            )
            processed_groups[group_path] = ds_out

        rechunked_dt[group_path] = dataset

    # root level consolidation
    # utils.simple_root_consolidation(dt_input, output_path, processed_groups)
    utils.updated_root_consolidation(dt_input, output_path, processed_groups)

    # Create result DataTree
    result_dt = utils.create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    utils.optimization_summary(dt_input, result_dt, output_path)

    return xr.open_datatree(
        output_path,
        engine="zarr",
        chunks={},
        consolidated=False,
        mask_and_scale=False,
    )
