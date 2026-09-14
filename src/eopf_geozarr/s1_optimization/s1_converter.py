from __future__ import annotations

import time
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog
import xarray as xr
import zarr
from pyproj import CRS

from eopf_geozarr.conversion import utils
from eopf_geozarr.s1_optimization.sentinel1_reprojection import reproject_sentinel1_with_gcps

if TYPE_CHECKING:
    from zarr.core.common import JSON
    from zarr_cm import LayoutObject, MultiscalesAttrs, Transform

log = structlog.get_logger()


def flatten_dynamic_root_name(dt: xr.DataTree) -> xr.DataTree:
    """Remove the single dynamic product-name group under root, promoting
    all of its children (and merging its attrs) up to root itself."""
    if len(dt.children) != 1:
        raise ValueError(
            f"expected exactly one top-level child (the product name), got {list(dt.children)}"
        )

    product_name = next(iter(dt.children))
    product_node = dt[product_name]

    for child_name in list(product_node.children):
        dt[child_name] = product_node[child_name]

    dt.attrs = {**product_node.attrs, **dt.attrs}

    return dt.drop_nodes(product_name)


def calculate_s1grdh_multiscales(
    measurements_ds: xr.Dataset,
    gcp_ds: xr.Dataset,
    output_path: str,
    output_group: zarr.Group,
    product_name: str,
    pyramid_dims: tuple[str, str] = ("y", "x"),
    spatial_chunk: int = 1024,
    min_dimension: int = 256,
    crs: CRS | None = None,
    enable_sharding: bool = True,
    keep_scale_offset: bool = True,
    compression_level: int = 3,
    **kwargs: Any,
) -> dict[str, xr.Dataset]:  # Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]:
    # resolution_groups: dict[str, xr.Dataset] = {}
    base_path = f"{product_name}/multiscales"

    log.info("Applying Sentinel-1 reprojection for measurements")
    reproj_dataset = reproject_sentinel1_with_gcps(measurements_ds, gcp_ds, target_crs="EPSG:4326")

    # write reproj dataset to disk
    reproj_dataset = utils._rechunk_ds(reproj_dataset, spatial_chunk)
    # Measurement groups: apply custom encoding
    encoding = utils.create_uniform_encoding(
        reproj_dataset,
        spatial_chunk=spatial_chunk,
        shard_along_smallest_dimension=True,
        enable_sharding=enable_sharding,
        keep_scale_offset=keep_scale_offset,
        compression_level=compression_level,
    )
    utils.write_geo_metadata(reproj_dataset, crs=crs, input_is_image_array=False)

    # Write dataset -> adds geo metadata
    reproj_dataset = utils.stream_write_dataset(
        reproj_dataset,
        path=f"{base_path}/r2",
        group=output_group,
        encoding=encoding,
        enable_sharding=enable_sharding,
        # crs=crs,
    )

    # measurements_ds = reproj_dataset

    # Write /2 reduced overview subgroups: r2, r4, r8, …
    rows = reproj_dataset.sizes[pyramid_dims[0]]
    cols = reproj_dataset.sizes[pyramid_dims[1]]
    n_levels = utils.overview_levels(rows, cols, min_dimension)
    log.info("Generating overview levels", n_levels=n_levels)

    level_datasets: dict[str, xr.Dataset] = {"r2": reproj_dataset}
    scale_levels: dict[str, list[float]] = {
        "r2": [2.0, 2.0]
    }  # -> change scale measurements_ds.shae ??

    reproj_trafo = list(reproj_dataset.rio.transform())
    reproj_trafo = reproj_trafo[:6] if len(reproj_trafo) > 6 else reproj_trafo
    spatial_levels: dict[str, dict[str, list[float] | list[int]]] = {
        "r2": {"spatial:transform": reproj_trafo, "spatial:shape": [rows, cols]}
    }

    current = reproj_dataset

    # assumption of same shape in multiscale arrays -> this assumption is also applied in the geozarr spec so it should be alright
    curr_shape = next(iter(reproj_dataset.data_vars.values())).shape
    if len(curr_shape) == 2:
        curr_shape_xy = curr_shape
    elif len(curr_shape) == 3:
        # catch multi-dim shapes -> ignore polarisations
        curr_shape_xy = curr_shape[1:]
    else:
        curr_shape_xy = curr_shape

    for level in range(2, n_levels + 1):
        # Downsample all variables using existing lazy operations
        group_name = f"r{level * 2}"
        level_datasets[group_name] = current
        output_filepath = f"{base_path}/{group_name}"
        log.info("Calculating overview", group=output_filepath, shape=dict(current.sizes))

        lazy_vars = {}
        for var_name, var_data in current.data_vars.items():
            if var_data.ndim < 2:
                continue

            lazy_vars[var_name] = utils.coarsen_variable(
                str(var_name), var_data, factor=2
            )  # , other_fill_value=0

        # Create dataset with lazy variables and coordinates
        current = xr.Dataset(lazy_vars, attrs=measurements_ds.attrs)

        # calculate scale level beforehand
        downsample_shape = next(iter(current.data_vars.values())).shape
        if len(downsample_shape) == 2:
            # catch multi-dim shapes -> ignore polarisations
            downsample_shape_xy = downsample_shape[1:]
        if len(downsample_shape) == 3:
            # catch multi-dim shapes -> ignore polarisations
            downsample_shape_xy = downsample_shape[1:]
        else:
            downsample_shape_xy = downsample_shape

        scales = [c / d for c, d in zip(curr_shape_xy, downsample_shape_xy, strict=True)]
        scale_levels[group_name] = scales

        # we only need the first 6 values form affine traFo
        downsampled_trafo = list(current.rio.transform())
        downsampled_trafo = (
            downsampled_trafo[:6] if len(downsampled_trafo) > 6 else downsampled_trafo
        )

        spatial_levels[group_name] = {
            "spatial:transform": downsampled_trafo,
            "spatial:shape": list(downsample_shape_xy),
        }

        curr_shape_xy = downsample_shape_xy

        # remove parent encoding
        current = utils.clear_encoding(current)
        dataset = utils._rechunk_ds(current, spatial_chunk)
        # vals = dataset['grd'].values

        # Measurement groups: apply custom encoding
        encoding = utils.create_uniform_encoding(
            dataset,
            spatial_chunk=spatial_chunk,
            enable_sharding=enable_sharding,
            shard_along_smallest_dimension=False,
            keep_scale_offset=keep_scale_offset,
            compression_level=compression_level,
        )

        # Strip _FillValue from DataArray encoding for downsampled levels too
        if not keep_scale_offset:
            for data_var in current.data_vars:
                current[data_var].encoding.pop("_FillValue", None)

        # Add the geo metadata before writing for
        utils.write_geo_metadata(dataset, crs=crs)

        ds_out = utils.stream_write_dataset(
            dataset,
            path=output_filepath,
            group=output_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
            # crs=crs,
        )

        # issues:
        # not considering 0 or 65?? nodata vals -> issue in resampling during coarsening

        # Store results -> add metadta to zarr root here!
        level_datasets[group_name] = ds_out
        # resolution_groups[group_name] = ds_out

    layout: list[LayoutObject] = [
        {
            "asset": "r2",
            "spatial:transform": spatial_levels["r2"]["spatial:transform"],
            "spatial:shape": spatial_levels["r2"]["spatial:shape"],
        }
    ]

    for level in range(2, n_levels + 1):
        group_name = f"r{level * 2}"
        transform: Transform = {"scale": scale_levels[group_name], "translation": [0.0, 0.0]}
        lo: LayoutObject = {
            "asset": group_name,
            "derived_from": f"r{level * 2 - 2}" if level > 2 else "r2",
            "transform": transform,
            # add spatial ones
            "spatial:shape": spatial_levels[group_name]["spatial:shape"],
            "spatial:transform": spatial_levels[group_name]["spatial:transform"],
        }
        layout.append(lo)

    # add metadata to layout
    # "resampling_method": "average"

    # add metadata to root and multiscale-parent node
    root_rw = zarr.open_group(output_path, mode="a")

    base_spatial = utils.grid_spatial_attrs(
        transform=reproj_dataset.rio.transform(recalc=True),
        shape=(reproj_dataset.sizes["y"], reproj_dataset.sizes["x"]),
    )

    for group_name, level_ds in level_datasets.items():
        _level_spatial = utils.grid_spatial_attrs(
            transform=level_ds.rio.transform(recalc=True),
            shape=(level_ds.sizes["y"], level_ds.sizes["x"]),
        )

        level_conv = utils.build_convention_attrs(spatial=_level_spatial, crs=crs)

        root_rw[f"{base_path}/{group_name}"].attrs.update(cast("dict[str, JSON]", level_conv))

    if n_levels > 0:
        ms: MultiscalesAttrs = {"layout": layout, "resampling_method": "average"}
        conv = utils.build_convention_attrs(multiscales=ms, spatial=base_spatial, crs=crs)
    else:
        conv = utils.build_convention_attrs(spatial=base_spatial, crs=crs)

    root_rw[base_path].attrs.update(cast("dict[str, JSON]", conv))

    return level_datasets


def convert_s1grdh_optimized(
    dt_input: xr.DataTree,
    *,
    enable_sharding: bool,
    output_path: str,
    spatial_chunk: int,
    compression_level: int,
    # validate_output: bool,
    keep_scale_offset: bool,
    # max_retries: int = 3,
    gcp_group: str = "/conditions/gcp",
) -> dict[str, dict]:
    start_time = time.time()

    ouput_group = zarr.open_group(output_path)
    processed_groups = {}
    crs = CRS.from_epsg(4326)
    measurement_group_path: str | None = None

    # remove name from each node
    # dt = flatten_dynamic_root_name(dt=dt_input)
    dt = dt_input

    # data is polarised as vv and vh in same dataset -> redundant gcps
    product_name = next(iter(dt.children))
    ds_gcp = dt[f"{product_name}/{gcp_group}"].to_dataset()
    if ds_gcp["polarization"].shape[0] > 1:
        arrs = [ds_gcp.isel(polarization=i) for i in range(ds_gcp.polarization.shape[0])]

        # just select the first polarisation lvl if its similar
        if all((arr1 == arr2).all() for arr1, arr2 in pairwise(arrs)):
            ds_gcp = ds_gcp.isel(polarization=0)

    # rechunk everything
    for group_path in dt.groups:
        if group_path == "/":
            continue

        group_node = dt[group_path]

        # Skip parent groups that have children (only process leaf groups)
        if hasattr(group_node, "children") and len(group_node.children) > 0:
            continue

        base_dataset = group_node.to_dataset()

        # Skip empty groups
        if not base_dataset.data_vars:
            log.info("Skipping empty group: {}", group_path=group_path)
            continue

        log.info("Copying original group: {}", group_path=group_path)

        dataset = utils._rechunk_ds(base_dataset, spatial_chunk)

        # reprojection with gcps
        if "/measurements" in group_path:
            dataset = utils._rechunk_ds(dataset, spatial_chunk)

            # del reproj_dataset
            # gc.collect()

            # Measurement groups: apply custom encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
                shard_along_smallest_dimension=True,
                enable_sharding=enable_sharding,
                keep_scale_offset=keep_scale_offset,
                compression_level=compression_level,
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

            # rewrite Grup path to allow multiscales
            # measurement_group_path = f"{group_path}/r0"
            measurement_group_path = f"{group_path}"

            # Add the geo metadata before writing for geozarr
            utils.write_geo_metadata(dataset, crs=crs, input_is_image_array=False)

            # Write dataset -> adds geo metadata
            measurements = utils.stream_write_dataset(
                dataset,
                path=measurement_group_path,
                group=ouput_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                # crs=crs,
            )
            processed_groups[measurement_group_path] = measurements
        else:
            encoding = utils.create_uniform_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
                enable_sharding=enable_sharding,
                shard_along_smallest_dimension=False,
                keep_scale_offset=keep_scale_offset,
                compression_level=compression_level,
            )

            # add geo metadata?

            # Write dataset -> adds geo metadata
            ds_out = utils.stream_write_dataset(
                dataset,
                path=group_path,
                group=ouput_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                # crs=crs,
            )
            processed_groups[group_path] = ds_out

    if measurement_group_path is None:
        raise ValueError("No '/measurements' group found in input DataTree")

    measurement_ds = processed_groups[measurement_group_path]
    crs = measurement_ds.rio.crs

    calculate_s1grdh_multiscales(
        measurement_ds,
        ds_gcp,
        output_path=output_path,
        output_group=ouput_group,
        product_name=product_name,
        crs=crs,
        # pyramid_dims=pyramids_factors,
        enable_sharding=enable_sharding,
        keep_scale_offset=keep_scale_offset,
        compression_level=compression_level,
        spatial_chunk=spatial_chunk,
    )

    # issues:
    # - wgs84 crs -> scale from transform? weird scale values
    # - what pyramid levels do we generally want?

    # root level consolidation
    utils.simple_root_consolidation(dt_input, output_path, processed_groups)

    # Create result DataTree
    result_dt = utils.create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    utils.optimization_summary(dt_input, result_dt, output_path)

    return processed_groups
