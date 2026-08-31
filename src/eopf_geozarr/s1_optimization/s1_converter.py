import xarray as xr
import structlog
from eopf_geozarr.conversion import utils
from eopf_geozarr.s1_optimization.sentinel1_reprojection import reproject_sentinel1_with_gcps
from eopf_geozarr.conversion.fs_utils import sanitize_dataset_attributes, get_storage_options
from itertools import pairwise
from typing import List, TYPE_CHECKING, cast
import zarr
import numpy as np
import time
from collections.abc import Hashable, Mapping
from eopf_geozarr.data_api.geozarr.types import (
    CF_SCALE_OFFSET_KEYS,
    XARRAY_ENCODING_KEYS,
    XarrayDataArrayEncoding,
)
from eopf_geozarr.s2_optimization.common import DISTRIBUTED_AVAILABLE

if TYPE_CHECKING:
    from zarr_cm import MultiscalesAttrs
    from zarr_cm import spatial as spatial_cm

    from eopf_geozarr.types import OverviewLevelJSON
    from collections.abc import Mapping

from pyproj import CRS

log = structlog.get_logger()



def __transform_from_coordinates(
    dataset: xr.Dataset,
) -> tuple[float, float, float, float, float, float] | None:
    """Construct an affine transform from dataset coordinates when possible."""
    if "x" not in dataset.coords or "y" not in dataset.coords:
        return None

    x_coords = dataset.coords["x"].values
    y_coords = dataset.coords["y"].values
    if len(x_coords) < 2 or len(y_coords) < 2:
        return None

    pixel_size_x = float(np.abs(x_coords[1] - x_coords[0]))
    pixel_size_y = float(np.abs(y_coords[1] - y_coords[0]))
    x_min = float(x_coords.min())
    y_max = float(y_coords.max())
    return (pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max)



def __rio_transform_matches_coordinates(
    transform: tuple[float, float, float, float, float, float] | None,
    coordinate_transform: tuple[float, float, float, float, float, float] | None,
) -> bool:
    """Check whether rio-derived metadata matches the current x/y grid."""
    if transform is None or coordinate_transform is None:
        return False

    return all(np.isclose(a, b) for a, b in zip(transform, coordinate_transform, strict=False))


def __preferred_spatial_transform(
    dataset: xr.Dataset,
) -> tuple[float, float, float, float, float, float] | None:
    """Prefer rio metadata only when it matches the current coordinate grid."""
    coordinate_transform = __transform_from_coordinates(dataset)
    rio_transform: tuple[float, float, float, float, float, float] | None = None

    if hasattr(dataset, "rio") and hasattr(dataset.rio, "transform"):
        try:
            rio_value = dataset.rio.transform
            if callable(rio_value):
                rio_value = rio_value()
            # rio transform value is dynamically typed; it is iterable at runtime.
            rio_iter = cast("tuple[float, ...]", tuple(rio_value))  # pyright: ignore[reportArgumentType]
            rio_values = tuple(float(value) for value in rio_iter[:6])
            if len(rio_values) == 6:
                rio_transform = (
                    rio_values[0],
                    rio_values[1],
                    rio_values[2],
                    rio_values[3],
                    rio_values[4],
                    rio_values[5],
                )
        except (AttributeError, TypeError, ValueError):
            rio_transform = None

    if (
        rio_transform is not None
        and not all(value == 0 for value in rio_transform)
        and __rio_transform_matches_coordinates(rio_transform, coordinate_transform)
    ):
        return rio_transform

    return coordinate_transform or rio_transform



def __write_geo_metadata(
    dataset: xr.Dataset,
    grid_mapping_var_name: str = "spatial_ref",
    crs: CRS | None = None,
) -> None:
    """
    Write geographic metadata to the dataset.

    Args:
        dataset: Dataset to write metadata to
        grid_mapping_var_name: Name for grid mapping variable
        crs: Coordinate Reference System to use (if None, attempts to detect from dataset)
    """
    # Use provided CRS or try to detect from dataset
    if crs is None:
        for var in dataset.data_vars.values():
            if hasattr(var, "rio") and var.rio.crs:
                crs = var.rio.crs
                break
            if "proj:epsg" in var.attrs:
                epsg = var.attrs["proj:epsg"]
                crs = CRS.from_epsg(epsg)
                break

    if crs is not None:
        # Write CRS using rioxarray
        # NOTE: for now rioxarray only supports writing grid mapping using CF conventions
        dataset.rio.write_crs(crs, grid_mapping_name=grid_mapping_var_name, inplace=True)
        dataset.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
        dataset.attrs["grid_mapping"] = grid_mapping_var_name

        for var in dataset.data_vars.values():
            var.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
            var.attrs["grid_mapping"] = grid_mapping_var_name

        # Also add proj: and spatial: zarr conventions at dataset level
        # TODO : Remove once rioxarray supports writing these conventions directly
        # https://github.com/corteva/rioxarray/pull/883

        # Assemble spatial convention data
        spatial_data: spatial_cm.SpatialAttrs = {
            "spatial:dimensions": ["y", "x"],  # Required field
            "spatial:registration": "pixel",  # Default registration type
        }

        # Calculate and add spatial bbox if coordinates are available
        if "x" in dataset.coords and "y" in dataset.coords:
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values
            x_min, x_max = float(x_coords.min()), float(x_coords.max())
            y_min, y_max = float(y_coords.min()), float(y_coords.max())
            spatial_data["spatial:bbox"] = [x_min, y_min, x_max, y_max]

            spatial_transform = __preferred_spatial_transform(dataset)

            # Only add spatial:transform if we have valid transform data (not all zeros)
            if spatial_transform is not None and not all(t == 0 for t in spatial_transform):
                spatial_data["spatial:transform"] = list(spatial_transform)

            # Add spatial shape if data variables exist
            if dataset.data_vars:
                first_var = next(iter(dataset.data_vars.values()))
                if first_var.ndim >= 2:
                    height, width = first_var.shape[-2:]
                    spatial_data["spatial:shape"] = [height, width]

        # Build validated spatial + proj convention attrs (data + CMOs) via zarr-cm
        dataset.attrs.update(utils.build_convention_attrs(spatial=spatial_data, crs=crs))


def __stream_write_dataset(
    dataset: xr.Dataset,
    *,
    path: str,
    group: zarr.Group,
    encoding: dict[str, XarrayDataArrayEncoding],
    enable_sharding: bool,
    crs: CRS | None = None,
) -> xr.Dataset:
    """
    Stream write a lazy dataset with advanced chunking and sharding.

    This is where the magic happens: all the lazy downsampling operations
    are executed as the data is streamed to storage with optimal performance.

    Args:
        dataset: Dataset to write
        dataset_path: Output path for dataset
        encoding: Encoding dictionary for variables
        enable_sharding: Enable Zarr v3 sharding
        crs: Coordinate Reference System for geographic metadata

    Returns:
        Written dataset
    """
    # Check if level already exists
    if path in group:
        log.info(
            "Level path {} already exists. Skipping write.",
            dataset_path=path,
        )
        # The zarr backend accepts a zarr `Store` here at runtime, but xarray's
        # `open_dataset` stub only types the first arg as path/buffer/datastore.
        return xr.open_dataset(
            group.store,  # type: ignore[arg-type]
            engine="zarr",
            chunks={},
            decode_coords="all",
            group=path,
        )

    log.info("Streaming computation and write to {}", dataset_path=path)
    log.info("Variables", variables=list(dataset.data_vars.keys()))

    # Rechunk dataset to align with encoding
    dataset = utils.rechunk_dataset_for_encoding(dataset, encoding)

    # Add the geo metadata before writing for
    # - /measurements/ groups -> now inly single one
    if "/measurements" in path:
        __write_geo_metadata(dataset, crs=crs)

    # Sanitize NaN values in dataset attributes before writing
    dataset = sanitize_dataset_attributes(dataset)

    # Write with streaming computation and progress tracking
    # The to_zarr operation will trigger all lazy computations
    write_job = dataset.to_zarr(
        group.store,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding=encoding,
        group=path,
        compute=False,  # Create job first for progress tracking
    )
    write_job = write_job.persist()

    if DISTRIBUTED_AVAILABLE:
        try:
            import distributed

            # Try to get current client for better status monitoring
            try:
                client = distributed.Client.current()
                # client.compute is untyped (returns Any); verify we got a
                # Future rather than asserting it with a cast.
                future = client.compute(write_job)
                if not isinstance(future, distributed.Future):
                    raise TypeError(f"expected a distributed.Future, got {type(future).__name__}")
                log.info("Using distributed client for write job monitoring")

                try:
                    distributed.progress(future, notebook=False)
                except Exception as progress_error:
                    log.warning("Could not display progress bar: {}", e=progress_error)

                # Get result and raise if computation failed
                future.result()
            except ValueError:
                # No current client, fall back to regular distributed.progress
                log.info("No distributed client available, using regular progress")
                distributed.progress(write_job, notebook=False)
                write_job.compute()

        except Exception as e:
            log.warning("Could not use distributed features: {}", e=e)
            write_job.compute()
    else:
        log.info("Writing zarr file...")
        write_job.compute()

    log.info("✅ Streaming write complete for dataset {}", dataset_path=path)
    return dataset


def flatten_dynamic_root_name(dt: xr.DataTree) -> xr.DataTree:
    """Remove the single dynamic product-name group under root, promoting
    all of its children (and merging its attrs) up to root itself."""
    if len(dt.children) != 1:
        raise ValueError(f"expected exactly one top-level child (the product name), got {list(dt.children)}")

    product_name = next(iter(dt.children))
    product_node = dt[product_name]

    for child_name in list(product_node.children):
        dt[child_name] = product_node[child_name]

    dt.attrs = {**product_node.attrs, **dt.attrs}

    return dt.drop_nodes(product_name)


def add_multiscales(dt_input: xr.DataTree,
                    pyramid_lvls: List[int],
                    ) -> xr.DataTree:

    return dt_input

def convert_s1grdh_optimized(dt_input: xr.DataTree,
                                *,
                                enable_sharding: bool,
                                output_path: str,
                                spatial_chunk: int,
                                compression_level: int,
                                validate_output: bool,
                                keep_scale_offset: bool,
                                max_retries: int = 3,
                                gcp_group: str = '/conditions/gcp'
                            ) -> dict[str, dict]:
    start_time = time.time()
    
    ouput_group = zarr.open_group(output_path)
    processed_groups = {}
    crs = None

    # remove name from each node
    dt = flatten_dynamic_root_name(dt=dt_input)

    # data is polarised as vv and vh in same dataset -> redundant gcps
    ds_gcp = dt[gcp_group].to_dataset()
    if ds_gcp['polarization'].shape[0] > 1:
        arrs = [ds_gcp.isel(polarization=i) for  i in range(ds_gcp.polarization.shape[0])]

        # just select the first polarisation lvl if its similar
        if bool([(arr1 == arr2).all() for arr1, arr2 in pairwise(arrs)]):
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
            log.info("Applying Sentinel-1 reprojection for group %s", group_path)
            reproj_dataset = reproject_sentinel1_with_gcps(dataset, ds_gcp, target_crs="EPSG:4326")

            # for debugging dont transform 
            # reproj_dataset = dataset
            
            dataset = utils._rechunk_ds(reproj_dataset, spatial_chunk)

            # Measurement groups: apply custom encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
                enable_sharding=enable_sharding,
                keep_scale_offset=keep_scale_offset,
                compression_level=compression_level,
            )
        else:
            encoding = utils.create_uniform_encoding(
                            dataset,
                            spatial_chunk=spatial_chunk,
                            enable_sharding=enable_sharding,
                            keep_scale_offset=keep_scale_offset,
                            compression_level=compression_level,
                        )

        # Write dataset
        ds_out = __stream_write_dataset(
            dataset,
            path=group_path,
            group=ouput_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
            crs=crs,
        )
        processed_groups[group_path] = ds_out
        pass

        # add pyramids

        # add geo metadata

    # root level consolidation
    __simple_root_consolidation(dt_input, output_path, processed_groups)

    # Create result DataTree
    result_dt = __create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    __optimization_summary(dt_input, result_dt, output_path)

    return processed_groups


def __optimization_summary(dt_input: xr.DataTree, dt_output: xr.DataTree, output_path: str) -> None:
    """Print optimization summary statistics."""
    # Count groups
    input_groups = len(dt_input.groups) if hasattr(dt_input, "groups") else 0
    output_groups = len(dt_output.groups) if hasattr(dt_output, "groups") else 0

    log.info(
        "OPTIMIZATION SUMMARY",
        input_groups=input_groups,
        output_groups=output_groups,
        output_path=output_path,
        groups=[g for g in dt_output.groups if g != "."],
    )


def __create_result_datatree(output_path: str) -> xr.DataTree:
    """Create result DataTree from written output."""
    storage_options = get_storage_options(output_path)
    return xr.open_datatree(
        output_path,
        engine="zarr",
        chunks="auto",
        storage_options=storage_options,
    )


def __write_store_root_bbox(output_path: str) -> None:
    """Write the minispec store-root metadata (bbox, CRS, conventions).

    Thin wrapper kept for backwards compatibility; the implementation lives in
    :func:`eopf_geozarr.conversion.utils.write_store_root_geo_metadata`.
    """
    utils.write_store_root_geo_metadata(output_path)


def __simple_root_consolidation(dt_input: xr.DataTree, output_path: str, datasets: Mapping[str, object]) -> None:
    """Simple root-level metadata consolidation with proper zarr group creation."""
    # create missing intermediary groups (/conditions, /quality, etc.)
    # using the keys of the datasets dict
    missing_groups = set()
    for group_path in datasets:
        # extract all the parent paths
        parts = group_path.strip("/").split("/")
        for i in range(1, len(parts)):
            parent_path = "/" + "/".join(parts[:i])
            if parent_path not in datasets:
                missing_groups.add(parent_path)

    for group_path in missing_groups:
        dt_parent = xr.DataTree()
        dt_parent.to_zarr(
            output_path + group_path,
            mode="a",
            zarr_format=3,
            consolidated=False,
        )

    # Create root zarr group if it doesn't exist
    log.info("Creating root zarr group")
    dt_root = xr.DataTree()
    dt_root.to_zarr(
        output_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )
    dt_root = xr.DataTree()
    for group_path in datasets:
        dt_root[group_path] = xr.DataTree()

    dt_root.to_zarr(
        output_path,
        mode="r+",
        consolidated=False,
        zarr_format=3,
    )
    log.info("Root zarr group created")

    # Write the store-root spatial footprint (geozarr minispec, Store Root section).
    # Aggregates child-group `spatial:bbox` values, reprojects them to EPSG:4326
    # and writes the union on the root `zarr.json`.
    __write_store_root_bbox(output_path)

    utils.write_store_root_stac_metadata(output_path, root_attrs=dt_input.attrs)  # type: ignore

    # consolidate reflectance group metadata
    zarr.consolidate_metadata(output_path + "/measurements", zarr_format=3)

    # consolidate root group metadata
    zarr.consolidate_metadata(output_path, zarr_format=3)