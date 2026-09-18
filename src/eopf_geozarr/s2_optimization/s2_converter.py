"""
S2 optimization converter with streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from __future__ import annotations

import time
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import structlog
import xarray as xr
import zarr
from pydantic.experimental.missing_sentinel import MISSING
from pyproj import CRS
from zarr_cm import LayoutObject, MultiscalesAttrs, Transform

from eopf_geozarr import zcm
from eopf_geozarr.conversion import sentinel_modes, utils
from eopf_geozarr.s2_optimization.s2_band_mapping import BAND_INFO

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.core.common import JSON

    from eopf_geozarr.new_types import OverviewLevelJSON


log = structlog.get_logger()

MultiscalesFlavor = Literal["experimental_multiscales_convention"]

pyramid_levels = {
    0: 10,  # Level 0: 10m (native for b02,b03,b04,b08)
    1: 20,  # Level 1: 20m (native for b05,b06,b07,b11,b12,b8a + all quality)
    2: 60,  # Level 2: 60m (native for b01,b09,b10)
    3: 120,  # Level 3: 120m (2x downsampling from 60m)
    4: 360,  # Level 4: 360m (3x downsampling from 120m)
    5: 720,  # Level 5: 720m (2x downsampling from 360m)
}


def _transform_from_coordinates(
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
    # x_min = float(x_coords.min())
    # y_max = float(y_coords.max())

    # Coordinates label pixel centres, but an affine transform maps pixel *edges*,
    # so the origin sits half a pixel outside the first centre. Without this the
    # transform can never agree with rioxarray's and is half a cell off.
    x_min = float(x_coords.min()) - pixel_size_x / 2
    y_max = float(y_coords.max()) + pixel_size_y / 2

    return (pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max)


def _rio_transform_matches_coordinates(
    transform: tuple[float, float, float, float, float, float] | None,
    coordinate_transform: tuple[float, float, float, float, float, float] | None,
) -> bool:
    """Check whether rio-derived metadata matches the current x/y grid."""
    if transform is None or coordinate_transform is None:
        return False

    return all(np.isclose(a, b) for a, b in zip(transform, coordinate_transform, strict=False))


def _preferred_spatial_transform(
    dataset: xr.Dataset,
) -> tuple[float, float, float, float, float, float] | None:
    """Prefer rio metadata only when it matches the current coordinate grid."""
    coordinate_transform = _transform_from_coordinates(dataset)
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
        and _rio_transform_matches_coordinates(rio_transform, coordinate_transform)
    ):
        return rio_transform

    return coordinate_transform or rio_transform


def inject_missing_bands(
    dataset: xr.Dataset,
    dt_input: xr.DataTree,
    target_resolution: int,
    spatial_chunk: int,
    *,
    bands: set[str] | None = None,
) -> xr.Dataset:
    """Inject bands whose native resolution is finer than `target_resolution`.

    For each spectral band defined in `BAND_INFO` whose native resolution is
    finer than `target_resolution`, this function checks whether the band is
    already present in `dataset`.  If not, it looks for the band in the
    appropriate source group (e.g. `/measurements/reflectance/r10m`),
    downsamples it to the target grid using the type-aware resampling from
    `determine_variable_type`, and merges it into `dataset`.

    Args:
        dataset: The target-resolution dataset (e.g. the r20m or r60m
            reflectance group).
        dt_input: The full input DataTree (used to locate finer-resolution
            source bands).
        target_resolution: Target resolution in metres (e.g. 20 or 60).
        spatial_chunk: Spatial chunk size
        bands: If provided, only inject these band names.  If `None`
            (default), inject every eligible band from `BAND_INFO`.

    Returns:
        `dataset` with any missing finer-resolution bands added.
    """
    for band_name, info in BAND_INFO.items():
        if bands is not None and band_name not in bands:
            continue
        native_res = info.native_resolution  # type: ignore[attr-defined]
        if native_res >= target_resolution:
            continue
        if band_name in dataset.data_vars:
            continue

        source_path = f"/measurements/reflectance/r{native_res}m"
        if source_path not in dt_input.groups:
            continue

        source_ds = dt_input[source_path].to_dataset()

        if band_name not in source_ds.data_vars:
            continue

        band_src = source_ds[band_name]
        factor = target_resolution // native_res
        band_ds = utils.coarsen_variable(band_name, band_src, factor)

        # add attribute value acknoleding the own resampling
        trgt_attrs = band_src.attrs
        trgt_attrs.update(
            {"_derived_from": f"r{native_res}m", "_factor": factor, "_resampling_mode": "mean"}
        )

        # Replace coordinates with the target dataset's coordinates so that
        # xarray.Dataset.assign does not try to align on mismatched values.
        band_ds = xr.DataArray(
            band_ds.values,
            dims=band_ds.dims,
            coords={d: dataset.coords[d] for d in band_ds.dims if d in dataset.coords},
            attrs=trgt_attrs,
            name=band_name,
        )

        # Preserve source encoding so downstream encoding logic can inspect it
        band_ds.encoding = band_src.encoding.copy()

        dataset = dataset.assign({band_name: band_ds})
        log.info(
            "Injected downsampled band from finer resolution",
            band=band_name,
            source=f"r{native_res}m",
            target=f"r{target_resolution}m",
            shape=band_ds.shape,
        )

    return utils._rechunk_ds(dataset, spatial_chunk)


def unused_create_multiscale_from_datatree(
    dt_input: xr.DataTree,
    *,
    output_group: zarr.Group,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    crs: CRS | None = None,
    keep_scale_offset: bool,
    compression_level: int = 3,
) -> dict[str, dict]:
    """
    Create multiscale versions preserving original structure.
    Keeps all original groups, adds r120m, r360m, r720m downsampled versions.

    Args:
        dt_input: Input DataTree with original structure
        output_path: Base output path
        enable_sharding: Enable Zarr v3 sharding
        spatial_chunk: Spatial chunk size
        crs: Coordinate Reference System for datasets

    Returns:
        Dictionary of processed groups
    """
    processed_groups: dict[str, Any] = {}

    # helper dicts to keep track of attrs for multiscales
    spatial_levels: dict[str, dict[str, list[float] | list[int]]] = {}

    # cheap determination if its L2A or L1C
    filename = dt_input.name
    s2_type = sentinel_modes.S2Type.from_filename(filename)

    if s2_type is None:
        log.info("not found a matching s2_type {}: is not matching L2A or L1C", s2_type=s2_type)

    # Step 1: Copy all original groups as-is
    for group_path in dt_input.groups:
        if group_path == ".":
            continue

        # Skip the quicklook groups (`/quality/l1c_quicklook`, `/quality/l2a_quicklook`).
        # They duplicate RGB data that downstream consumers can derive from the
        # reflectance bands, so dropping them shrinks the output store and
        # speeds up conversion. See EOPF-Explorer/data-model#81.
        if group_path.startswith(("/quality/l1c_quicklook", "/quality/l2a_quicklook")):
            log.info("Skipping quicklook group", group_path=group_path)
            continue

        group_node = dt_input[group_path]

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

        # Determine if this is a measurement-related resolution group
        group_name = group_path.split("/")[-1]
        is_measurement_group = (
            group_name.startswith("r")
            and group_name.endswith("m")
            and "/measurements/" in group_path
        )

        if "quality/probability" in group_path:
            pass

        if is_measurement_group:
            # Inject bands whose native resolution is finer than this group's
            # (e.g. b08 native at 10m into r20m/r60m) so they propagate through
            # the full overview chain (r120m … r720m).
            if group_path.startswith("/measurements/reflectance/"):
                try:
                    group_resolution = int(group_name[1:-1])
                except ValueError:
                    group_resolution = 0

                # just add the b08 band
                if s2_type == "L2A":
                    if group_resolution > 10:
                        dataset = inject_missing_bands(
                            dataset,
                            dt_input,
                            group_resolution,
                            spatial_chunk,
                            bands={"b08"},
                        )

                # add all lower level bands here!
                elif s2_type == "L1C":
                    if group_resolution == 20:
                        dataset = inject_missing_bands(
                            dataset,
                            dt_input,
                            group_resolution,
                            spatial_chunk,
                            bands={
                                "b02",
                                "b03",
                                "b04",
                                "b08",
                            },
                        )
                    elif group_resolution == 60:
                        dataset = inject_missing_bands(
                            dataset,
                            dt_input,
                            group_resolution,
                            spatial_chunk,
                            bands={
                                "b02",
                                "b03",
                                "b04",
                                "b08",
                                "b05",
                                "b06",
                                "b07",
                                "b8a",
                                "b11",
                                "b12",
                            },
                        )

            # Measurement groups: apply custom encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
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

            # Drop scalar (0-D) coordinates such as a source `band` label: the
            # minispec's DataArray rules forbid scalar arrays in a GeoZarr dataset.
            scalar_coords = [name for name, coord in dataset.coords.items() if coord.ndim == 0]
            if scalar_coords:
                dataset = dataset.drop_vars(scalar_coords)
                for name in scalar_coords:
                    encoding.pop(str(name), None)

            utils.write_geo_metadata(dataset, crs=crs)

            # add spatial: metadta to outside dict for multuiscale layouts
            spatial_levels[group_name] = {
                "spatial:shape": dataset.attrs["spatial:shape"],
                "spatial:transform": dataset.attrs["spatial:transform"],
            }

            ds_out = utils.stream_write_dataset(
                dataset,
                path=group_path,
                group=output_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                # crs=crs,
            )
            processed_groups[group_path] = ds_out

        else:
            # Non-measurement groups: preserve original encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
                enable_sharding=enable_sharding,
                keep_scale_offset=keep_scale_offset,
                compression_level=compression_level,
            )

            # Drop scalar (0-D) coordinates such as a source `band` label: the
            # minispec's DataArray rules forbid scalar arrays in a GeoZarr dataset.
            scalar_coords = [name for name, coord in dataset.coords.items() if coord.ndim == 0]
            if scalar_coords:
                dataset = dataset.drop_vars(scalar_coords)
                for name in scalar_coords:
                    encoding.pop(str(name), None)

            if "/quality/" in group_path or "/conditions/mask" in group_path:
                utils.write_geo_metadata(dataset, crs=crs)

            ds_out = utils.stream_write_dataset(
                dataset,
                path=group_path,
                group=output_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                # crs=crs,
            )
            processed_groups[group_path] = ds_out

    # predefined layout asset
    layout_: list[dict[str, Any]] = [
        {"asset": "r10m", **spatial_levels["r10m"]},
        {
            "asset": "r20m",
            "derived_from": "r10m",
            "transform": Transform({"scale": [2.0, 2.0], "translation": [0.0, 0.0]}),
            **spatial_levels["r20m"],
        },
        {
            "asset": "r60m",
            "derived_from": "r10m",
            "transform": Transform({"scale": [6.0, 6.0], "translation": [0.0, 0.0]}),
            **spatial_levels["r60m"],
        },
    ]

    scale_levels = tuple(pyramid_levels.values())

    # iterate over pre-defined pyramid-dict (or smth) and generate layout data -> use LayoutObject/...
    current = processed_groups["/measurements/reflectance/r60m"]
    current_level_name = "r60m"

    for src_scale_level, dst_scale_level in pairwise(scale_levels[2:]):
        dest_level_name = f"r{dst_scale_level}m"
        dest_level_path = f"/measurements/reflectance/{dest_level_name}"

        downsample_factor = dst_scale_level // src_scale_level
        log.info(
            "Creating level with resolution", level=dest_level_name, resolution=dst_scale_level
        )

        # Create downsampled dataset
        downsampled_dataset = create_downsampled_resolution_group(current, factor=downsample_factor)

        log.info("Writing level to path", level=dest_level_name, output_path=dest_level_path)

        # Create encoding
        encoding = utils.create_uniform_encoding(
            downsampled_dataset,
            spatial_chunk=spatial_chunk,
            enable_sharding=enable_sharding,
            keep_scale_offset=keep_scale_offset,
        )

        # Strip _FillValue from DataArray encoding for downsampled levels too
        if not keep_scale_offset:
            for data_var in downsampled_dataset.data_vars:
                downsampled_dataset[data_var].encoding.pop("_FillValue", None)

        # add geo metadata
        utils.write_geo_metadata(downsampled_dataset, crs=crs)

        transform: Transform = {
            "scale": [downsample_factor, downsample_factor],
            "translation": [0.0, 0.0],
        }

        spatial_levels[dest_level_name] = {
            "spatial:shape": downsampled_dataset.attrs["spatial:shape"],
            "spatial:transform": downsampled_dataset.attrs["spatial:transform"],
        }

        lo = {
            "asset": dest_level_name,
            "derived_from": current_level_name,
            "transform": transform,
            **spatial_levels[dest_level_name],
        }

        layout_.append(lo)

        # Write dataset
        ds_out = utils.stream_write_dataset(
            downsampled_dataset,
            path=dest_level_path,
            group=output_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
        )

        # Store results
        processed_groups[dest_level_path] = ds_out

        current = downsampled_dataset
        current_level_name = dest_level_name

    # add metadata to root and multiscale-parent node
    root_rw = zarr.open_group(output_path, mode="a")

    # create layoutobjects -> also checks accordance (i think)
    layout: list[LayoutObject] = [LayoutObject(**lo) for lo in layout_]
    ms: MultiscalesAttrs = {"layout": layout, "resampling_method": "average"}

    # add geozarr attrs to base of /measurements/reflectance/
    base_measurement_ds_10m = processed_groups["/measurements/reflectance/r10m"]
    base_spatial = utils.grid_spatial_attrs(
        transform=base_measurement_ds_10m.rio.transform(recalc=True),
        shape=(base_measurement_ds_10m.sizes["y"], base_measurement_ds_10m.sizes["x"]),
    )

    conv = utils.build_convention_attrs(multiscales=ms, spatial=base_spatial, crs=crs)
    root_rw["/measurements/reflectance/"].attrs.update(cast("dict[str, JSON]", conv))

    processed_groups["/measurements/reflectance"] = None

    return processed_groups


def unused_add_multiscales_metadata_to_parent(
    group: zarr.Group,
    res_groups: Mapping[str, xr.Dataset],
) -> Any:
    """Add GeoZarr-compliant multiscales metadata to parent group.

    Returns ``None`` in all cases: metadata is written directly to ``group``
    via ``group.attrs.update`` rather than returned as a DataTree.
    """
    # Sort by resolution (finest to coarsest)
    res_order = {
        "r10m": 10,
        "r20m": 20,
        "r60m": 60,
        "r120m": 120,
        "r360m": 360,
        "r720m": 720,
    }

    all_resolutions = sorted(set(res_groups.keys()), key=lambda x: res_order.get(x, 999))

    if len(all_resolutions) < 2:
        log.info(
            "Skipping {} - only one resolution available",
            base_path=group.path,
        )
        return None

    # Get CRS and bounds from first available dataset (load from output path)
    first_res = all_resolutions[0]
    first_dataset = res_groups[first_res]

    # Get CRS and bounds
    native_crs = first_dataset.rio.crs if hasattr(first_dataset, "rio") else None
    if native_crs is None:
        log.info("No CRS found, skipping multiscales metadata", base_path=group.path)
        return None

    # Calculate bounds directly from coordinates for consistency with the data arrays
    if "x" not in first_dataset.coords or "y" not in first_dataset.coords:
        log.error(
            "Missing x/y coordinates in dataset, cannot determine bounds", base_path=group.path
        )
        return None

    x_coords = first_dataset.x.values
    y_coords = first_dataset.y.values
    native_bounds = (
        float(x_coords.min()),
        float(y_coords.min()),
        float(x_coords.max()),
        float(y_coords.max()),
    )

    # Create overview_levels structure following the multiscales v1.0 specification
    overview_levels: list[OverviewLevelJSON] = []
    for res_name in all_resolutions:
        # Use resolution order for consistent scale calculations
        res_meters = res_order[res_name]

        dataset = res_groups[res_name]

        # Defensive guard retained for runtime safety even though the typed
        # contract (Mapping[str, xr.Dataset]) means mypy proves it unreachable.
        if dataset is None:
            continue

        # Get first data variable to extract dimensions
        first_var = next(iter(dataset.data_vars.values()))
        height, width = first_var.shape[-2:]

        transform = _preferred_spatial_transform(dataset)

        # Calculate zoom level (higher resolution = higher zoom)
        tile_width = 256
        zoom_for_width = max(0, int(np.ceil(np.log2(width / tile_width))))
        zoom_for_height = max(0, int(np.ceil(np.log2(height / tile_width))))
        zoom = max(zoom_for_width, zoom_for_height)

        # Calculate relative scale and translation vs parent resolution
        finest_res_meters = res_order[all_resolutions[0]]

        # Fix for issue #114: Translation values should be 0
        relative_translation = 0.0

        # Calculate proper relative scale based on actual parent-child dimension ratios
        if res_name == all_resolutions[0]:  # Base resolution
            relative_scale = 1.0
        else:
            # Define derivation chain to find parent resolution
            derivation_chain = {
                "r10m": None,
                "r20m": "r10m",
                "r60m": "r10m",
                "r120m": "r60m",
                "r360m": "r120m",
                "r720m": "r360m",
            }

            parent_res = derivation_chain.get(res_name)
            if parent_res and parent_res in res_groups:
                # Get actual dimensions of parent and child
                parent_dataset = res_groups[parent_res]
                parent_var = next(iter(parent_dataset.data_vars.values()))
                parent_height, parent_width = parent_var.shape[-2:]

                # Current (child) dimensions
                child_height, child_width = height, width

                # Calculate actual scale ratio based on dimensions
                # Use the larger of the two ratios to be conservative
                scale_x = parent_width / child_width if child_width > 0 else 1.0
                scale_y = parent_height / child_height if child_height > 0 else 1.0
                relative_scale = max(scale_x, scale_y)

                log.info(
                    "Calculated dynamic scale ratio",
                    level=res_name,
                    parent=parent_res,
                    parent_dims=(parent_height, parent_width),
                    child_dims=(child_height, child_width),
                    scale_x=scale_x,
                    scale_y=scale_y,
                    relative_scale=relative_scale,
                )
            else:
                # Fallback to absolute resolution ratio
                relative_scale = res_meters / finest_res_meters
                log.warning(
                    "Using fallback scale calculation",
                    level=res_name,
                    relative_scale=relative_scale,
                )

        # Get chunks in the correct format
        var_chunks = dataset.data_vars[first_var.name].chunks
        if var_chunks is not None:
            chunks = tuple(tuple(int(c) for c in chunk_dim) for chunk_dim in var_chunks)
        else:
            chunks = None
            log.warning(
                "Could not determine chunking information for overview level; 'chunks' will be set to None",
                level=res_name,
                variable=str(first_var.name),
            )

        layout_entry: OverviewLevelJSON = {
            "level": res_name,  # Use string-based level name
            "zoom": zoom,
            "width": width,
            "height": height,
            "translation_relative": relative_translation,
            "scale_absolute": res_meters,
            "scale_relative": relative_scale,
            "spatial_transform": None,
            "chunks": chunks,
            "spatial_shape": (height, width),
        }

        # The minispec requires spatial:transform on every layout entry, so it
        # is kept even when degenerate (e.g. all-zero coordinates).
        if transform is not None:
            layout_entry["spatial_transform"] = transform

        overview_levels.append(layout_entry)

    if len(overview_levels) < 2:
        log.info("    Could not create overview levels for {}", base_path=group.path)
        return None

    layout: list[zcm.ScaleLevel] | MISSING = MISSING

    layout = []

    # Define the correct derivation chain
    derivation_chain = {
        "r10m": None,  # base resolution
        "r20m": "r10m",
        "r60m": "r10m",
        "r120m": "r60m",
        "r360m": "r120m",
        "r720m": "r360m",
    }

    for i, overview_level in enumerate(overview_levels):
        # Create scale level with required fields
        asset = str(overview_level["level"])

        # Build complete dict for ScaleLevel initialization
        scale_level_data: dict[str, Any] = {"asset": asset}

        if i > 0:  # Not the first (base) resolution
            derived_from = derivation_chain.get(asset, str(all_resolutions[0]))
            multiscale_transform = zcm.Transform(
                scale=(overview_level["scale_relative"],) * 2,
                translation=(overview_level["translation_relative"],) * 2,
            )
            scale_level_data["derived_from"] = derived_from
            scale_level_data["transform"] = multiscale_transform

        # Add spatial properties
        assert "spatial_shape" in overview_level  # always populated by the producer above
        scale_level_data["spatial:shape"] = overview_level["spatial_shape"]
        if "spatial_transform" in overview_level:
            spatial_transform = overview_level["spatial_transform"]
            # The minispec requires spatial:transform on every layout entry,
            # so it is written even when degenerate (e.g. all-zero coordinates).
            if spatial_transform is not None:
                scale_level_data["spatial:transform"] = spatial_transform

        scale_level = zcm.ScaleLevel(**scale_level_data)
        layout.append(scale_level)

    # Validate + serialize the multiscales block via the project model (which
    # also covers the ZCM/TMS duality), then hand all three conventions to
    # zarr-cm, which validates each and emits the matching CMOs in order
    # (multiscales, spatial, proj).
    multiscales_data = cast(
        "MultiscalesAttrs",
        zcm.MultiscaleMeta(layout=tuple(layout), resampling_method="average").model_dump(),
    )

    attrs_to_write: dict[str, Any] = {}
    if native_crs and native_bounds:
        attrs_to_write.update(
            utils.build_convention_attrs(
                multiscales=multiscales_data,
                spatial={
                    "spatial:dimensions": ["y", "x"],
                    "spatial:bbox": list(native_bounds),  # [xmin, ymin, xmax, ymax]
                    "spatial:registration": "pixel",
                },
                crs=native_crs,
            )
        )

    # Write attributes directly to the zarr group
    group.attrs.update(attrs_to_write)

    log.info("Added %s multiscale levels to %s", len(overview_levels), group.path)

    return attrs_to_write


def create_downsampled_resolution_group(source_dataset: xr.Dataset, factor: int) -> xr.Dataset:
    """Create a downsampled version of a dataset by given factor."""
    if not source_dataset or len(source_dataset.data_vars) == 0:
        return xr.Dataset()

    # Get reference dimensions
    ref_var = next(iter(source_dataset.data_vars.values()))
    if ref_var.ndim < 2:
        return xr.Dataset()

    current_height, current_width = ref_var.shape[-2:]
    target_height = current_height // factor
    target_width = current_width // factor

    if target_height < 1 or target_width < 1:
        return xr.Dataset()

    # Downsample all variables using existing lazy operations
    lazy_vars = {}
    for var_name, var_data in source_dataset.data_vars.items():
        if var_data.ndim < 2:
            continue
        lazy_vars[var_name] = utils.coarsen_variable(str(var_name), var_data, factor)

    if not lazy_vars:
        return xr.Dataset()

    # Create dataset with lazy variables and coordinates
    return xr.Dataset(lazy_vars, attrs=source_dataset.attrs)


def initialize_crs_from_dataset(dt_input: xr.DataTree) -> CRS | None:
    """
    Initialize CRS from dataset by checking data variables.

    Args:
        dt_input: Input DataTree

    Returns:
        CRS object if found, None otherwise
    """
    # For CPM >= 2.6.0, the EPSG code is stored in root attributes
    epsg_cpm_260 = dt_input.attrs.get("other_metadata", {}).get(
        "horizontal_CRS_code",
        dt_input.attrs.get("other_metadata", {}).get("horizontal_crs_code", None),
    )
    if epsg_cpm_260 is not None:
        try:
            # Handle both integer (32632) and string ("EPSG:32632" or "32632") formats
            if isinstance(epsg_cpm_260, str):
                # Extract numeric part from string like "EPSG:32632" or "32632"
                epsg_code = int(epsg_cpm_260.split(":")[-1])
            else:
                # Already an integer
                epsg_code = int(epsg_cpm_260)
            crs = CRS.from_epsg(epsg_code)
            log.info("Initialized CRS from CPM 2.6.0+ metadata", epsg=epsg_code)
        except Exception as e:
            log.warning(
                "Failed to initialize CRS from CPM 2.6.0+ metadata",
                epsg=epsg_cpm_260,
                error=str(e),
            )
        else:
            return crs

    for group_path in dt_input.groups:
        if group_path == ".":
            continue
        group_node = dt_input[group_path]
        if not hasattr(group_node, "ds") or group_node.ds is None:
            continue
        dataset = group_node.ds

        # Check if dataset has rio accessor with CRS. rioxarray returns a
        # rasterio CRS; convert it to a pyproj CRS (the declared return type),
        # which also validates the value at runtime.
        if hasattr(dataset, "rio"):
            try:
                rio_crs = dataset.rio.crs
                if rio_crs is not None:
                    ds_crs = CRS.from_user_input(rio_crs)
                    log.info("Initialized CRS from dataset", crs=str(ds_crs))
                    return ds_crs
            except Exception:
                log.debug("Failed to get CRS from dataset rio accessor")

        # Check data variables for CRS information
        for var in dataset.data_vars.values():
            if hasattr(var, "rio"):
                try:
                    rio_crs = var.rio.crs
                    if rio_crs is not None:
                        var_crs = CRS.from_user_input(rio_crs)
                        log.info("Initialized CRS from variable", crs=str(var_crs))
                        return var_crs
                except Exception:
                    log.debug("Failed to get CRS from variable rio accessor")

            # Check for proj:epsg attribute
            if "proj:epsg" in var.attrs:
                try:
                    epsg = var.attrs["proj:epsg"]
                    crs = CRS.from_epsg(epsg)
                    log.info("Initialized CRS from EPSG code", epsg=epsg)
                except Exception:
                    log.debug("Failed to initialize CRS from proj:epsg attribute")
                else:
                    return crs

    log.warning("Could not initialize CRS from dataset")
    return None


def reduced_create_multiscale_from_datatree(
    full_band_60m_reference_dataset: xr.Dataset,
    processed_groups: dict[str, Any],
    # spatial_levels: dict[str, dict[str, list[float] | list[int]]],
    *,
    output_group: zarr.Group,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    crs: CRS | None = None,
    keep_scale_offset: bool,
) -> dict[str, dict]:
    """
    Create multiscale versions preserving original structure.
    Keeps all original groups, adds r120m, r360m, r720m downsampled versions.

    Args:
        dt_input: Input DataTree with original structure
        output_path: Base output path
        enable_sharding: Enable Zarr v3 sharding
        spatial_chunk: Spatial chunk size
        crs: Coordinate Reference System for datasets

    Returns:
        Dictionary of processed groups
    """

    # predefined layout asset
    layout_: list[dict[str, Any]] = []
    spatial_levels: dict[str, dict[str, list[float] | list[int]]] = {}

    scale_levels = tuple(pyramid_levels.values())
    base_resolution = 60

    # iterate over pre-defined pyramid-dict (or smth) and generate layout data -> use LayoutObject/...
    current = full_band_60m_reference_dataset
    current_level_name = str

    for src_scale_level, dst_scale_level in pairwise(scale_levels[2:]):
        downsample_factor = dst_scale_level // src_scale_level

        # dest_level_name = f"r{dst_scale_level}m"
        dest_level_saving_name = f"r{dst_scale_level // base_resolution}"
        dest_level_path = f"/multiscales/{dest_level_saving_name}"

        log.info(
            "Creating level with resolution",
            level=dest_level_saving_name,
            resolution=dst_scale_level,
        )

        # Create downsampled dataset
        downsampled_dataset = create_downsampled_resolution_group(current, factor=downsample_factor)

        log.info("Writing level to path", level=dest_level_saving_name, output_path=dest_level_path)

        # Create encoding
        encoding = utils.create_uniform_encoding(
            downsampled_dataset,
            spatial_chunk=spatial_chunk,
            enable_sharding=enable_sharding,
            keep_scale_offset=keep_scale_offset,
        )

        # Strip _FillValue from DataArray encoding for downsampled levels too
        if not keep_scale_offset:
            for data_var in downsampled_dataset.data_vars:
                downsampled_dataset[data_var].encoding.pop("_FillValue", None)

        # add geo metadata
        utils.write_geo_metadata(downsampled_dataset, crs=crs)

        spatial_levels[dest_level_saving_name] = {
            "spatial:shape": downsampled_dataset.attrs["spatial:shape"],
            "spatial:transform": downsampled_dataset.attrs["spatial:transform"],
        }

        # first asset level -> not 'derived_from' and 'transform' set
        if not layout_:
            lo = {
                "asset": dest_level_saving_name,
                **spatial_levels[dest_level_saving_name],
            }
        # already have a reference -> derive scale factor from it
        else:
            transform: Transform = {
                "scale": [downsample_factor, downsample_factor],
                "translation": [0.0, 0.0],
            }
            lo = {
                "asset": dest_level_saving_name,
                "derived_from": current_level_name,
                "transform": transform,
                **spatial_levels[dest_level_saving_name],
            }

        layout_.append(lo)

        # Write dataset
        ds_out = utils.stream_write_dataset(
            downsampled_dataset,
            path=dest_level_path,
            group=output_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
        )

        # Store results
        processed_groups[dest_level_path] = ds_out

        current = downsampled_dataset
        current_level_name = dest_level_saving_name

    # add metadata to root and multiscale-parent node
    root_rw = zarr.open_group(output_path, mode="a")

    # create layoutobjects -> also checks accordance (i think)
    layout: list[LayoutObject] = [LayoutObject(**lo) for lo in layout_]
    ms: MultiscalesAttrs = {"layout": layout, "resampling_method": "average"}

    # add geozarr attrs to base of /multiscales
    base_measurement_ds_120m = processed_groups["/multiscales/r2"]
    base_spatial = utils.grid_spatial_attrs(
        transform=base_measurement_ds_120m.rio.transform(recalc=True),
        shape=(base_measurement_ds_120m.sizes["y"], base_measurement_ds_120m.sizes["x"]),
    )

    conv = utils.build_convention_attrs(multiscales=ms, spatial=base_spatial, crs=crs)
    root_rw["/multiscales"].attrs.update(cast("dict[str, JSON]", conv))

    # add it as none here to be recognized later and can be created as a zarr root with necessary metadata
    processed_groups["/multiscales"] = None

    return processed_groups


def convert_s2_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    compression_level: int,
    keep_scale_offset: bool,
) -> xr.DataTree:
    """
    Convenience function for S2 optimization.

    Args:
        dt_input: Input Sentinel-2 DataTree
        output_path: Output path
        enable_sharding: Enable Zarr v3 sharding
        spatial_chunk: Spatial chunk size
        compression_level: Compression level 1-9
        validate_output: Whether to validate the output
        keep_scale_offset: Whether to preserve scale-offset encoding of the source data.
        max_retries: Maximum number of retries for network operations

    Returns:
        Optimized DataTree
    """

    start_time = time.time()

    log.info(
        "Starting S2 optimized conversion",
        num_groups=len(dt_input.groups),
        output_path=output_path,
    )

    # Initialize CRS from dataset
    crs = initialize_crs_from_dataset(dt_input)
    output_group = zarr.open_group(output_path)
    processed_groups: dict[str, Any] = {}
    full_band_60m_reference_dataset: xr.Dataset = xr.Dataset()

    # helper dicts to keep track of attrs for multiscales
    # spatial_levels: dict[str, dict[str, list[float] | list[int]]] = {}

    # cheap determination if its L2A or L1C
    filename = dt_input.name
    s2_type = sentinel_modes.S2Type.from_filename(filename)

    if s2_type is None:
        log.info("not found a matching s2_type {}: is not matching L2A or L1C", s2_type=s2_type)

    # Step 1: Copy all original groups as-is
    for group_path in dt_input.groups:
        if group_path == ".":
            continue

        # Skip the quicklook groups (`/quality/l1c_quicklook`, `/quality/l2a_quicklook`).
        # They duplicate RGB data that downstream consumers can derive from the
        # reflectance bands, so dropping them shrinks the output store and
        # speeds up conversion. See EOPF-Explorer/data-model#81.
        if group_path.startswith(("/quality/l1c_quicklook", "/quality/l2a_quicklook")):
            log.info("Skipping quicklook group", group_path=group_path)
            continue

        group_node = dt_input[group_path]

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

        # Determine if this is a measurement-related resolution group
        group_name = group_path.split("/")[-1]
        is_measurement_group = (
            group_name.startswith("r")
            and group_name.endswith("m")
            and "/measurements/" in group_path
        )

        if is_measurement_group:
            # Inject bands whose native resolution is finer than this group's
            # (e.g. b08 native at 10m into r60m fir L2A, all other ones for L1c) so they propagate through
            # the full overview chain (r120m … r720m).
            if group_path.startswith("/measurements/reflectance/"):
                try:
                    group_resolution = int(group_name[1:-1])
                except ValueError:
                    group_resolution = 0

                # just add the b08 band to 60m reference dataset
                if s2_type == "L2A" and group_resolution == 60:
                    full_band_60m_reference_dataset = inject_missing_bands(
                        dataset,
                        dt_input,
                        group_resolution,
                        spatial_chunk,
                        bands={"b08"},
                    )

                # add all lower level bands here!
                elif s2_type == "L1C" and group_resolution == 60:
                    full_band_60m_reference_dataset = inject_missing_bands(
                        dataset,
                        dt_input,
                        group_resolution,
                        spatial_chunk,
                        bands={
                            "b02",
                            "b03",
                            "b04",
                            "b08",
                            "b05",
                            "b06",
                            "b07",
                            "b8a",
                            "b11",
                            "b12",
                        },
                    )

            # Measurement groups: apply custom encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
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

            # Drop scalar (0-D) coordinates such as a source `band` label: the
            # minispec's DataArray rules forbid scalar arrays in a GeoZarr dataset.
            scalar_coords = [name for name, coord in dataset.coords.items() if coord.ndim == 0]
            if scalar_coords:
                dataset = dataset.drop_vars(scalar_coords)
                for name in scalar_coords:
                    encoding.pop(str(name), None)

            utils.write_geo_metadata(dataset, crs=crs)

            # # add spatial: metadta to outside dict for multuiscale layouts
            # spatial_levels[group_name] = {
            #     "spatial:shape": dataset.attrs["spatial:shape"],
            #     "spatial:transform": dataset.attrs["spatial:transform"],
            # }

            ds_out = utils.stream_write_dataset(
                dataset,
                path=group_path,
                group=output_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                # crs=crs,
            )
            processed_groups[group_path] = ds_out

        else:
            # Non-measurement groups: preserve original encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
                enable_sharding=enable_sharding,
                keep_scale_offset=keep_scale_offset,
                compression_level=compression_level,
            )

            # Drop scalar (0-D) coordinates such as a source `band` label: the
            # minispec's DataArray rules forbid scalar arrays in a GeoZarr dataset.
            scalar_coords = [name for name, coord in dataset.coords.items() if coord.ndim == 0]
            if scalar_coords:
                dataset = dataset.drop_vars(scalar_coords)
                for name in scalar_coords:
                    encoding.pop(str(name), None)

            if "/quality/" in group_path or "/conditions/mask" in group_path:
                utils.write_geo_metadata(dataset, crs=crs)

            ds_out = utils.stream_write_dataset(
                dataset,
                path=group_path,
                group=output_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                # crs=crs,
            )
            processed_groups[group_path] = ds_out

    processed_groups["/measurements/reflectance"] = None
    # Step 2: Multiscale calculation
    log.info("Step 2: Multiscale calculation")
    reduced_create_multiscale_from_datatree(
        full_band_60m_reference_dataset=full_band_60m_reference_dataset,
        processed_groups=processed_groups,
        # spatial_levels=spatial_levels,
        output_group=output_group,
        output_path=output_path,
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
        crs=crs,
        keep_scale_offset=keep_scale_offset,
    )
    # log.info("Created multiscale pyramids", num_groups=len(datasets))

    # Step 3: Root-level consolidation
    log.info("Step 3: Final root-level metadata consolidation")
    # utils.simple_root_consolidation(dt_input, output_path, datasets)
    utils.updated_root_consolidation(dt_input, output_path, processed_groups)

    # Create result DataTree
    result_dt = utils.create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    utils.optimization_summary(dt_input, result_dt, output_path)

    return result_dt
