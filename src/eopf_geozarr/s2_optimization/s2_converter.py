"""
Main S2 optimization converter.
"""

from __future__ import annotations

import time

import structlog
import xarray as xr
import zarr
from pyproj import CRS

from eopf_geozarr.conversion import utils
from eopf_geozarr.conversion.fs_utils import get_storage_options

# from eopf_geozarr.conversion.geozarr import get_zarr_group
from .s2_multiscale import create_multiscale_from_datatree

log = structlog.get_logger()


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


def convert_s2_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    compression_level: int,
    validate_output: bool,
    keep_scale_offset: bool,
    max_retries: int = 3,
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
    # removed during refactoring -> validate new
    # _validate_s2_input(dt_input)

    # Initialize CRS from dataset
    crs = initialize_crs_from_dataset(dt_input)

    # Step 1: Process data while preserving original structure
    log.info("Step 1: Processing data with original structure preserved")

    # Step 2: Create multiscale pyramids for each group in the original structure
    log.info("Step 2: Creating multiscale pyramids (preserving original hierarchy)")

    output_group = zarr.open_group(output_path)

    datasets = create_multiscale_from_datatree(
        dt_input,
        output_group=output_group,
        output_path=output_path,
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
        crs=crs,
        keep_scale_offset=keep_scale_offset,
        compression_level=compression_level,
    )

    log.info("Created multiscale pyramids", num_groups=len(datasets))

    # Step 3: Root-level consolidation
    log.info("Step 3: Final root-level metadata consolidation")
    # utils.simple_root_consolidation(dt_input, output_path, datasets)
    utils.updated_root_consolidation(dt_input, output_path, datasets)

    # Step 4: Validation

    # removed during refacto -> requires a redoing of output validation
    # if validate_output:
    #     log.info("Step 4: Validating optimized dataset")
    #     validation_results = validate_optimized_dataset(output_path)
    #     if not validation_results["is_valid"]:
    #         log.warning("Validation issues found", issues=validation_results["issues"])

    # Create result DataTree
    result_dt = create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    optimization_summary(dt_input, result_dt, output_path)

    return result_dt


def optimization_summary(dt_input: xr.DataTree, dt_output: xr.DataTree, output_path: str) -> None:
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


def create_result_datatree(output_path: str) -> xr.DataTree:
    """Create result DataTree from written output."""
    storage_options = get_storage_options(output_path)
    return xr.open_datatree(
        output_path,
        engine="zarr",
        chunks={},
        mask_and_scale=False,
        storage_options=storage_options,
    )
