"""Top-level Sentinel-3 OLCI L1 EFR -> GeoZarr conversion."""

from __future__ import annotations

import structlog
import xarray as xr
import zarr

from eopf_geozarr.conversion.utils import build_convention_attrs
from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
from eopf_geozarr.s3_olci_optimization.olci_multiscale import decimate_swath, swath_spatial_attrs

log = structlog.get_logger()


def is_sentinel3_olci_dataset(group: zarr.Group) -> bool:
    """Return True if *group* is a Sentinel-3 OLCI L1 EFR product.

    Detection is structural: the group must validate against
    ``Sentinel3OlciRoot`` and its ``measurements`` group must contain the
    first OLCI radiance band.
    """
    from eopf_geozarr.pyz.v2 import GroupSpec

    try:
        model = Sentinel3OlciRoot.model_validate(GroupSpec.from_zarr(group).model_dump())
    except ValueError as e:
        log.debug("Not an OLCI dataset", error=str(e))
        return False
    try:
        return "oa01_radiance" in model.measurements.members
    except KeyError:
        return False


def _overview_levels(rows: int, cols: int, min_dimension: int) -> int:
    """Return the number of /2 decimations before min(rows, cols) drops below min_dimension.

    A level is generated only when the *post*-decimation minimum spatial
    dimension is at least *min_dimension*.  For example, a 512x480 dataset
    with min_dimension=256 yields zero levels because 480//2=240 < 256, while
    a 1024x1024 dataset with min_dimension=256 yields two levels (512x512,
    then 256x256).
    """
    levels = 0
    r, c = rows, cols
    while min(r, c) // 2 >= min_dimension:
        r, c = r // 2, c // 2
        levels += 1
    return levels


def _copy_subtree(node: xr.DataTree, output_path: str, *, root_group: str) -> None:
    """Write every Dataset in *node*'s subtree to the Zarr store at *output_path*.

    Each path is mapped relative to the node's own path: the node root becomes
    *root_group*, and child nodes are placed at ``root_group/<relative_path>``.

    ``DataTree.to_zarr`` does not yet support a ``group`` keyword for
    specifying a root offset, so we write each leaf's :py:meth:`~xarray.DataTree.to_dataset`
    using ``xr.Dataset.to_zarr`` instead.
    """
    node_path = node.path  # e.g. "/conditions"
    for child in node.subtree:
        ds = child.to_dataset()
        if not ds.data_vars and not ds.coords:
            continue
        # Build the group path: strip the ancestor prefix and prepend root_group.
        relative = child.path[len(node_path) :]  # "" for root, "/sub" for children
        group_path = root_group + relative
        log.info("Copying ancillary subgroup", group=group_path)
        ds.to_zarr(
            output_path,
            group=group_path,
            mode="a",
            consolidated=False,
            zarr_format=3,
        )


def convert_olci_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool = False,
    spatial_chunk: int = 1024,
    compression_level: int = 3,
    min_dimension: int = 256,
    keep_scale_offset: bool = False,
) -> xr.DataTree:
    """Convert an EOPF OLCI L1 EFR DataTree to a GeoZarr multiscale store.

    Writes the native-resolution ``measurements`` group with GeoZarr
    convention metadata, then writes /2-decimated overview subgroups
    (``r2``, ``r4``, …) down to *min_dimension*.  Any ``conditions`` or
    ``quality`` groups present in *dt_input* are copied through unchanged.

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

    Returns
    -------
    xr.DataTree
        The opened output DataTree (lazy; backed by the written Zarr store).
        Overview subgroups (``r2``, ``r4``, …) are written to the Zarr store
        but are **not** represented as children of the returned DataTree,
        because xarray enforces dimension consistency between parent and child
        nodes and the overview subgroups have smaller spatial dimensions than
        the parent ``measurements`` group.  To read them, open the store
        directly with ``zarr.open_group(output_path)["measurements"]["r2"]``
        etc.

    Notes
    -----
    Parameters ``enable_sharding``, ``spatial_chunk``, ``compression_level``,
    and ``keep_scale_offset`` are accepted but not yet applied to the on-disk
    encoding.  Wiring them through the existing ``conversion`` helpers
    (``create_measurements_encoding``, sharding codec, etc.) is left for a
    follow-up task so as not to block the integration test.
    """
    measurements = dt_input["/measurements"].to_dataset()

    # Attach GeoZarr spatial convention metadata for native-resolution swath.
    conv = build_convention_attrs(spatial=swath_spatial_attrs(), crs=None)
    measurements.attrs.update(dict(conv))

    log.info("Writing native-resolution measurements", shape=dict(measurements.sizes))
    measurements.to_zarr(
        output_path,
        group="measurements",
        mode="w",
        consolidated=False,
        zarr_format=3,
    )

    # Write /2 decimated overview subgroups: r2, r4, r8, …
    rows = measurements.sizes["rows"]
    cols = measurements.sizes["columns"]
    n_levels = _overview_levels(rows, cols, min_dimension)
    log.info("Generating overview levels", n_levels=n_levels)

    current = measurements
    for level in range(1, n_levels + 1):
        current = decimate_swath(current, factor=2)
        group_name = f"r{2**level}"
        log.info("Writing overview", group=f"measurements/{group_name}", shape=dict(current.sizes))
        current.to_zarr(
            output_path,
            group=f"measurements/{group_name}",
            mode="a",
            consolidated=False,
            zarr_format=3,
        )

    # Copy conditions/quality through unchanged (if present).
    # DataTree.to_zarr does not support a root ``group`` argument, so we
    # iterate the subtree and write each leaf Dataset individually.
    for grp in ("conditions", "quality"):
        try:
            node_item = dt_input[f"/{grp}"]
        except KeyError:
            continue
        if not isinstance(node_item, xr.DataTree):
            continue
        log.info("Copying ancillary group", group=grp)
        _copy_subtree(node_item, output_path, root_group=grp)

    # xarray DataTree enforces dimension consistency between parent and child
    # nodes, so opening the whole store via ``xr.open_datatree`` would fail
    # because the overview subgroups have smaller spatial dimensions than
    # the parent ``measurements`` group.  Instead, we build the DataTree
    # manually from the top-level groups only: overview levels (r2, r4, …)
    # are in the zarr store and accessible via ``zarr.open_group``, but are
    # intentionally not exposed as DataTree children.
    root = zarr.open_group(output_path, mode="r")
    tree_dict: dict[str, xr.Dataset] = {}
    for key in root.group_keys():
        child = root[key]
        if isinstance(child, zarr.Group) and list(child.array_keys()):
            tree_dict[f"/{key}"] = xr.open_dataset(
                output_path,
                engine="zarr",
                group=key,
                chunks={},
                consolidated=False,
            )
    return xr.DataTree.from_dict(tree_dict)
