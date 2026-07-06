"""Top-level Sentinel-3 OLCI L1 EFR -> GeoZarr conversion."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import structlog
import xarray as xr
import zarr

from eopf_geozarr.conversion.utils import build_convention_attrs
from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
from eopf_geozarr.s3_olci_optimization.olci_band_mapping import OLCI_BANDS
from eopf_geozarr.s3_olci_optimization.olci_multiscale import (
    reduce_swath,
    swath_spatial_attrs,
)

if TYPE_CHECKING:
    from zarr.core.common import JSON
    from zarr_cm import LayoutObject, MultiscalesAttrs, Transform

log = structlog.get_logger()


def _sanitize_olci_array_attrs(attrs: dict[str, object]) -> dict[str, object]:
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
        return OLCI_BANDS[0] in model.measurements.members
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


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Return *ds* with all inherited source encoding cleared.

    When the input DataTree was opened from a Zarr v2 store, xarray carries
    ``numcodecs.Blosc`` compressors (and potentially scale-offset filters) in
    each variable's ``.encoding``.  Passing that encoding to
    ``Dataset.to_zarr(zarr_format=3)`` raises::

        TypeError: Expected a BytesBytesCodec. Got <class 'numcodecs.blosc.Blosc'>

    because numcodecs codecs are not valid Zarr v3 BytesBytesCodecs.  Clearing
    the encoding lets the Zarr v3 writer choose its own default codecs.

    This converter expects raw (non-mask-scaled) input: the caller must open
    the source DataTree with ``mask_and_scale=False`` so that CF
    ``scale_factor``/``add_offset`` stay in ``.attrs`` and integer fill pixels
    are identified via ``attrs["_FillValue"]``.  Only Zarr v2 *codec* encoding
    (e.g. ``numcodecs.Blosc`` compressors) is stripped here — CF metadata is
    untouched.
    """
    ds = ds.copy()
    ds.encoding = {}
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].encoding.clear()
    return ds


def _sanitize_data_vars(ds: xr.Dataset) -> xr.Dataset:
    """Return *ds* with stale source attrs stripped from all data variables.

    Applies :func:`_sanitize_olci_array_attrs` to every data variable in *ds*.
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
        new_var.attrs = _sanitize_olci_array_attrs(dict(var.attrs))
        new_vars[str(name)] = new_var
    return ds.assign(new_vars)


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
        # Strip any inherited Zarr v2 encoding before writing to a v3 store.
        ds = _clear_encoding(ds)
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
    convention metadata, then writes /2-reduced overview subgroups
    (``r2``, ``r4``, …) down to *min_dimension*.  Any ``conditions`` or
    ``quality`` groups present in *dt_input* are copied through unchanged,
    along with any child subgroups of ``measurements`` (e.g. ``orphans``).

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
        Every written group that holds arrays is included — including nested
        ancillary groups such as ``conditions/geometry`` — except the overview
        subgroups (``r2``, ``r4``, …), which are written to the Zarr store
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
    # Strip any inherited Zarr v2 encoding (e.g. numcodecs.Blosc compressors)
    # so the v3 writer can choose its own default codecs without raising a
    # "Expected a BytesBytesCodec" error.  The caller is expected to have opened
    # the source with mask_and_scale=False, so CF attrs (scale_factor,
    # add_offset, _FillValue) live in .attrs and are preserved here.
    measurements = _clear_encoding(measurements)
    # Sanitize radiance variable attrs: strip source-only / misleading attrs
    # (_eopf_attrs, dtype, valid_min, valid_max) while keeping CF scale/offset
    # and _FillValue so that downstream readers and reduce_swath can work
    # correctly with raw integer data.
    measurements = _sanitize_data_vars(measurements)

    log.info("Writing native-resolution measurements", shape=dict(measurements.sizes))
    measurements.to_zarr(
        output_path,
        group="measurements",
        mode="w",
        consolidated=False,
        zarr_format=3,
    )

    # Write /2 reduced overview subgroups: r2, r4, r8, …
    rows = measurements.sizes["rows"]
    cols = measurements.sizes["columns"]
    n_levels = _overview_levels(rows, cols, min_dimension)
    log.info("Generating overview levels", n_levels=n_levels)

    current = measurements
    for level in range(1, n_levels + 1):
        current = reduce_swath(current, factor=2)
        current = _clear_encoding(current)
        # Attrs already sanitized at native level and passed through by
        # reduce_swath; no second sanitize pass needed.
        group_name = f"r{2**level}"
        log.info("Writing overview", group=f"measurements/{group_name}", shape=dict(current.sizes))
        current.to_zarr(
            output_path,
            group=f"measurements/{group_name}",
            mode="a",
            consolidated=False,
            zarr_format=3,
        )

    # Build and attach GeoZarr convention metadata (spatial + multiscales CMO)
    # to the measurements group attrs.
    layout: list[LayoutObject] = [{"asset": "."}]
    for lvl in range(1, n_levels + 1):
        transform: Transform = {"scale": [2.0, 2.0], "translation": [0.0, 0.0]}
        lo: LayoutObject = {
            "asset": f"r{2**lvl}",
            "derived_from": "." if lvl == 1 else f"r{2 ** (lvl - 1)}",
            "transform": transform,
            "resampling_method": "average",
        }
        layout.append(lo)

    if n_levels > 0:
        ms: MultiscalesAttrs = {"layout": layout, "resampling_method": "average"}
        conv = build_convention_attrs(multiscales=ms, spatial=swath_spatial_attrs(), crs=None)
    else:
        conv = build_convention_attrs(spatial=swath_spatial_attrs(), crs=None)

    zarr.open_group(output_path, mode="a")["measurements"].attrs.update(
        cast("dict[str, JSON]", conv)
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

    # Copy any child subgroups of measurements (e.g. orphans) through unchanged.
    try:
        meas_node = dt_input["/measurements"]
    except KeyError:
        meas_node = None
    if isinstance(meas_node, xr.DataTree):
        for child in meas_node.children.values():
            log.info("Copying measurements subgroup", group=f"measurements/{child.name}")
            _copy_subtree(child, output_path, root_group=f"measurements/{child.name}")

    # xarray DataTree enforces dimension consistency between parent and child
    # nodes, so opening the whole store via ``xr.open_datatree`` would fail
    # because the overview subgroups have smaller spatial dimensions than
    # the parent ``measurements`` group.  Instead, we build the DataTree
    # manually from every group that holds arrays — including nested ancillary
    # groups such as ``conditions/geometry`` — skipping the overview levels
    # (``measurements/r2``, ``r4``, …): those are in the zarr store and
    # accessible via ``zarr.open_group``, but are intentionally not exposed
    # as DataTree children.
    root = zarr.open_group(output_path, mode="r")
    tree_dict: dict[str, xr.Dataset] = {}
    for group_path, node in root.members(max_depth=None):
        if not isinstance(node, zarr.Group) or not list(node.array_keys()):
            continue
        if re.fullmatch(r"measurements/r\d+", group_path):
            continue
        tree_dict[f"/{group_path}"] = xr.open_dataset(
            output_path,
            engine="zarr",
            group=group_path,
            chunks={},
            consolidated=False,
        )
    return xr.DataTree.from_dict(tree_dict)
