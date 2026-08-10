"""Pipeline routing for DataTree products handed to the CPM writer plugin.

CPM products carry their identity in the ``stac_discovery`` root attributes
(see ``StacProductConvention`` in eopf-cpm), so routing prefers the declared
``product:type`` and only falls back to structural inspection when the
attribute is absent. Routing must work on any ``xarray.DataTree`` regardless
of backend: trees produced by the CPM SAFE reader are built in memory and
have no backing zarr store to introspect.

This module deliberately has no dependency on the ``eopf`` package so it can
be imported and tested without eopf-cpm installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import xarray as xr

PipelineName = Literal["s2-optimized", "generic"]

#: CPM product types routed to the Sentinel-2 optimized pipeline
#: (e.g. ``S02MSIL1C``, ``S02MSIL2A``).
S2_PRODUCT_TYPE_PREFIX = "S02MSI"

#: Native Sentinel-2 resolution groups expected under measurements/reflectance.
_S2_NATIVE_RESOLUTIONS = frozenset({"r10m", "r20m", "r60m"})


def product_type_of(dtree: xr.DataTree) -> str | None:
    """
    Return the CPM product type declared in the root STAC attributes, if any.

    Parameters
    ----------
    dtree : xr.DataTree
        Product root node.

    Returns
    -------
    str | None
        The value of ``stac_discovery.properties["product:type"]`` (e.g.
        ``"S02MSIL1C"``), or None when the attribute chain is absent.
    """
    stac = dtree.attrs.get("stac_discovery")
    if not isinstance(stac, Mapping):
        return None
    properties = stac.get("properties")
    if not isinstance(properties, Mapping):
        return None
    product_type = properties.get("product:type")
    if isinstance(product_type, str):
        return product_type
    return None


def looks_like_sentinel2(dtree: xr.DataTree) -> bool:
    """
    Report whether a DataTree looks like a Sentinel-2 L1C/L2A product.

    The declared CPM product type wins when present; otherwise the tree is
    inspected for the native ``measurements/reflectance/r{10,20,60}m`` groups.

    Parameters
    ----------
    dtree : xr.DataTree
        Product root node.

    Returns
    -------
    bool
    """
    product_type = product_type_of(dtree)
    if product_type is not None:
        return product_type.startswith(S2_PRODUCT_TYPE_PREFIX)

    measurements = dtree.children.get("measurements")
    if measurements is None:
        return False
    reflectance = measurements.children.get("reflectance")
    if reflectance is None:
        return False
    return set(reflectance.children) >= _S2_NATIVE_RESOLUTIONS


def select_pipeline(
    dtree: xr.DataTree,
    *,
    force_s2_optimized: bool | None = None,
) -> PipelineName:
    """
    Select the conversion pipeline for a product.

    Parameters
    ----------
    dtree : xr.DataTree
        Product root node.
    force_s2_optimized : bool, optional
        True forces the Sentinel-2 optimized pipeline, False forces the
        generic pipeline, and None (default) auto-detects.

    Returns
    -------
    PipelineName
        ``"s2-optimized"`` or ``"generic"``.
    """
    if force_s2_optimized is True:
        return "s2-optimized"
    if force_s2_optimized is False:
        return "generic"
    if looks_like_sentinel2(dtree):
        return "s2-optimized"
    return "generic"
