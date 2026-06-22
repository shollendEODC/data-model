"""Top-level Sentinel-3 OLCI L1 EFR -> GeoZarr conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot

if TYPE_CHECKING:
    import zarr

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
