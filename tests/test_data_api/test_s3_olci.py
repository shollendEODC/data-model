"""Tests for the Sentinel-3 OLCI data-api model."""

import pytest
from pydantic import ValidationError

from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec  # noqa: F401


def _olci_arr() -> dict[str, object]:
    """Return a minimal v2 ArraySpec-shaped dict for a 2-D uint16 array."""
    return ArraySpec(
        shape=(4, 5),
        chunks=(4, 5),
        dtype="<u2",
        fill_value=0,
        attributes={"_ARRAY_DIMENSIONS": ["rows", "columns"]},
    ).model_dump()


def test_validates_minimal_olci_product() -> None:
    """A dict with 21 radiance bands + geolocation validates as Sentinel3OlciRoot."""
    radiance = {f"oa{i:02d}_radiance": _olci_arr() for i in range(1, 22)}
    coords = {c: _olci_arr() for c in ("latitude", "longitude", "altitude")}
    root = {
        "zarr_format": 2,
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {
            "measurements": {
                "zarr_format": 2,
                "attributes": {},
                "members": {**radiance, **coords},
            },
        },
    }
    model = Sentinel3OlciRoot.model_validate(root)
    assert "oa01_radiance" in model.measurements.members


def test_rejects_non_olci_product() -> None:
    """A dict with an unexpected key in measurements is rejected by closed=True."""
    not_olci = {
        "zarr_format": 2,
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {
            "measurements": {
                "zarr_format": 2,
                "attributes": {},
                "members": {
                    "reflectance": {
                        "zarr_format": 2,
                        "attributes": {},
                        "members": {},
                    }
                },
            }
        },
    }
    with pytest.raises(ValidationError):
        Sentinel3OlciRoot.model_validate(not_olci)
