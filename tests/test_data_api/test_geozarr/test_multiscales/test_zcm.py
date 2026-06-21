import jsondiff
import pytest
from pydantic import ValidationError
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.geozarr.multiscales.zcm import (
    MultiscalesAttrs,
    ScaleLevel,
    ScaleLevelJSON,
)


def test_multiscales_rt(zcm_multiscales_example: dict[str, object]) -> None:
    """
    Test that the multiscales metadata round-trips input JSON
    """
    value = zcm_multiscales_example
    value_tup = tuplify_json(value)
    attrs_json = value_tup["attributes"]
    model = MultiscalesAttrs(**attrs_json)
    observed = model.model_dump()
    expected = attrs_json
    assert jsondiff.diff(observed, expected) == {}


def test_scale_level_from_group() -> None:
    """
    Test that the ScaleLevel metadata rejects a dict with
    from_group but no "scale" attribute
    """
    meta = {"group": "1", "from_group": "0"}
    with pytest.raises(ValidationError):
        ScaleLevel.model_validate(meta)


def test_scalelevel_json() -> None:
    """
    Test that we can create a ScaleLevel from its JSON representation
    """
    x: ScaleLevelJSON = {
        "asset": "example_asset",
        "derived_from": "example_derived",
        "transform": {
            "scale": (1.0, 1.0),
            "translation": (0.0, 0.0),
        },
        "resampling_method": "nearest",
    }
    assert ScaleLevel.model_validate(x).model_dump() == x
