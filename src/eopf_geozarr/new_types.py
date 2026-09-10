"""Types and constants for the GeoZarr data API."""

from __future__ import annotations

from typing import Final, NotRequired, TypedDict

CF_SCALE_OFFSET_KEYS: Final[set[str]] = {"scale_factor", "add_offset", "dtype"}

XARRAY_ENCODING_KEYS: Final[set[str]] = {
    "chunks",
    "preferred_chunks",
    "compressors",
    "filters",
    "shards",
    "_FillValue",
    "fill_value",
} | CF_SCALE_OFFSET_KEYS


class XarrayDataArrayEncoding(TypedDict):
    """
    The dict form of the encoding for xarray.DataArray
    """

    chunks: NotRequired[tuple[int, ...]]
    preferred_chunks: NotRequired[tuple[int, ...]]
    compressors: NotRequired[tuple[object, ...] | None]
    filters: NotRequired[tuple[object, ...]]
    shards: NotRequired[tuple[int, ...] | None]
    _FillValue: NotRequired[object]
    fill_value: NotRequired[object]
    scale_factor: NotRequired[float]
    add_offset: NotRequired[float]
    dtype: NotRequired[object]


# Why is endpoint URL specified twice?
class S3ClientOptions(TypedDict):
    """
    S3 client options
    """

    region_name: NotRequired[str]
    endpoint_url: NotRequired[str]


class S3FsOptions(TypedDict):
    """
    S3FS options
    """

    anon: NotRequired[bool]
    use_ssl: NotRequired[bool]
    client_kwargs: NotRequired[S3ClientOptions]
    endpoint_url: NotRequired[str]
    asynchronous: NotRequired[bool]


class S3Credentials(TypedDict):
    """
    S3 credentials
    """

    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_session_token: str | None
    aws_default_region: str
    aws_profile: str | None
    AWS_ENDPOINT_URL: str | None


class OverviewLevelJSON(TypedDict):
    level: int | str
    width: int
    height: int
    translation_relative: float
    scale_relative: int | float
    zoom: NotRequired[int]
    scale_absolute: NotRequired[int | float]
    spatial_transform: NotRequired[tuple[float, ...] | None]
    spatial_shape: NotRequired[tuple[int, ...]]
    chunks: NotRequired[tuple[tuple[int, ...], ...] | None]
