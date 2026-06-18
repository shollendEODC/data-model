"""STAC item builder for S1 GRD RTC Zarr V3 stores."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pyproj
import pystac
import zarr

SAR_EXT = "https://stac-extensions.github.io/sar/v1.0.0/schema.json"
SAT_EXT = "https://stac-extensions.github.io/sat/v1.0.0/schema.json"
PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
RENDER_EXT = "https://stac-extensions.github.io/render/v1.0.0/schema.json"
DATACUBE_EXT = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"
TIMESTAMPS_EXT = "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json"

ZARR_MEDIA_TYPE = "application/vnd.zarr; version=3"

_ORBIT_PREFERENCE = ("ascending", "descending")
# Short suffix for orbit-keyed asset names (gamma0-rtc-backscatter-asc / -desc).
_ORBIT_SHORT = {"ascending": "asc", "descending": "desc"}

# γ⁰ RTC backscatter is float32 with a NaN fill at every resolution level; the arrays carry no attrs
# in the store so these product invariants are hardcoded (see the store hierarchy in data_api/s1_rtc).
GAMMA0_DTYPE = "float32"
GAMMA0_NODATA = "nan"
GAMMA0_UNIT = "gamma0 (linear power)"
BORDER_MASK_DTYPE = "uint8"
GSD = 10


def _rgb_render(orbit: str) -> dict[str, object]:
    """Build the dual-pol RGB composite render config for the given orbit group.

    Produces a 3-band false-colour composite (R=VV, G=VH, B=VV/VH ratio) that
    titiler renders into previews/tiles. ``bidx=[1]`` selects the single time
    slice from each multi-band variable; the single ``rescale`` pair (linear
    gamma0 units) is applied to every band.
    """
    vv = f"/{orbit}:vv"
    vh = f"/{orbit}:vh"
    return {
        "title": "VV, VH, VV/VH composite",
        "expression": f"{vv};{vh};({vv})/({vh})",
        # Linear gamma0 stretch tuned for the S1 RTC product (the 0.0,0.1 default was too bright).
        "rescale": [[0.0, 0.2]],
        "bidx": [1],
        "tilesize": 256,
    }


def _gamma0_bands() -> list[dict[str, object]]:
    """STAC 1.1 band objects for the two polarisations carried by a γ⁰ RTC asset."""
    return [
        {
            "name": pol,
            "description": f"γ⁰ RTC backscatter, {pol.upper()} polarization",
            "data_type": GAMMA0_DTYPE,
            "nodata": GAMMA0_NODATA,
            "unit": GAMMA0_UNIT,
        }
        for pol in ("vv", "vh")
    ]


def _utm_to_wgs84(proj_code: str, utm_bbox: list[float]) -> tuple[float, float, float, float]:
    """Convert UTM [xmin, ymin, xmax, ymax] to WGS84 (west, south, east, north)."""
    xmin, ymin, xmax, ymax = utm_bbox
    transformer = pyproj.Transformer.from_crs(proj_code, "EPSG:4326", always_xy=True)
    xs = [xmin, xmax, xmin, xmax]
    ys = [ymin, ymin, ymax, ymax]
    lons, lats = transformer.transform(xs, ys)
    return min(lons), min(lats), max(lons), max(lats)


def _open_root(zarr_store: str) -> zarr.Group:
    """Open the cube root, preferring consolidated metadata.

    A cube grown by appending a time-slice to an *existing same-orbit* group can end up without root
    consolidated metadata (re-consolidating an append on the S3 store is unreliable). The builder must
    not require it — fall back to reading the hierarchy directly, exactly as titiler does. See the
    data-model issue on the S1 RTC consolidated-metadata regression.
    """
    try:
        return zarr.open_consolidated(zarr_store, zarr_format=3)
    except ValueError as exc:
        if "consolidated metadata" not in str(exc).lower():
            raise
        return zarr.open_group(zarr_store, mode="r", zarr_format=3)


def build_s1_rtc_stac_item(zarr_store: str, collection_id: str) -> pystac.Item:
    """Build a STAC item from a consolidated S1 GRD RTC Zarr V3 store.

    Parameters
    ----------
    zarr_store:
        Local path or ``s3://`` URI to the Zarr store.
    collection_id:
        STAC collection ID to attach to the item.

    Returns
    -------
    pystac.Item

    Raises
    ------
    ValueError
        If the store contains no acquisitions.
    """
    # TEMPORARY (#246): the store is written as s1-rtc-{tile}.zarr so its filename equals
    # the item id, which titiler-eopf reconstructs as the render path (it ignores the asset
    # href). Revert this prefix to "s1-grd-rtc-" when titiler-eopf#108 lands.
    tile_id = Path(zarr_store).name.removeprefix("s1-rtc-").removesuffix(".zarr")

    root = _open_root(zarr_store)

    all_times_ns: list[int] = []
    wgs84_bboxes: list[tuple[float, float, float, float]] = []
    # Per present orbit, in preference order: the metadata needed for assets, projection and datacube.
    present: list[dict[str, object]] = []

    for orbit_dir in _ORBIT_PREFERENCE:
        if orbit_dir not in root:
            continue
        og = cast("zarr.Group", root[orbit_dir])
        attrs = dict(og.attrs)
        proj_code = cast("str", attrs["proj:code"])
        utm_bbox = cast("list[float]", attrs["spatial:bbox"])

        r10m = cast("zarr.Group", og["r10m"])
        times = np.array(cast("zarr.Array", r10m["time"])).tolist()
        if not times:
            continue

        # proj:shape / proj:transform live on the r10m group attrs in real stores; read best-effort so
        # minimal/legacy stores without them still build (just without those projection refinements).
        r10m_attrs = dict(r10m.attrs)
        all_times_ns.extend(times)
        wgs84_bboxes.append(_utm_to_wgs84(proj_code, utm_bbox))
        present.append(
            {
                "orbit": orbit_dir,
                "proj_code": proj_code,
                "utm_bbox": utm_bbox,
                "shape": r10m_attrs.get("spatial:shape"),
                "transform": r10m_attrs.get("spatial:transform"),
            }
        )

    if not all_times_ns:
        raise ValueError(f"No acquisitions found in Zarr store: {zarr_store}")

    # Temporal range
    start_dt = dt.datetime.fromtimestamp(min(all_times_ns) / 1e9, tz=dt.UTC)
    end_dt = dt.datetime.fromtimestamp(max(all_times_ns) / 1e9, tz=dt.UTC)

    # WGS84 bbox union across all present orbit directions
    west = min(b[0] for b in wgs84_bboxes)
    south = min(b[1] for b in wgs84_bboxes)
    east = max(b[2] for b in wgs84_bboxes)
    north = max(b[3] for b in wgs84_bboxes)
    wgs84_bbox = [west, south, east, north]

    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }

    # The preferred orbit (ascending if present) drives the single-valued projection fields and the
    # default render/preview; every present orbit gets its own first-class asset below.
    preferred = present[0]
    preferred_orbit = cast("str", preferred["orbit"])
    preferred_proj_code = cast("str", preferred["proj_code"])
    preferred_bbox = cast("list[float]", preferred["utm_bbox"])

    properties: dict[str, object] = {
        "start_datetime": start_dt.isoformat(),
        "end_datetime": end_dt.isoformat(),
        "title": f"Sentinel-1 GRD RTC γ⁰ — tile {tile_id}",
        "description": (
            "Radiometric-terrain-corrected (RTC) γ⁰ backscatter datacube from Sentinel-1 GRD, "
            "reprojected onto the Sentinel-2 MGRS/UTM grid."
        ),
        # `created` is the earliest acquisition (stable across rebuilds); `updated` tracks this build
        # (the cube is appended over time). See timestamps extension.
        "created": start_dt.isoformat(),
        "updated": dt.datetime.now(tz=dt.UTC).isoformat(),
        # Identity invariants (constant across the cube; platform is per-acquisition so omitted here —
        # a cube can mix S1A and S1C).
        "constellation": "sentinel-1",
        "instruments": ["c-sar"],
        "gsd": GSD,
        # SAR extension
        "sar:instrument_mode": "IW",
        "sar:frequency_band": "C",
        "sar:center_frequency": 5.405,
        "sar:polarizations": ["VV", "VH"],
        "sar:product_type": "GRD",
        # SAT extension: default orbit for the preview; per-acquisition items override this.
        "sat:orbit_state": preferred_orbit,
        # Projection extension
        "proj:code": preferred_proj_code,
        "proj:bbox": preferred_bbox,
        # Render extension: dual-pol RGB composite for previews/tiles
        "renders": {"rgb": _rgb_render(preferred_orbit)},
    }
    if preferred["shape"] is not None:
        properties["proj:shape"] = preferred["shape"]
    if preferred["transform"] is not None:
        properties["proj:transform"] = preferred["transform"]

    stac_extensions = [
        SAR_EXT,
        SAT_EXT,
        PROJ_EXT,
        RENDER_EXT,
        DATACUBE_EXT,
        TIMESTAMPS_EXT,
    ]

    # Datacube extension: temporal extent (not a values list — the cube grows by appending) + spatial
    # x/y extents on the projected grid, plus the cube variables.
    epsg = int(preferred_proj_code.split(":")[1])
    xmin, ymin, xmax, ymax = preferred_bbox
    properties["cube:dimensions"] = {
        "time": {"type": "temporal", "extent": [start_dt.isoformat(), end_dt.isoformat()]},
        "x": {"type": "spatial", "axis": "x", "extent": [xmin, xmax], "reference_system": epsg},
        "y": {"type": "spatial", "axis": "y", "extent": [ymin, ymax], "reference_system": epsg},
    }
    properties["cube:variables"] = {
        "vv": {"dimensions": ["time", "y", "x"], "type": "data", "unit": GAMMA0_UNIT},
        "vh": {"dimensions": ["time", "y", "x"], "type": "data", "unit": GAMMA0_UNIT},
        "border_mask": {"dimensions": ["time", "y", "x"], "type": "data"},
    }

    item = pystac.Item(
        id=f"s1-rtc-{tile_id}",
        geometry=geometry,
        bbox=wgs84_bbox,
        datetime=None,
        properties=properties,
        stac_extensions=stac_extensions,
        collection=collection_id,
    )

    store_str = str(zarr_store)
    item.add_asset(
        "zarr-store",
        pystac.Asset(
            href=store_str,
            media_type=ZARR_MEDIA_TYPE,
            roles=["data"],
            title="S1 GRD RTC Zarr store",
        ),
    )

    # One γ⁰ asset per present orbit group (fixes the duplicate-href vv/vh ambiguity and the missing
    # descending asset): VV/VH are addressable as named `bands`, not indistinguishable duplicate assets.
    # A separate border-mask asset exposes the valid-data mask variable in the same group.
    for info in present:
        orbit = cast("str", info["orbit"])
        short = _ORBIT_SHORT[orbit]
        group_href = f"{store_str}/{orbit}"
        item.add_asset(
            f"gamma0-rtc-backscatter-{short}",
            pystac.Asset(
                href=group_href,
                media_type=ZARR_MEDIA_TYPE,
                roles=["data"],
                title=f"γ⁰ RTC backscatter ({orbit})",
                extra_fields={
                    "bands": _gamma0_bands(),
                    "data_type": GAMMA0_DTYPE,
                    "nodata": GAMMA0_NODATA,
                    "unit": GAMMA0_UNIT,
                    "gsd": GSD,
                },
            ),
        )
        item.add_asset(
            f"border-mask-{short}",
            pystac.Asset(
                href=group_href,
                media_type=ZARR_MEDIA_TYPE,
                roles=["data"],
                title=f"Valid-data border mask ({orbit})",
                extra_fields={
                    "bands": [
                        {
                            "name": "border_mask",
                            "description": "Valid-data mask (0 = border/no-data, non-zero = valid)",
                            "data_type": BORDER_MASK_DTYPE,
                            "nodata": 0,
                        }
                    ],
                    "gsd": GSD,
                },
            ),
        )

    return item
