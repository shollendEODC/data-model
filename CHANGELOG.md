# Changelog — `cpm-geozarr-eodc-large-refactor-s1slc` vs. `cpm-geozarr-eodc`

Large structural refactor of the package: the old pydantic-zarr "data API" validation layer is gone, replaced by lightweight product-type routing and per-sensor optimized converters. See `CHANGELOG_DETAILED.md` for the full file-by-file breakdown and known follow-up items.

Changelog written by Claude assitance.

**139 files changed, 4,311 insertions(+), 202,493 deletions(-)** (excluding `uv.lock`)

## New packages

- `s1_optimization/` — dedicated Sentinel-1 GRDH pipeline (`s1_converter.py`), plus `sentinel1_reprojection.py` (moved from `conversion/`).
- `generic/` — `generic_converter.py`, a slimmed-down fallback pipeline forproducts that aren't S1/S2/S3-optimized (rechunk + shard only, no GeoZarr compliance metadata — narrower scope than the old generic path).
- `s3_optimization/` — renamed from `s3_olci_optimization/`; gained `slstr_converter.py` (new Sentinel-3 SLSTR pipeline, no old-branch equivalent).
- `chunk_info/` — new hardcoded per-product chunking presets.
- `new_types.py`, `zcm.py` — replace `types.py` and the old `data_api/geozarr/multiscales/` pydantic models.
- `conversion/sentinel_modes.py` — `S1Type`/`S1Mode`/`S2Type` enums.

## Deleted packages (full removal)

- `codecs/` (custom scale-offset zarr codec was unused/unwired even on the base branch)
- `data_api/` (entire tree: `s1.py`, `s2.py`, `s3_olci.py`, `geozarr/` — pydantic-zarr `GroupSpec` structural validation models for all three sensors, ~4,500 lines)
- `pyz/` (pydantic-zarr `GroupSpec`/`ArraySpec` wrappers + Jupyter HTML repr helpers)
- `types.py`
- `s3_olci_optimization/olci_converter.py` (content re-added at new path with changes; see detailed changelog)
- `conversion/geozarr.py`, `conversion/open_source.py` (old generic pipeline + source-opening helper)
- `s2_optimization/`: `common.py`, `s2_data_consolidator.py`, `s2_resampling.py`, `s2_validation.py`

## Tests

**All 40 existing test files and their fixture data (`tests/_test_data/`, ~100 JSON files) were deleted**, with no replacement tests added on this branch yet(`tests/conftest.py` also removed). Tests will have to be rewritten from scratch — this is expected, not a gap, but the branch currently has **zero test coverage**.

## Architecture shift

- Structural validation (pydantic-zarr `GroupSpec` schemas per sensor) -> replaced by cheap heuristic detection in `cpm/routing.py` (`looks_like_sentinel{1,2,3_olci,3_slstr}`, keyed off `stac_discovery` `product:type` with a structural fallback).
- CLI (`cli.py`, argparse-based) is being retired in favor of the `eopf` CPM package's own CLI, via `cpm/writer.py::get_cli_command()` (click-based `convert-geozarr` command). `cli.py` currently has all logic commented out; `python -m eopf_geozarr` is a no-op until it's repointed or removed.
- GeoZarr minispec compliance validation (`data_api/geozarr/validation.py`) and OGC TileMatrixSet multiscales support (`data_api/geozarr/multiscales/tms.py`) are dropped with no replacement, per explicit decision to redo later / not needed.
- Retry-with-backoff for network writes (`max_retries`) is now a dead/unused parameter across the pipelines — no retry loop remains.

## Notable behavior changes

- S2 overview pyramids (r120m/r360m/r720m): now use masked-mean resampling uniformly for all variables, including SCL classification and cloud/snow probability masks (previously type-aware: nearest/max/mean per variable type).
- Scale/offset packing moved from a Zarr-native codec (unused on base branch anyway) to CF-attribute-based encoding (`scale_factor`/`add_offset` via xarray).
