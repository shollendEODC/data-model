# Detailed Changelog — `cpm-geozarr-eodc-large-refactor-s1slc` vs. `cpm-geozarr-eodc`

Full file-by-file diff between the base branch `cpm-geozarr-eodc` and the current working tree of `cpm-geozarr-eodc-large-refactor-s1slc` (includes uncommitted changes — this reflects the actual current state of the repo).

`git diff --shortstat cpm-geozarr-eodc -- . ':!uv.lock'`:
**139 files changed, 4,311 insertions(+), 202,493 deletions(-)**

(The deletion count is dominated by ~100 JSON test-fixture files, several of which were large sample product metadata dumps.)

Changelog written by Claude assitance.

---

## 1. Added files (9)

| File | Purpose |
|---|---|
| `src/eopf_geozarr/chunk_info/chunk_info.py` | Hardcoded chunking presets per product type (`s1slc_chunks`, `s1grdh_chunks`, `s2l2a_chunks`, `s3_srtsr_rbt`). Currently a plain module of dict literals, not yet wired into the converters. |
| `src/eopf_geozarr/conversion/sentinel_modes.py` | `S1Type`, `S1Mode`, `S2Type` `StrEnum`s with `from_filename()` classmethods, used for L1C/L2A and IW/EW branching. |
| `src/eopf_geozarr/generic/generic_converter.py` | New fallback pipeline (`create_generic_geozarr_dataset`) for products that don't match S1/S2/S3-optimized detection. Rechunk + shard + write only — does **not** build GeoZarr multiscales/CF metadata, and has no `crs_groups`/`gcp_group` handling (narrower scope than the old `conversion/geozarr.py::create_geozarr_dataset`). |
| `src/eopf_geozarr/new_types.py` | Replaces `types.py`. Same `TypedDict`s for S3 storage options, plus a broader `XarrayDataArrayEncoding`. Drops `ResamplingMethod`, `XARRAY_DIMS_KEY`, and four coordinate-attrs `TypedDict`s from the old file — all confirmed to have zero live references anywhere, so safe. |
| `src/eopf_geozarr/s1_optimization/s1_converter.py` | New Sentinel-1 GRDH conversion pipeline (`convert_s1grdh_optimized`). No pydantic-zarr schema; iterates DataTree groups generically instead of the old fixed hierarchy. |
| `src/eopf_geozarr/s3_optimization/olci_converter.py` | Re-added under the new package path with substantial changes (see §3). |
| `src/eopf_geozarr/s3_optimization/slstr_converter.py` | New Sentinel-3 SLSTR pipeline (`own_convert_slstr_optimized`) — genuinely new, no base-branch equivalent. |
| `.vscode/oldlaunch.json`, `.vscode/xlaunch.json` | Editor config, not package code. |

## 2. Deleted files (112, plus ~100 test fixtures counted separately in §6)

### Core "data API" validation layer — entire `data_api/` tree

All pydantic-zarr `GroupSpec`/`ArraySpec` structural validation models, one per sensor, used to validate/detect input product structure before conversion:

- `data_api/__init__.py`
- `data_api/s1.py` (1,237 lines) — Sentinel-1 GRDH schema (antenna_pattern, attitude,
  azimuth_fm_rate, coordinate_conversion, doppler_centroid, gcp, orbit, reference_replica, replica, terrain_height, calibration, noise/noise_azimuth/noise_range, measurements, conditions, quality, polarization, root)
- `data_api/s2.py` (693 lines) — Sentinel-2 L1C/L2A schema + band reference data (`NATIVE_BANDS`, `RESOLUTION_TO_METERS`, `Sentinel2BandInfo`)
- `data_api/s3_olci.py` (94 lines) — Sentinel-3 OLCI schema
- `data_api/geozarr/__init__.py`
- `data_api/geozarr/common.py` (268 lines) — CF standard-name table fetch/validation (`get_cf_standard_names`, `check_standard_name`), coordinate/grid_mapping referential checks
- `data_api/geozarr/geoproj.py` (38 lines) — `Proj`/`GeoProj` CRS attrs model
- `data_api/geozarr/projjson.py` (690 lines) — full PROJJSON pydantic model
- `data_api/geozarr/spatial.py` (33 lines) — `Spatial` convention attrs model
- `data_api/geozarr/store.py` (214 lines) — store-root/multiscale-group minispec profile models (`GeoZarrStoreAttrs`, `GeoZarrScaleLevel`, `GeoZarrMultiscaleMeta`)
- `data_api/geozarr/types.py` (116 lines) — includes OGC TileMatrixSet `TypedDict`s (`TileMatrixJSON`, `TileMatrixSetJSON`, `TMSMultiscalesJSON`) — **no ZCM-only replacement supports TMS-format multiscales anymore**
- `data_api/geozarr/v2.py`, `v3.py` — Zarr V2/V3 array/dataset structural models
- `data_api/geozarr/validation.py` (462 lines) — `validate_store`, `ValidationReport`, full walk-the-store minispec compliance checker. **No replacement exists.**
- `data_api/geozarr/multiscales/__init__.py`
- `data_api/geozarr/multiscales/geozarr.py` (103 lines) — ZCM ∪ TMS union model (`MultiscaleGroupAttrs`, "at least one of ZCM/TMS present")
- `data_api/geozarr/multiscales/tms.py` (56 lines) — OGC TMS types
- `data_api/geozarr/multiscales/zcm.py` — **renamed**, not deleted; moved to `src/eopf_geozarr/zcm.py` (see §4)

### `pyz/` — pydantic-zarr wrapper package

- `pyz/__init__.py`, `pyz/common.py` (466 lines), `pyz/v2.py`, `pyz/v3.py` — thin `GroupSpec`/`ArraySpec` subclasses adding `__repr__`/`_repr_html_` (Jupyter notebook pretty-printing). No functional logic beyond display; disappears naturally since `GroupSpec` trees are no longer constructed.

### `codecs/`

- `codecs/__init__.py` — renamed to `generic/__init__.py` (empty file, name change only)
- `codecs/scale_offset.py` (27 lines) — a `scale_offset_from_cf` helper building a Zarr-native `ScaleOffset` codec. Confirmed **zero callers anywhere in this repo's git history** — was already dead code on the base branch.

### `conversion/`

- `conversion/geozarr.py` (1,790 lines committed / already ~95% dead code by the time it was deleted) — the old generic pipeline: `create_geozarr_dataset`, `iterative_copy`, `consolidate_metadata`/`async_consolidate_metadata`,  `calculate_aligned_chunk_size`, `setup_datatree_metadata_geozarr_spec_compliant`,  `get_zarr_group`, band-by-band write-with-retry logic (`max_retries`, `time.sleep`  backoff). **Retry logic has no replacement** — `stream_write_dataset` (its successor in `conversion/utils.py`) is single-shot.
- `conversion/open_source.py` — `open_source_datatree()`: opened a source path (local/S3/URL) into an `xr.DataTree`, handling storage options, format detection, `mask_and_scale`, and an on-disk `CacheStore` read cache for repeated S3 reads. **`conversion/__init__.py` still referenced this file's import after deletion, this broke `import eopf_geozarr` entirely; already fixed by commenting out the import (confirmed working now).** For the primary CPM/`eopf` entry point, source-opening is now correctly delegated to the external `eopf` package's `convert()`. Calling the S1/S2/S3 optimized converters directly (outside CPM) still requires an already-open `DataTree` — no live helper builds one from a raw path anymore, and the `CacheStore` read-caching feature has no replacement.

### `s2_optimization/`

- `common.py` — shared S2 helpers (merged into `s2_multiscale.py`/`conversion/utils.py`)
- `s2_data_consolidator.py`, `s2_resampling.py`, `s2_validation.py` — deleted in an earlier commit on this branch (`369a52a`); their logic is now split between `s2_optimization/s2_multiscale.py` and `conversion/utils.py`. Note: the old `s2_resampling.py` had **type-aware resampling** (nearest for SCL classification, max/OR for quality masks, mean for reflectance); the new `conversion/utils.py::coarsen_variable` applies masked-mean uniformly to all variables when building synthetic r120m/r360m/r720m overviews.

### `s3_olci_optimization/`

- `olci_converter.py` deleted at the old path; content re-added at `s3_optimization/olci_converter.py` with changes (see §3). Other files in this package were pure renames (§4).

### `types.py`

Replaced by `new_types.py` (§1).

---

## 3. Files with both a path change *and* content changes

`git diff` reports these as delete+add rather than a clean rename because the content changed enough that git's rename-detection threshold wasn't met.

- `s3_olci_optimization/olci_converter.py` → `s3_optimization/olci_converter.py` (875 lines). Notable: the file now contains **two** near-duplicate top-level converters — `own_convert_olci_optimized` (the one actually wired into  `cpm/writer.py`) and a dead `convert_olci_optimized` left over from an earlier draft, which diverges in behavior (raw copy vs. re-encoded copy for `quality`/`conditions` groups). Recommend deleting the unused one. Also: the pydantic-based `is_sentinel3_olci_dataset` detection function is now fully commented out (superseded by `cpm/routing.py::looks_like_sentinel3_olci`).

## 4. Renames (clean, git-detected)

| Old path | New path |
|---|---|
| `conversion/sentinel1_reprojection.py` | `s1_optimization/sentinel1_reprojection.py` |
| `data_api/geozarr/multiscales/zcm.py` | `zcm.py` (top-level) |
| `s3_olci_optimization/__init__.py` | `s3_optimization/__init__.py` |
| `s3_olci_optimization/olci_band_mapping.py` | `s3_optimization/olci_band_mapping.py` |
| `s3_olci_optimization/olci_multiscale.py` | `s3_optimization/olci_multiscale.py` |
| `s3_olci_optimization/olci_reproject.py` | `s3_optimization/olci_reproject.py` |
| `codecs/__init__.py` | `generic/__init__.py` |
| `data_api/geozarr/__init__.py` | `s1_optimization/__init__.py` |
| `tests/test_data_api/__init__.py` | `generic/__init__.py`* |
| `tests/test_data_api/test_geozarr/__init__.py` | `s1_optimization/__init__.py`* |

\* git's rename detection matched these empty `__init__.py` files by content similarity across the test→src move; not a meaningful semantic rename.

## 5. Modified files (content changes, same path)

| File | Nature of change |
|---|---|
| `src/eopf_geozarr/__init__.py` | Public re-exports of the old generic pipeline (`create_geozarr_dataset`, `consolidate_metadata`, etc.) commented out — package now only exports `downsample_2d_array`, `is_grid_mapping_variable`, `validate_existing_band_data`. |
| `src/eopf_geozarr/cli.py` | ~95% of the file (all argparse subcommands: `convert`, `info`, `validate`, `convert-s2-optimized`, `convert-s3-olci-optimized`, dask cluster setup) commented out. `main()` currently does nothing — `python -m eopf_geozarr` is a no-op. Being superseded by `cpm/writer.py::get_cli_command()`, a click command exposed through the external `eopf` package's CLI instead. |
| `src/eopf_geozarr/conversion/__init__.py` | Drops re-exports of the deleted `geozarr.py`/`open_source.py` functions. |
| `src/eopf_geozarr/conversion/fs_utils.py` | Import source for `S3FsOptions`/`S3Credentials` switched from `types.py` to `new_types.py`; otherwise unchanged. |
| `src/eopf_geozarr/conversion/utils.py` | Substantially expanded — absorbed most of the responsibility of the deleted `data_api/geozarr/common.py`, `spatial.py`, `geoproj.py`, and the old `conversion/geozarr.py`'s metadata-writing helpers (`proj_attrs_for_crs`, `build_convention_attrs`, `grid_spatial_attrs`, `write_geo_metadata`, `write_store_root_geo_metadata`, `create_uniform_encoding`, `stream_write_dataset`, `simple_root_consolidation`). CRS + bbox root-level metadata (checked specifically) are correctly written by `write_store_root_geo_metadata`, called from `simple_root_consolidation`. |
| `src/eopf_geozarr/cpm/routing.py` | Product-type routing heuristics — the replacement for all per-sensor `data_api` structural validation. |
| `src/eopf_geozarr/cpm/writer.py` | `GeoZarrWriter` (EOWriter integration) + `get_cli_command()` (new click-based `convert-geozarr` CLI, replacing `cli.py`). Doesn't yet expose `compression_level`, `keep_scale_offset`, `max_retries`, or `--verbose` as CLI options, though the underlying `write()` method accepts them. |
| `src/eopf_geozarr/s2_optimization/s2_converter.py` | `is_sentinel2_dataset` (pydantic-based detection) now fully commented out, superseded by `cpm/routing.py::looks_like_sentinel2`. |
| `src/eopf_geozarr/s2_optimization/s2_multiscale.py` | Gained `inject_missing_bands`, quicklook/TCI group skip (tracked decision, issue #81), and the uniform mean-based `coarsen_variable` path noted in §2. |
| `src/eopf_geozarr/s1_optimization/s1_converter.py` | New file, see §1. Contains a probable logic bug: a GCP-dedup check `bool([(arr1 == arr2).all() for arr1, arr2 in pairwise(arrs)])` wraps a non-empty list in `bool()`, which is always `True` regardless of the comparisons — likely always collapses to the first polarization's GCPs. Also accepts but never uses `validate_output`/`max_retries` parameters. |
| `.vscode/launch.json` | Editor config. |

---

## 6. Test suite

**Every test file and fixture on the base branch was deleted, with no replacement added on this branch:**

- 40 test files across `tests/` (`test_*.py`) and `tests/test_data_api/` (including `conftest.py` in both locations)
- ~100 JSON fixture files under `tests/_test_data/` (geoproj, geozarr, optimized_geozarr, optimized_olci, projjson, s1/s2/s3 examples, v3_s2, zcm_multiscales examples) — these account for the bulk of the 202k deleted lines in the overall diffstat
- `tests/__init__.py`, `tests/test_docs.py`

This is understood to be intentional (tests are being rewritten against the new architecture), but as of this diff **the branch has zero test coverage**.There is an untracked `new_tests/test_s2.py` and a root-level `tests/test_s2.py` in progress, not yet part of a full suite.

---

## 7. Summary of known functional gaps (status as reviewed)

| Item | Status |
|---|---|
| Broken `conversion/__init__.py` import (`from .open_source import ...`) | **Fixed** — import commented out, package now imports cleanly. |
| `old_*` reference folders kept for this review | **Removed** — cleanup done. |
| CLI is a no-op (`cli.py::main()`) | Open — intentional, being replaced by `eopf` CPM's CLI; `cli.py`/`__main__.py` not yet repointed or removed. |
| Generic pipeline scope (no GeoZarr multiscale/CF metadata) | Open — confirmed intentional (generic overviews not required; only reflectance needs them, which is handled by the optimized S2 path). |
| Retry logic (`max_retries`) unused | Open — confirmed not needed. |
| Minispec validation (`validation.py`) dropped | Open — confirmed intentional, to be redone later. |
| OGC TileMatrixSet multiscales support dropped | Open — confirmed not needed. |
| CF `standard_name` vocabulary check dropped | Open — flagged as possibly worth a lightweight reimplementation (see prior discussion: lazy `functools.lru_cache`-based single-function check, not import-time network I/O). |
| S1 GCP-dedup `bool([...])` bug | Open — fix proposed (use `all(...)` instead of `bool(list)`), not yet applied. |
| S2 `BandInfo` missing `long_name`/`standard_name`/`units` defaults | Open — fix proposed (restore as dataclass fields via `__post_init__`), not yet applied. |
