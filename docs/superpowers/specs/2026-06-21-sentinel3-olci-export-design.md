# Sentinel-3 OLCI L1 EFR → GeoZarr export — design

Status: draft for review
Date: 2026-06-21
Branch: `feat/sentinel3-export` (off `chore/new-conventions-metadata`)

## Goal

Add a first Sentinel-3 exporter to eopf-geozarr, scoped to **OLCI Level-1 EFR**
(Ocean and Land Colour Instrument, Earth-observation Full Resolution). It
converts an EOPF OLCI product into a GeoZarr-spec-compliant, multiscale Zarr
store, modeled on the existing Sentinel-2 exporter but adapted to OLCI's
fundamentally different geometry.

This is the first of several Sentinel-3 product types (SLSTR, SRAL, SYNERGY are
out of scope here and will get their own specs).

## Source format (verified against a real product)

Introspected from a real `_NT_` (non-time-critical, fully consolidated) product
on the EODC EOPF sample store, e.g.:
`S3A_OL_1_EFR____20251101T073957..._NT_004.zarr` in bucket
`e05ab01a9d56408d82ac32d69a5aae2a:202511-s03olcefr-eu` on `objects.eodc.eu`.

Key facts:

- **Zarr v2**, consolidated metadata at root (`.zmetadata`). NOTE: `_NR_`
  (near-real-time) copies of the same products are often *incomplete* (chunks
  but no metadata) — use `_NT_` products as the source of truth.
- Top-level groups mirror S2: `measurements`, `quality`, `conditions` (plus
  `*/orphans` subgroups, an OLCI artifact for removed/duplicate pixels).
- `measurements/`: 21 radiance bands `oa01_radiance` … `oa21_radiance`, each
  `[4090, 4865]` `uint16`, dims `(rows, columns)` — a **single full-resolution
  ~300 m grid; all 21 bands share the same shape** (unlike S2's 10/20/60 m).
  Each band has CF `scale_factor`/`add_offset`, `units`,
  `standard_name=toa_upwelling_spectral_radiance`, and
  `coordinates: latitude longitude altitude time_stamp`.
- **Geolocation is per-pixel / curvilinear**: `measurements/latitude`,
  `measurements/longitude`, `measurements/altitude` are 2D `[4090, 4865]`
  arrays (`latitude`/`longitude` are scaled `int32`, scale `1e-6`, with fill
  values). There is **no projected CRS, no EPSG, no affine transform** — root
  attrs are empty. OLCI is delivered in satellite swath geometry.
- `conditions/geometry`: sun/view angles (`sza`, `saa`, `oza`, `oaa`) on a
  **coarser across-track tie-point grid** `[4090, 77]`.
- `conditions/instrument`: per-band/detector spectral data (`lambda0`,
  `solar_flux`, `fwhm`) shaped `[21, 3700]`; `relative_spectral_covariance`
  `[21, 21]`.
- `conditions/meteorology`: ECMWF fields on the `[4090, 77]` tie-point grid,
  some with a `pressure_level` (25) dim.
- `conditions/image`: `altitude`, `detector_index`, `frame_offset`,
  `latitude`, `longitude` at full `[4090, 4865]`; `time_stamp` `[4090]`.

## Core design decision: native swath geometry (no reprojection)

OLCI L1 has no projected grid. We **preserve native swath geometry**: keep the
per-pixel `latitude`/`longitude` as 2-D auxiliary coordinate variables
(CF "two-dimensional coordinates" + GeoZarr geographic convention). **No
reprojection / resampling** — faithful and lossless.

Consequences:
- The GeoZarr `proj` convention is geographic (lat/lon), not a projected CRS +
  affine transform. We attach 2-D coordinate arrays via CF `coordinates` and the
  appropriate GeoZarr `spatial`/`proj` metadata for curvilinear data (no
  `spatial:transform`; lat/lon carried as coordinate arrays).
- Multiscale overviews are produced by **2×2 block reduction** of `rows`/`columns`:
  - **Radiance bands → block-average** (mean over each 2×2 block), like the S2
    reflectance path. This is a genuine reduction, so each level carries new
    information that justifies storing it (a literal `[::2,::2]` *subsample* would
    add no information over the base array and must NOT be re-saved).
  - **Coordinate arrays (`latitude`/`longitude`/`altitude`) → decimate**
    (take a fixed sub-pixel, e.g. block top-left), NOT average — so each overview
    pixel's geolocation remains a real measured position rather than an
    interpolated one. Radiance and coords are reduced together so each level's
    grid stays internally consistent.
  - Levels are declared with the GeoZarr **`multiscales`** convention. Per the
    multiscales spec, the per-level `transform` holds the **relative** index
    relationship (`scale: [2, 2]` from the source level) — which remains valid
    for a 2×2 reduction. We do **NOT** emit `spatial:transform` (the absolute
    affine), because OLCI has no regular grid; absolute geolocation is carried by
    each level's own 2-D lat/lon coordinate arrays. (Reprojection to a regular
    grid is explicitly a *future* option, not in this deliverable.)

## Scope (v1): measurements-first

- **`measurements/`** → GeoZarr-compliant multiscale group:
  - 21 radiance bands + the 2-D `latitude`/`longitude`/`altitude` coordinate
    arrays, CF `coordinates` linkage preserved, scale/offset preserved
    (same handling as S2 reflectance encoding).
  - `/2` overview pyramid (decimation), down to a configurable min dimension.
- **`conditions/` and `quality/`** → copied through faithfully but unoptimized
  (the way the S2 path copies non-reflectance groups as-is).
- **`orphans/`** subgroups → copied through as-is (not specially handled).
- Out of scope for v1: GeoZarr-converting the tie-point geometry grid, 3-D
  meteorology, and 1-D instrument arrays; SLSTR/SRAL/SYNERGY; reprojection.

## Architecture — mirror the S2 package

Reuse the S2 exporter's shape (a self-contained product package + a data_api
model + CLI auto-detection). New code:

```
src/eopf_geozarr/s3_olci_optimization/        # parallels s2_optimization/
  __init__.py
  olci_band_mapping.py     # 21 OLCI bands (oa01..oa21), band metadata; the
                           #   "all bands one resolution" config
  olci_multiscale.py       # swath /2 decimation pyramid + GeoZarr metadata;
                           #   decimates radiance + 2D coord arrays together
  olci_converter.py        # convert_olci_optimized(dt, *, output_path, ...)
                           #   entry point + is_sentinel3_olci_dataset()
  common.py                # (or reuse s2_optimization.common)

src/eopf_geozarr/data_api/s3_olci.py           # Sentinel3OlciRoot pydantic model
                                               #   (GroupSpec/TypedDict members)
```

Reused as-is from existing code:
- Generic encoding / chunk-alignment / fill-value helpers in
  `conversion/utils.py` and `fs_utils.py`.
- GeoZarr convention metadata helpers (`conversion/utils.build_convention_attrs`,
  zarr-cm), root consolidation, snapshot/round-trip test patterns.
- Type-aware resampling: OLCI radiance is all "reflectance-like" (block-average
  on decimation); we can reuse `s2_resampling` averaging or a thin OLCI variant.
  v1 only needs averaging/decimation since all bands are one continuous
  radiance type (no SCL/quality-mask variety in `measurements/`).

### Product detection

S2/S1 are detected **structurally** (validate the zarr group against
`Sentinel1Root | Sentinel2Root` pydantic models in `is_sentinel2_dataset`). OLCI
root attrs are empty, so structural detection is the right approach: add
`Sentinel3OlciRoot` and extend the adapter to
`Sentinel1Root | Sentinel2Root | Sentinel3OlciRoot`. The CLI `convert` command's
auto-detect dispatches to `convert_olci_optimized` when the product validates as
OLCI.

### CLI

- Auto-detect in `convert` (as S2 does).
- Dedicated `convert-s3-olci-optimized` subcommand mirroring
  `convert-s2-optimized` (spatial-chunk, sharding, compression-level,
  keep-scale-offset, skip-validation, dask-cluster, verbose).

## Data model (`data_api/s3_olci.py`)

Follow the S2 pattern (pydantic `GroupSpec` + `closed=True` TypedDict members),
NOT the S1 dynamic-mapping pattern, because OLCI has a fixed known structure:

- `Sentinel3OlciRoot(GroupSpec[..., Sentinel3OlciRootMembers])` with
  `measurements`, `quality`, `conditions` members.
- `Sentinel3OlciMeasurementsMembers`: the 21 `oaNN_radiance` arrays +
  `latitude`/`longitude`/`altitude` (+ optional `orphans`). Genuinely-optional
  members stay `NotRequired`/`total=False` (lesson from the S1 model: don't make
  variant keys required, or real products fail validation). Accessors narrow
  with `.get()` + guard.

Type checking: pyright (project standard). No `typing.Any`. Convention metadata
built via `zarr_cm` / `build_convention_attrs`.

## Testing (match existing pattern)

1. **Structure-dump fixture**: generate a JSON metadata-dump of a real OLCI
   `_NT_` product via pydantic-zarr `GroupSpec` (as `s1_examples`/`s2_examples`
   were made), commit to `tests/_test_data/s3_examples/`, add an
   `s3_olci_group_example` fixture in `conftest.py` that materializes it to zarr
   via `create_group_from_json`. First confirm the dump round-trips cleanly for
   OLCI.
2. **Unit tests**: band mapping, swath decimation correctness (radiance and
   lat/lon decimate consistently), GeoZarr metadata emitted, model validation
   (incl. a real-product round-trip).
3. **Golden-file snapshot** of the converted structure (like
   `optimized_geozarr_examples`), with the URL-only-diff regeneration discipline.
4. **Synthetic in-memory OLCI builder** for an integration test (mirrors the
   S1/S2 integration mocks) exercising `convert_olci_optimized` end-to-end.
5. **CLI e2e** for `convert-s3-olci-optimized` on the materialized fixture.

## Open questions / risks

- **CF curvilinear + GeoZarr `spatial`/`proj` for swath data**: confirm the
  exact GeoZarr metadata form for 2-D-coordinate (non-gridded) data during
  implementation; the conventions are grid-oriented and may need a geographic /
  coordinate-array representation rather than `spatial:transform`.
- **Decimation vs. averaging for overviews**: v1 uses simple /2; confirm whether
  block-averaging radiance (and what to do with lat/lon — decimate, not average)
  is preferred for the pyramid.
- **EOPF data access plumbing**: opening these remote products needs the right
  storage handling (tenant:bucket name breaks s3fs; `_NR_` products lack
  metadata). Test data is committed as JSON dumps, so this only matters for
  regenerating fixtures, not for CI.

## Decomposition / sequencing

This spec is one implementation plan: the OLCI measurements-first exporter.
Follow-ups (separate specs): OLCI conditions/quality GeoZarr conversion;
reprojection-to-grid option; SLSTR / SRAL / SYNERGY.
