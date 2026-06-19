# Spec: Shard the S1 RTC `conditions` arrays (gamma_area / LIA / incidence_angle)

**Status:** implemented on `feat/s1-gamma-area-sharding` (PR targets #180 `feat/s1-rtc-stac-builder`).
**Cross-repo origin:** Task **T5** of the data-pipeline plan
`data-pipeline/claude-docs/plans/s1_ingest_upload_perf.md` — the *biggest absolute* lever in the
S1 RTC ingest upload-bottleneck work. This spec keeps T5 in the data-model review loop, as that
plan decided (2026-06-18); the data-pipeline transfer changes (T1–T4) do **not** depend on it.

## Problem

A real S1 RTC cube (`s1-rtc-31TEG`, staging, measured 2026-06-18) is **3807 objects / 3.5 GB**,
and **3604 of them (94.7%)** are the `conditions/gamma_area_<relorbit>` arrays:

- shape `[10980, 10980]`, dtype `float32`, codecs `bytes + blosc`, inner chunk `366²`,
  **no `sharding_indexed` codec** → ~900 tiny chunk objects per array × ~4 arrays ≈ 3604 objects.

These arrays are time-invariant (one per relative orbit), yet because they are unsharded they
dominate the object count, which in turn dominated the ingest's S3 transfer wall-time (a live pod
sat ~34 min in "Uploading store" at 9 millicores — pure object-count latency, not bandwidth).

The multiscale **display pyramid** (`vv` / `vh` / `border_mask`, r10m…r720m) is **already sharded**
(one shard per time slice spanning the full spatial extent, inner chunk `366²`) — so the fix is to
apply that *same, existing* layout to the one array family that was left out.

## Objective

Write the `conditions` arrays with the **same `sharding_indexed` layout** the `vv`/`vh` pyramid
already uses: one shard spanning the full `(y, x)` extent, 512-aligned inner chunks. Each condition
array collapses from ~900 chunk objects to **1 shard object** (`~3604 → ~8` for the cube;
`3807 → ~210` total).

## Scope

- **In:** the condition-array `create_array` in `ingest_s1tiling_conditions`
  (`src/eopf_geozarr/conversion/s1_ingest.py`). All condition arrays go through this one call —
  `gamma_area`, `lia`, `incidence_angle` — and all share the same full-resolution 2D shape, so all
  are sharded by the single change. (Sharding `lia`/`incidence_angle` too is *more correct and less
  code* than special-casing `gamma_area`, and identical in rationale: fewer cloud objects.)
- **Out:** the display pyramid (already sharded — leave untouched); the overwrite-in-place branch
  (`conditions[name][:, :] = data`) is unchanged — an existing array keeps its codec; re-ingest of
  *old, unsharded* cubes (those are not auto-migrated — see Migration).

## Design

In `ingest_s1tiling_conditions`, the new-array branch mirrors the pyramid:

```python
inner_chunks = (calculate_aligned_chunk_size(h, 512), calculate_aligned_chunk_size(w, 512))
arr = conditions.create_array(
    array_name, shape=(h, w), dtype="float32",
    chunks=inner_chunks, shards=(h, w),          # one shard over the full extent (the only change)
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
    fill_value=float("nan"), dimension_names=["y", "x"],
)
```

`calculate_aligned_chunk_size` returns a **divisor** of the dimension near 512, so `(h, w)` is a
clean multiple of the inner chunk — the Zarr v3 shard-divisibility requirement (the same reason the
pyramid's `shard=(1, level_h, level_w)` / inner `(1, aligned, aligned)` is valid).

## Web-optimized-GeoZarr constraint check

`gamma_area`/`lia`/`incidence_angle` are **`conditions` arrays** (per-relative-orbit normalization
factors), **not** part of the multiscale pyramid TiTiler renders (`vv`/`vh`/`border_mask`). So
sharding them does **not** touch the web-render path; it only changes how a client reads a condition
array — a **windowed** read becomes one ranged GET into the shard instead of many chunk GETs + a
listing (better for cloud partial access), and the ingest writes **1 object instead of ~900** (the
upload lever). A full-array sequential read is the same bytes either way (see Benchmark caveat).
Values are byte-identical.

## Acceptance criteria

- [x] Condition arrays written with a sharding codec: `arr.shards == (h, w)`, inner
      `arr.chunks == (aligned, aligned)` (same config as `vv`). *(test `test_gamma_area_is_sharded`)*
- [x] Object-count collapse proven: a multi-inner-chunk array lands as **1** on-disk shard object,
      not one per inner chunk; values byte-identical through the shard.
      *(test `test_sharding_collapses_chunk_objects_to_one` — 9 inner chunks → 1 object)*
- [x] Existing data-integrity / shape / dtype / attr tests stay green (sharding is read-transparent).
- [x] **Real-S3 validation** (laptop→DE, `esa-zarr-sentinel-explorer-tests`, 2026-06-19): see
      Benchmark below — object collapse 100→1 proven on real S3, byte-identical read-back,
      divisibility valid at the production 10980². (Full-tile re-ingest census ~3807→~210 still
      pending an in-cluster run.)
- [x] Re-ingest path for existing (unsharded) cubes documented — see Migration.

## Benchmark (real OVH S3, 2026-06-19)

A 3660² (= 10×366 → 100 inner 366² chunks) gamma_area-like **smooth/compressible** surface
(models the real normalization factor, not random noise), sharded vs unsharded, on the live bucket;
upload via batched `fs.put(batch_size=32)` so concurrency is *already on* for the unsharded case:

| layout | S3 objects | PUT (best of 3) | full-array GET | byte-identical |
|---|---|---|---|---|
| unsharded | 100 | 4.43 s | 1.38 s | ✓ |
| **sharded** | **1** | **2.55 s (1.7×)** | 1.61 s | ✓ |

- **Object count 100 → 1** (production ~900 → 1 per `gamma_area`; cube 3604 → ~4). The core lever.
- **PUT 1.7× faster** *even with* batched concurrency hiding per-object latency; the ratio grows with
  object count (900/array ≈ 28 batch-waves → 1), so the production win is far larger than 1.7×.
- **Divisibility valid at the real 10980²**: `calculate_aligned_chunk_size(10980,512)=366`,
  `10980 % 366 == 0`, so `shards=(10980,10980)` builds without error.
- **Honest caveat:** a *full-array* sequential read is **not** faster sharded (1.61 vs 1.38 s) — it is
  the same bytes in one un-parallelizable object vs many concurrent chunk GETs. T5's win is
  object-count (upload + S3 listing/metadata overhead) and **windowed/partial** cloud reads (one
  ranged GET into the shard vs many chunk GETs), not full-read throughput.

## Verification

1. Unit: `uv run pytest tests/test_s1_rtc_ingest.py` (57 green; +2 sharding tests).
2. Object census on a re-ingested real tile → ~3807 → ~210 objects.
3. TiTiler still renders `vv`/`vh` for that cube (render path unaffected).
4. `xarray.open_zarr` / `zarr` reads `gamma_area_*` byte-identical to the unsharded version.

## Migration

Old cubes written before this change stay **unsharded** until rewritten — Zarr does not re-chunk in
place. Re-ingest (data-pipeline `argo submit --from cronworkflow/eopf-explorer-s1rtc` per tile, cron
suspended) rebuilds the conditions with the sharded layout. Until a cube is re-ingested it is still
correct, just object-heavy. Sequence the re-ingest after the rebuilt image is deployed.
