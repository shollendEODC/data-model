"""
test_compare_s2l2a_zarr.py

Simple pytest script comparing Sentinel-2 L2A Zarr stores in pairs.
Edit STORE_PAIRS below (one entry per pair, e.g. "AB", "CD") and run:

    pytest test_compare_s2l2a_zarr.py -v
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import zarr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# ---- EDIT: define all store pairs to compare ------------------------------
STORE_PAIRS: dict[str, tuple[str, str]] = {
    "L2A,": (
        "/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S2B_MSIL2A_20260721T100559_N0512_R022_T33UWQ_20260721T143508.zarr",
        "/home/samuel/data/samples/cpm_v300rc4a/reference_zarr_stores/S2C_MSIL2A_20260716T100601_N0512_R022_T32UQU_20260716T152417.zarr",
    ),
    "L1C": (
        "/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S2C_MSIL1C_20260909T124301_N0512_R095_T27WXN_20260909T143930.zarr",
        "/home/samuel/data/samples/cpm_v300rc4a/reference_zarr_stores/S2C_MSIL1C_20260909T124301_N0512_R095_T27WXN_20260909T143930.zarr",
    ),
}
# ---------------------------------------------------------------------------

RTOL = 1e-5
ATOL = 1e-8
SAMPLE_FRACTION = 0.02  # fraction of chunks to sample per array (set to 1.0 for full compare)
MAX_SAMPLES = 50  # max chunks sampled per array
IGNORE_ATTRS = {
    "history",
    "created",
    "created_at",
    "date_created",
    "processing_time",
    "processing_history",
}
XR_WINDOW = 64  # size of the corner window used for the xarray-decoded spot-check

# Only compare groups/arrays whose path starts with one of these prefixes.
# Empty tuple = compare everything. These cover the full-resolution per-pixel
# raster data (reflectance bands, masks, AOT/WVP, cloud/snow probability).
# /conditions/geometry is deliberately excluded - it's coarse angular metadata
# (sun/view angles), not per-pixel raster data.
PATH_PREFIXES: tuple[str, ...] = (
    "measurements",
    "quality/atmosphere",
    "quality/mask",
    "quality/probability",
    "conditions/mask",
)


def in_scope(path: str) -> bool:
    if not PATH_PREFIXES:
        return True
    return any(path == prefix or path.startswith(prefix + "/") for prefix in PATH_PREFIXES)


def collect_tree(
    group: zarr.Group, prefix: str = ""
) -> tuple[dict[str, zarr.Group], dict[str, zarr.Array]]:
    """Recursively collect all group paths and array paths under `group`."""
    groups: dict[str, zarr.Group] = {}
    arrays: dict[str, zarr.Array] = {}
    for name, node in group.groups():
        path = f"{prefix}/{name}" if prefix else name
        groups[path] = node
        sub_groups, sub_arrays = collect_tree(node, path)
        groups.update(sub_groups)
        arrays.update(sub_arrays)
    for name, node in group.arrays():
        path = f"{prefix}/{name}" if prefix else name
        arrays[path] = node
    return groups, arrays


def parent_group(array_path: str) -> str:
    """'r10m/B02' -> 'r10m'; 'B02' (root array) -> ''"""
    return array_path.rsplit("/", 1)[0] if "/" in array_path else ""


@dataclass
class PairContext:
    """Everything derived from one A/B-style store pair, computed once at
    collection time so every test for that pair reuses the same data."""

    store_a: str
    store_b: str
    group_a: zarr.Group
    group_b: zarr.Group
    groups_a: dict[str, zarr.Group]
    groups_b: dict[str, zarr.Group]
    arrays_a: dict[str, zarr.Array]
    arrays_b: dict[str, zarr.Array]
    common_groups: list[str]
    common_arrays: list[str]
    variable_groups: list[str]


def build_pair_context(store_a: str, store_b: str) -> PairContext:
    group_a = zarr.open_group(store_a, mode="r")
    group_b = zarr.open_group(store_b, mode="r")
    groups_a, arrays_a = collect_tree(group_a)
    groups_b, arrays_b = collect_tree(group_b)
    groups_a = {p: g for p, g in groups_a.items() if in_scope(p)}
    groups_b = {p: g for p, g in groups_b.items() if in_scope(p)}
    arrays_a = {p: a for p, a in arrays_a.items() if in_scope(p)}
    arrays_b = {p: a for p, a in arrays_b.items() if in_scope(p)}
    common_groups = sorted(set(groups_a) & set(groups_b))
    common_arrays = sorted(set(arrays_a) & set(arrays_b))
    variable_groups = sorted({parent_group(p) for p in common_arrays})
    return PairContext(
        store_a=store_a,
        store_b=store_b,
        group_a=group_a,
        group_b=group_b,
        groups_a=groups_a,
        groups_b=groups_b,
        arrays_a=arrays_a,
        arrays_b=arrays_b,
        common_groups=common_groups,
        common_arrays=common_arrays,
        variable_groups=variable_groups,
    )


# One PairContext per pair, built once at collection time.
PAIRS: dict[str, PairContext] = {
    name: build_pair_context(a, b) for name, (a, b) in STORE_PAIRS.items()
}

# Flat (pair, path) parametrize lists - each pair contributes its own paths,
# since different pairs can have different common groups/arrays/variable_groups.
GROUP_PATH_PARAMS = [(pair, path) for pair, ctx in PAIRS.items() for path in ctx.common_groups]
ARRAY_PATH_PARAMS = [(pair, path) for pair, ctx in PAIRS.items() for path in ctx.common_arrays]
VARIABLE_GROUP_PARAMS = [
    (pair, group) for pair, ctx in PAIRS.items() for group in ctx.variable_groups
]


def open_xr_group(store_path: str, group_path: str) -> xr.Dataset:
    """Open one group of a store through xarray with full CF decoding on."""
    return xr.open_zarr(
        store_path,
        group=(group_path or None),
        mask_and_scale=True,
        decode_cf=True,
        consolidated=False,
    )


def open_xr_group_mask_scale_only(store_path: str, group_path: str) -> xr.Dataset:
    """Open one group with decode_cf=False and mask_and_scale=True exactly.

    Note: in this xarray version, decode_cf=False disables all CF sub-decoding
    regardless of mask_and_scale, so this mode passes raw values straight
    through unmasked/unscaled - included as an explicit, literal check of
    that combination rather than an attempt to isolate mask_and_scale on its
    own (see open_xr_group for the mode that actually applies masking)."""
    return xr.open_zarr(
        store_path,
        group=(group_path or None),
        mask_and_scale=True,
        decode_cf=False,
        consolidated=False,
    )


# The two xarray decode modes exercised below: full CF decoding, and the
# literal decode_cf=False + mask_and_scale=True combination.
XR_OPENERS: dict[str, Callable[[str, str], xr.Dataset]] = {
    "full_cf": open_xr_group,
    "decode_cf_false_mask_and_scale": open_xr_group_mask_scale_only,
}


def corner_selectors(sizes: dict[str, int], window: int = XR_WINDOW) -> dict[str, dict[str, slice]]:
    """Build isel selectors for the four spatial corners (top-left, top-right,
    bottom-left, bottom-right) of a 'y'/'x'-indexed array. Boundary/no-data
    effects (swath cutlines, incomplete overview blocks at coarse resolutions)
    concentrate at the edges of a tile, so a single top-left window can miss
    them entirely - checking all four corners catches this. Non-spatial dims
    (e.g. band, detector, angle) are left unrestricted. Falls back to a single
    window over all dims if the array has no 'y'/'x' dims."""
    if "y" not in sizes or "x" not in sizes:
        return {"window": {dim: slice(0, min(size, window)) for dim, size in sizes.items()}}

    def edge_slice(size: int, edge: str) -> slice:
        w = min(size, window)
        return slice(0, w) if edge == "start" else slice(max(0, size - w), size)

    base = {dim: slice(0, size) for dim, size in sizes.items() if dim not in ("y", "x")}
    y_size, x_size = sizes["y"], sizes["x"]

    return {
        "top_left": {
            **base,
            "y": edge_slice(y_size, "start"),
            "x": edge_slice(x_size, "start"),
        },
        "top_right": {
            **base,
            "y": edge_slice(y_size, "start"),
            "x": edge_slice(x_size, "end"),
        },
        "bottom_left": {
            **base,
            "y": edge_slice(y_size, "end"),
            "x": edge_slice(x_size, "start"),
        },
        "bottom_right": {
            **base,
            "y": edge_slice(y_size, "end"),
            "x": edge_slice(x_size, "end"),
        },
    }


def raw_slices_for_selector(
    raw_arr: zarr.Array, dims: tuple[str, ...], selector: dict[str, slice]
) -> tuple[slice, ...]:
    """Translate a decoded-DataArray isel-style selector dict into a positional
    tuple of slices for the underlying raw zarr array, assuming raw_arr's
    dimension order matches `dims` (both come from the same zarr array's
    dimension_names)."""
    return tuple(selector[d] for d in dims)


def diff_attrs(attrs_a: dict, attrs_b: dict) -> list[str]:
    keys_a = set(attrs_a) - IGNORE_ATTRS
    keys_b = set(attrs_b) - IGNORE_ATTRS

    # _FillValue is allowed to be present as an attribute on only one side:
    # some writers only ever set zarr's native storage-level fill_value and
    # never expose it as a CF `_FillValue` *attribute*, while others do.
    # Presence/absence alone isn't a real difference - only the *value*
    # matters when both sides happen to have it (checked below). Whether
    # masking actually behaves correctly either way is a separate, stronger
    # check - see test_xr_mask_and_scale_effective.
    presence_keys_a = keys_a - {"_FillValue"}
    presence_keys_b = keys_b - {"_FillValue"}

    diffs = [
        f"key '{key}' present in A but missing in B"
        for key in sorted(presence_keys_a - presence_keys_b)
    ]
    diffs.extend(
        f"key '{key}' present in B but missing in A"
        for key in sorted(presence_keys_b - presence_keys_a)
    )
    for key in sorted(keys_a & keys_b):
        va, vb = attrs_a[key], attrs_b[key]
        try:
            equal = (
                np.isclose(va, vb, rtol=RTOL, atol=ATOL)
                if isinstance(va, (int, float))
                else va == vb
            )
        except (TypeError, ValueError):
            equal = va == vb
        if not equal:
            diffs.append(f"key '{key}' differs: A={va!r} vs B={vb!r}")
    return diffs


def iter_chunk_slices(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> Iterator[tuple[slice, ...]]:
    ranges = [range(0, dim, chunk) for dim, chunk in zip(shape, chunks, strict=True)]
    for starts in itertools.product(*ranges):
        yield tuple(slice(s, min(s + c, d)) for s, c, d in zip(starts, chunks, shape, strict=True))


def values_close(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"chunk shape mismatch: {a.shape} vs {b.shape}"
    if np.issubdtype(a.dtype, np.complexfloating) or np.issubdtype(a.dtype, np.floating):
        close = np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        if not close:
            diff = np.abs(a.astype("complex128") - b.astype("complex128"))
            return False, f"max |diff|={np.nanmax(diff):.3e}"
        return True, ""
    equal = np.array_equal(a, b)
    if not equal:
        n_bad = np.count_nonzero(a != b)
        return False, f"{n_bad}/{a.size} elements differ (exact dtype)"
    return True, ""


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_no_missing_or_extra_groups(pair: str) -> None:
    ctx = PAIRS[pair]
    missing_in_b = sorted(set(ctx.groups_a) - set(ctx.groups_b))
    extra_in_b = sorted(set(ctx.groups_b) - set(ctx.groups_a))
    assert not missing_in_b, f"[{pair}] Groups present in A but missing in B: {missing_in_b}"
    assert not extra_in_b, f"[{pair}] Groups present in B but missing in A: {extra_in_b}"


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_no_missing_or_extra_arrays(pair: str) -> None:
    ctx = PAIRS[pair]
    missing_in_b = sorted(set(ctx.arrays_a) - set(ctx.arrays_b))
    extra_in_b = sorted(set(ctx.arrays_b) - set(ctx.arrays_a))
    assert not missing_in_b, f"[{pair}] Arrays present in A but missing in B: {missing_in_b}"
    assert not extra_in_b, f"[{pair}] Arrays present in B but missing in A: {extra_in_b}"


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_root_attrs_match(pair: str) -> None:
    ctx = PAIRS[pair]
    diffs = diff_attrs(dict(ctx.group_a.attrs), dict(ctx.group_b.attrs))
    assert not diffs, f"[{pair}] Root attrs differ:\n  " + "\n  ".join(diffs)


@pytest.mark.parametrize(("pair", "path"), GROUP_PATH_PARAMS)
def test_group_attrs_match(pair: str, path: str) -> None:
    ctx = PAIRS[pair]
    diffs = diff_attrs(dict(ctx.groups_a[path].attrs), dict(ctx.groups_b[path].attrs))
    assert not diffs, f"[{pair}] Attrs differ for group '{path}':\n  " + "\n  ".join(diffs)


@pytest.mark.parametrize(("pair", "path"), ARRAY_PATH_PARAMS)
def test_array_metadata_match(pair: str, path: str) -> None:
    ctx = PAIRS[pair]
    arr_a, arr_b = ctx.arrays_a[path], ctx.arrays_b[path]
    diffs = []
    if arr_a.shape != arr_b.shape:
        diffs.append(f"shape differs: A={arr_a.shape} vs B={arr_b.shape}")
    if arr_a.dtype != arr_b.dtype:
        diffs.append(f"dtype differs: A={arr_a.dtype} vs B={arr_b.dtype}")
    if arr_a.chunks != arr_b.chunks:
        diffs.append(f"chunks differ: A={arr_a.chunks} vs B={arr_b.chunks}")
    assert not diffs, f"[{pair}] Metadata differs for '{path}':\n  " + "\n  ".join(diffs)


@pytest.mark.parametrize(("pair", "path"), ARRAY_PATH_PARAMS)
def test_array_attrs_match(pair: str, path: str) -> None:
    ctx = PAIRS[pair]
    diffs = diff_attrs(dict(ctx.arrays_a[path].attrs), dict(ctx.arrays_b[path].attrs))
    assert not diffs, f"[{pair}] Attrs differ for array '{path}':\n  " + "\n  ".join(diffs)


@pytest.mark.parametrize(("pair", "path"), ARRAY_PATH_PARAMS)
def test_array_values_match(pair: str, path: str) -> None:
    ctx = PAIRS[pair]
    arr_a, arr_b = ctx.arrays_a[path], ctx.arrays_b[path]

    if arr_a.shape != arr_b.shape:
        pytest.skip("shape mismatch already reported by test_array_metadata_match")

    all_chunks = list(iter_chunk_slices(arr_a.shape, arr_a.chunks))
    n_select = max(
        1,
        min(
            MAX_SAMPLES,
            len(all_chunks),
            int(np.ceil(len(all_chunks) * SAMPLE_FRACTION)),
        ),
    )
    selected = random.sample(all_chunks, n_select)

    mismatches, first_detail = 0, ""
    for sl in selected:
        print(np.asarray(arr_a[sl]), np.asarray(arr_a[sl]).dtype)
        print(np.asarray(arr_b[sl]), np.asarray(arr_a[sl]).dtype)
        close, detail = values_close(np.asarray(arr_a[sl]), np.asarray(arr_b[sl]))
        if not close:
            mismatches += 1
            if not first_detail:
                first_detail = f"at slice {sl}: {detail}"

    assert mismatches == 0, (
        f"[{pair}] Values differ for '{path}': {mismatches}/{len(selected)} "
        f"sampled chunks mismatch. {first_detail}"
    )


# --------------------------------------------------------------------------- #
# 4. xarray CF / mask_and_scale decoding
#    (everything above reads raw zarr arrays directly - these tests instead go
#    through xr.open_zarr(..., mask_and_scale=True) to check that
#    scale_factor/add_offset/_FillValue decoding behaves the same way, and
#    consistently, for both stores. Each check runs under two decode modes -
#    full CF decoding, and the literal decode_cf=False + mask_and_scale=True
#    combination - and samples all four corners of each array, since
#    boundary/no-data effects concentrate there.)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", sorted(XR_OPENERS))
@pytest.mark.parametrize(("pair", "group"), VARIABLE_GROUP_PARAMS)
def test_xr_open_zarr_succeeds(pair: str, group: str, mode: str) -> None:
    ctx = PAIRS[pair]
    opener = XR_OPENERS[mode]
    try:
        opener(ctx.store_a, group)
    except Exception as e:
        pytest.fail(
            f"[{pair}] xr.open_zarr ({mode}) failed for store A group '{group or '/'}': {e}"
        )
    try:
        opener(ctx.store_b, group)
    except Exception as e:
        pytest.fail(
            f"[{pair}] xr.open_zarr ({mode}) failed for store B group '{group or '/'}': {e}"
        )


@pytest.mark.parametrize("mode", sorted(XR_OPENERS))
@pytest.mark.parametrize(("pair", "group"), VARIABLE_GROUP_PARAMS)
def test_xr_decoded_values_match(pair: str, group: str, mode: str) -> None:
    ctx = PAIRS[pair]
    opener = XR_OPENERS[mode]
    ds_a = opener(ctx.store_a, group)
    ds_b = opener(ctx.store_b, group)
    common_vars = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))  # type: ignore[reportArgumentType]

    diffs = []
    for var in common_vars:
        da_a = ds_a[var]
        da_b = ds_b[var]

        if da_a.dtype != da_b.dtype:
            diffs.append(f"'{var}': decoded dtype differs: A={da_a.dtype} vs B={da_b.dtype}")
            continue

        # Spot-check all four corners rather than loading the whole (possibly
        # huge) array - boundary/no-data effects concentrate at the edges.
        for corner_name, sel in corner_selectors(da_a.sizes).items():
            vals_a = da_a.isel(**sel).values
            vals_b = da_b.isel(**sel).values

            if vals_a.shape != vals_b.shape:
                diffs.append(
                    f"'{var}' [{corner_name}]: window shape differs: {vals_a.shape} vs {vals_b.shape}"
                )
                continue

            nan_mask_a = (
                np.isnan(vals_a)
                if np.issubdtype(vals_a.dtype, np.floating)
                else np.zeros_like(vals_a, dtype=bool)
            )
            nan_mask_b = (
                np.isnan(vals_b)
                if np.issubdtype(vals_b.dtype, np.floating)
                else np.zeros_like(vals_b, dtype=bool)
            )
            if not np.array_equal(nan_mask_a, nan_mask_b):
                diffs.append(
                    f"'{var}' [{corner_name}]: masked/NaN positions differ between A and B"
                )
                continue

            if not np.allclose(vals_a, vals_b, rtol=RTOL, atol=ATOL, equal_nan=True):
                max_diff = np.nanmax(np.abs(vals_a - vals_b))
                diffs.append(
                    f"'{var}' [{corner_name}]: decoded values differ, max |diff|={max_diff:.3e}"
                )

    assert not diffs, (
        f"[{pair}] xarray-decoded mismatch in group '{group or '/'}' (mode={mode}):\n  "
        + "\n  ".join(diffs)
    )


@pytest.mark.parametrize("mode", sorted(XR_OPENERS))
@pytest.mark.parametrize(("pair", "group"), VARIABLE_GROUP_PARAMS)
def test_xr_mask_and_scale_effective(pair: str, group: str, mode: str) -> None:
    """Sanity-check that decoding actually engaged: if a variable's raw attrs
    declare scale_factor/add_offset/_FillValue, the decoded array must come
    back as floating point, and any raw fill-value pixels must show up as NaN
    at every corner of the array. Checked independently for both stores."""
    if mode == "decode_cf_false_mask_and_scale":
        pytest.skip(
            "decode_cf=False disables masking entirely regardless of mask_and_scale in "
            "this xarray version, so this check would fail unconditionally here (even "
            "for two identical, correctly-attributed stores) and gives no comparative "
            "signal. See test_xr_decoded_values_match for the A-vs-B comparison under "
            "this mode."
        )
    ctx = PAIRS[pair]
    opener = XR_OPENERS[mode]
    ds_a = opener(ctx.store_a, group)
    ds_b = opener(ctx.store_b, group)

    for store_label, ds, raw_arrays in (
        ("A", ds_a, ctx.arrays_a),
        ("B", ds_b, ctx.arrays_b),
    ):
        for var in ds.data_vars:
            full_path = f"{group}/{var}" if group else var
            raw_arr = raw_arrays.get(full_path)  # type: ignore[reportArgumentType]
            if raw_arr is None:
                continue  # variable only exists in this store post-decode grouping quirk; skip
            raw_attrs = dict(raw_arr.attrs)
            has_scale_offset = "scale_factor" in raw_attrs or "add_offset" in raw_attrs
            fill_value = raw_attrs.get("_FillValue")

            if has_scale_offset:
                assert np.issubdtype(ds[var].dtype, np.floating), (
                    f"[{pair}][store {store_label}][{mode}] '{full_path}' has scale_factor/add_offset "
                    f"in raw attrs but decoded dtype is {ds[var].dtype}, not floating - mask_and_scale "
                    f"decoding does not appear to have taken effect."
                )

            if fill_value is None:
                continue

            for corner_name, sel in corner_selectors(ds[var].sizes).items():  # type: ignore[reportArgumentType]
                # raw_arr's dimension order matches ds[var]'s dimension order (both
                # come from the same underlying zarr array's dimension_names).
                raw_slices = raw_slices_for_selector(raw_arr, ds[var].dims, sel)  # type: ignore[reportArgumentType]
                raw_window = np.asarray(raw_arr[raw_slices])
                decoded_window = ds[var].isel(**sel).values  # type: ignore[reportArgumentType]

                if np.issubdtype(raw_window.dtype, np.floating):
                    # For floating-point source data the fill sentinel *is* NaN
                    # itself - that's how CF represents "no data" for floats.
                    # Some zarr encoders even serialize the _FillValue attribute
                    # as a base64-encoded byte string in this case (to survive a
                    # JSON round-trip of a NaN), so comparing raw_window against
                    # the raw attribute value directly isn't reliable. Just check
                    # NaN positions in the raw data directly instead.
                    expected_nan = np.isnan(raw_window)
                else:
                    try:
                        typed_fill = np.array(fill_value).astype(raw_window.dtype)
                    except (TypeError, ValueError):
                        continue  # fill_value isn't in a comparable form for this dtype; skip
                    expected_nan = raw_window == typed_fill

                actual_nan = np.isnan(decoded_window)
                assert np.array_equal(expected_nan, actual_nan), (
                    f"[{pair}][store {store_label}][{mode}] '{full_path}' [{corner_name}]: raw "
                    f"_FillValue={fill_value!r} pixels don't line up with NaN after decoding."
                )


"""
test_compare_s2l2a_zarr.py

Simple pytest script comparing two Sentinel-2 L2A Zarr stores.
Just edit STORE_A / STORE_B below and run:

    pytest test_compare_s2l2a_zarr.py -v
"""

# ---- EDIT THESE TWO PATHS -------------------------------------------------
STORE_A = "/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S2B_MSIL2A_20260721T100559_N0512_R022_T33UWQ_20260721T143508.zarr"
STORE_B = "/home/samuel/data/samples/cpm_v300rc4a/reference_zarr_stores/S2C_MSIL2A_20260716T100601_N0512_R022_T32UQU_20260716T152417.zarr"
# ---------------------------------------------------------------------------


def test_nan_decoding() -> None:
    ds_a = open_xr_group(STORE_A, "measurements/reflectance/r360m")
    ds_b = open_xr_group(STORE_B, "measurements/reflectance/r360m")
    print(" ")
    print(ds_a["b02"].attrs)
    print(ds_a["b02"].encoding)
    print(ds_b["b02"].attrs)
    print(ds_b["b02"].encoding)

    a = ds_a["b02"].values  # full array, not just the corner window
    b = ds_b["b02"].values
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    mismatch = nan_a != nan_b
    print(mismatch.sum(), "of", mismatch.size, "pixels disagree")

    print("A: total NaN fraction:", nan_a.mean())
    print("B: total NaN fraction:", nan_b.mean())
    print("mismatch fraction:", (nan_a != nan_b).mean())

    return
