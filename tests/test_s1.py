"""
test_s1.py

Compares an S1 IW SLC product's converted-from-SAFE zarr store against its
geozarr store, burst by burst. Unlike S2 (whole-product "pair" comparison),
an S1 SLC product's natural comparison unit is the burst - each top-level
group (e.g. "S01SIWSLC_..._IW1_258173") is an independent subtree with its
own conditions/measurements/quality. A single product can have anywhere from
~10 to ~30 bursts, and each burst's "measurements/slc" array is large
(complex64, one array per burst), so by default only one representative
burst per subswath (IW1/IW2/IW3) is exercised - enough to cover every
subswath's structure without paying the full per-burst cost on every run.
Mark a test `@pytest.mark.slow` to run it over every burst instead (see
pyproject.toml's `slow` marker) - e.g. `pytest tests/test_s1.py -m slow`.

Burst IDs are discovered from the store itself (see build_pair_context /
_common_bursts below), not hardcoded - they're convention-derived from the
product (subswath + burst number) and their count varies per product.

Encapsulation for single-mode/single-subswath runs: burst IDs already embed
the subswath, so `pytest tests/test_s1.py -k IW1` isolates one subswath with
no extra plumbing, same as S2's `-k L2A`.
"""

from __future__ import annotations

import random
import re

import numpy as np
import pytest
import utils
import zarr

# ---------------------------------------------------------------------------

S1_SLC_STORE_CONFIGS: dict[str, dict[str, str]] = {
    "SLC": {
        "ref_input_path": "/home/samuel/data/samples/cpm_v300rc4a/safe_products/S1C_IW_SLC__1SDV_20260731T234807_20260731T234821_008794_0116E8_C079.SAFE",
        "geozarr_path": "/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S1C_IW_SLC__1SDV_20260731T234807_20260731T234821_008794_0116E8_C079.zarr",
    },
}

SAMPLE_FRACTION = 0.02  # fraction of chunks to sample per array (set to 1.0 for full compare)
MAX_SAMPLES = 50  # max chunks sampled per array
S1_IGNORE_ATTRS: set[str] = {
    "history",
    "created",
    "created_at",
    "date_created",
    "processing_time",
    "processing_history",
}

_SUBSWATH_RE = re.compile(r"_(IW\d)_")


def _subswath_of(burst: str) -> str:
    match = _SUBSWATH_RE.search(burst)
    return match.group(1) if match else "unknown"


def _representative_bursts(bursts: list[str]) -> list[str]:
    """One burst per subswath (the first, sorted) - the cheap default set.
    Full per-burst coverage is available by marking a test `slow`."""
    by_subswath: dict[str, str] = {}
    for burst in sorted(bursts):
        by_subswath.setdefault(_subswath_of(burst), burst)
    return sorted(by_subswath.values())


# Product-level state (store paths + opened root groups), cached per mode so
# the SAFE -> zarr conversion and root zarr.open_group calls only happen
# once per mode regardless of how many tests/pytest_generate_tests calls
# need them.
_PRODUCT_ROOTS_CACHE: dict[str, tuple[str, str, zarr.Group, zarr.Group]] = {}


def _get_product_roots(mode: str) -> tuple[str, str, zarr.Group, zarr.Group]:
    if mode not in _PRODUCT_ROOTS_CACHE:
        cfg = S1_SLC_STORE_CONFIGS[mode]
        product_files = utils.TestFiles(
            sensor="S1",
            mode=mode,
            ref_input_path=cfg["ref_input_path"],
            geozarr_path=cfg["geozarr_path"],
        )
        store_a, store_b = product_files.input_path, product_files.geozarr_path
        group_a = zarr.open_group(store_a, mode="r")
        group_b = zarr.open_group(store_b, mode="r")
        _PRODUCT_ROOTS_CACHE[mode] = (store_a, store_b, group_a, group_b)
    return _PRODUCT_ROOTS_CACHE[mode]


def _common_bursts(mode: str) -> list[str]:
    _, _, group_a, group_b = _get_product_roots(mode)
    bursts_a = {name for name, _ in group_a.groups()}
    bursts_b = {name for name, _ in group_b.groups()}
    return sorted(bursts_a & bursts_b)


_BURST_CTX_CACHE: dict[tuple[str, str], utils.PairContext] = {}


def _get_burst_ctx(mode: str, burst: str) -> utils.PairContext:
    key = (mode, burst)
    if key not in _BURST_CTX_CACHE:
        store_a, store_b, _, _ = _get_product_roots(mode)
        burst_group_a = zarr.open_group(store_a, mode="r", path=burst)
        burst_group_b = zarr.open_group(store_b, mode="r", path=burst)
        _BURST_CTX_CACHE[key] = utils.build_pair_context(
            store_a, store_b, burst_group_a, burst_group_b
        )
    return _BURST_CTX_CACHE[key]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate the (mode, burst)/(mode, burst, group_path)/(mode, burst,
    array_path)/(mode, burst, group) parametrize lists dynamically, by
    actually opening each mode's stores - burst IDs and the common
    groups/arrays/variable-groups within a burst can't be known without
    doing that. Bursts default to one-per-subswath unless the test is
    marked `slow`, in which case every common burst is used."""
    modes = sorted(S1_SLC_STORE_CONFIGS)
    names = set(metafunc.fixturenames)
    run_all_bursts = metafunc.definition.get_closest_marker("slow") is not None

    def bursts_for(mode: str) -> list[str]:
        all_bursts = _common_bursts(mode)
        return all_bursts if run_all_bursts else _representative_bursts(all_bursts)

    if {"mode", "burst", "group_path"} <= names:
        metafunc.parametrize(
            ("mode", "burst", "group_path"),
            [
                (m, b, path)
                for m in modes
                for b in bursts_for(m)
                for path in _get_burst_ctx(m, b).common_groups
            ],
        )
    elif {"mode", "burst", "array_path"} <= names:
        metafunc.parametrize(
            ("mode", "burst", "array_path"),
            [
                (m, b, path)
                for m in modes
                for b in bursts_for(m)
                for path in _get_burst_ctx(m, b).common_arrays
            ],
        )
    elif {"mode", "burst", "group"} <= names:
        metafunc.parametrize(
            ("mode", "burst", "group"),
            [
                (m, b, g)
                for m in modes
                for b in bursts_for(m)
                for g in _get_burst_ctx(m, b).variable_groups
            ],
        )
    elif {"mode", "burst"} <= names:
        metafunc.parametrize(
            ("mode", "burst"),
            [(m, b) for m in modes for b in bursts_for(m)],
        )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", sorted(S1_SLC_STORE_CONFIGS))
def test_no_missing_or_extra_bursts(mode: str) -> None:
    _, _, group_a, group_b = _get_product_roots(mode)
    bursts_a = {name for name, _ in group_a.groups()}
    bursts_b = {name for name, _ in group_b.groups()}
    missing_in_b = sorted(bursts_a - bursts_b)
    extra_in_b = sorted(bursts_b - bursts_a)
    assert not missing_in_b, f"[{mode}] Bursts present in A but missing in B: {missing_in_b}"
    assert not extra_in_b, f"[{mode}] Bursts present in B but missing in A: {extra_in_b}"


def test_no_missing_or_extra_groups(mode: str, burst: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    missing_in_b = sorted(set(ctx.groups_a) - set(ctx.groups_b))
    extra_in_b = sorted(set(ctx.groups_b) - set(ctx.groups_a))
    assert not missing_in_b, (
        f"[{mode}][{burst}] Groups present in A but missing in B: {missing_in_b}"
    )
    assert not extra_in_b, f"[{mode}][{burst}] Groups present in B but missing in A: {extra_in_b}"


def test_no_missing_or_extra_arrays(mode: str, burst: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    missing_in_b = sorted(set(ctx.arrays_a) - set(ctx.arrays_b))
    extra_in_b = sorted(set(ctx.arrays_b) - set(ctx.arrays_a))
    assert not missing_in_b, (
        f"[{mode}][{burst}] Arrays present in A but missing in B: {missing_in_b}"
    )
    assert not extra_in_b, f"[{mode}][{burst}] Arrays present in B but missing in A: {extra_in_b}"


def test_burst_root_attrs_match(mode: str, burst: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    diffs = utils.diff_attrs(
        dict(ctx.group_a.attrs), dict(ctx.group_b.attrs), ignore_attrs=S1_IGNORE_ATTRS
    )
    assert not diffs, f"[{mode}][{burst}] Burst root attrs differ:\n  " + "\n  ".join(diffs)


def test_group_attrs_match(mode: str, burst: str, group_path: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    diffs = utils.diff_attrs(
        dict(ctx.groups_a[group_path].attrs),
        dict(ctx.groups_b[group_path].attrs),
        ignore_attrs=S1_IGNORE_ATTRS,
    )
    assert not diffs, f"[{mode}][{burst}] Attrs differ for group '{group_path}':\n  " + "\n  ".join(
        diffs
    )


def test_array_metadata_match(mode: str, burst: str, array_path: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    arr_a, arr_b = ctx.arrays_a[array_path], ctx.arrays_b[array_path]
    diffs = []
    if arr_a.shape != arr_b.shape:
        diffs.append(f"shape differs: A={arr_a.shape} vs B={arr_b.shape}")
    if arr_a.dtype != arr_b.dtype:
        diffs.append(f"dtype differs: A={arr_a.dtype} vs B={arr_b.dtype}")
    if arr_a.chunks != arr_b.chunks:
        diffs.append(f"chunks differ: A={arr_a.chunks} vs B={arr_b.chunks}")
    assert not diffs, f"[{mode}][{burst}] Metadata differs for '{array_path}':\n  " + "\n  ".join(
        diffs
    )


def test_array_attrs_match(mode: str, burst: str, array_path: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    diffs = utils.diff_attrs(
        dict(ctx.arrays_a[array_path].attrs),
        dict(ctx.arrays_b[array_path].attrs),
        ignore_attrs=S1_IGNORE_ATTRS,
    )
    assert not diffs, f"[{mode}][{burst}] Attrs differ for array '{array_path}':\n  " + "\n  ".join(
        diffs
    )


def test_array_values_match(mode: str, burst: str, array_path: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    arr_a, arr_b = ctx.arrays_a[array_path], ctx.arrays_b[array_path]

    if arr_a.shape != arr_b.shape:
        pytest.skip("shape mismatch already reported by test_array_metadata_match")

    all_chunks = list(utils.iter_chunk_slices(arr_a.shape, arr_a.chunks))
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
        close, detail = utils.values_close(np.asarray(arr_a[sl]), np.asarray(arr_b[sl]))
        if not close:
            mismatches += 1
            if not first_detail:
                first_detail = f"at slice {sl}: {detail}"

    assert mismatches == 0, (
        f"[{mode}][{burst}] Values differ for '{array_path}': {mismatches}/{len(selected)} "
        f"sampled chunks mismatch. {first_detail}"
    )


# --------------------------------------------------------------------------- #
# xarray open check
#    Unlike S2, S1 SLC's "measurements/slc" carries no scale_factor/
#    add_offset/_FillValue (verified directly against the sample store), so
#    the CF mask_and_scale-effective checks that matter for S2 wouldn't
#    exercise anything here. This is kept to a single, cheap "does it open
#    through xarray at all" check per variable group instead.
# --------------------------------------------------------------------------- #
def test_xr_open_zarr_succeeds(mode: str, burst: str, group: str) -> None:
    ctx = _get_burst_ctx(mode, burst)
    full_group = f"{burst}/{group}" if group else burst
    try:
        utils.open_xr_group(ctx.store_a, full_group)
    except Exception as e:
        pytest.fail(f"[{mode}][{burst}] xr.open_zarr failed for store A group '{full_group}': {e}")
    try:
        utils.open_xr_group(ctx.store_b, full_group)
    except Exception as e:
        pytest.fail(f"[{mode}][{burst}] xr.open_zarr failed for store B group '{full_group}': {e}")
