import itertools
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr
from eopf.store.convert import convert
from global_test_settings import ATOL, RTOL

_tmp_root_dir: Path | None = None


def _tmp_root() -> Path:
    """Lazily create one shared tmp root for this test session's SAFE/SEN3
    -> zarr conversions, instead of a fresh untracked /tmp/<random> dir per
    TestFiles instance. Removed in full by cleanup_tmp_root()."""
    global _tmp_root_dir
    if _tmp_root_dir is None:
        _tmp_root_dir = Path(tempfile.mkdtemp(prefix="eopf_geozarr_test_"))
    return _tmp_root_dir


def cleanup_tmp_root() -> None:
    """Remove the shared conversion tmp root, if one was created. Call once
    at the end of the test session (see conftest.py)."""
    global _tmp_root_dir
    if _tmp_root_dir is not None:
        shutil.rmtree(_tmp_root_dir, ignore_errors=True)
        _tmp_root_dir = None


@dataclass
class TestFiles:
    sensor: str
    mode: str
    ref_input_path: str
    input_path: str = field(init=False, default="")
    geozarr_path: str

    def __post_init__(self) -> None:
        """Runs automatically right after __init__ - the idiomatic place to
        validate a dataclass's fields as soon as it's constructed."""
        self.validate_model()

    def validate_model(self) -> None:
        # validate geozarr path
        geozarr = Path(self.geozarr_path)
        if not geozarr.exists():
            raise FileNotFoundError(f"geozarr_path does not exist: {geozarr}")
        try:
            zarr.open_group(geozarr, mode="r")
        except Exception as e:
            raise ValueError(f"geozarr_path is not a readable zarr store: {geozarr}") from e

        # validate and checkif zarr or safe
        inputpath = Path(self.ref_input_path)
        if not inputpath.exists():
            raise FileNotFoundError(f"inputpath does not exist: {inputpath}")

        if inputpath.suffix == ".zarr":
            try:
                zarr.open_group(inputpath, mode="r")
                self.input_path = self.ref_input_path

            except Exception as e:
                raise ValueError(f"inputpath is not a readable zarr store: {inputpath}") from e

        elif inputpath.suffix in [".SAFE", ".SEN3"]:
            # convert the file to geozarr under the shared, managed tmp root
            # (see _tmp_root/cleanup_tmp_root) and compare against that.
            out_dir = _tmp_root() / f"{self.sensor}_{self.mode}_{inputpath.stem}"
            out_dir.mkdir(parents=True, exist_ok=True)
            outpath = str(out_dir / f"{inputpath.stem}.zarr")

            print(f"converting to: {outpath}")

            convert_to_tmp(input_path=self.ref_input_path, output_path=outpath)

            try:
                zarr.open_group(outpath, mode="r")

                # overwrite the orginal input path
                self.input_path = outpath

            except Exception as e:
                raise ValueError(f"inputpath is not a readable zarr store: {inputpath}") from e

        return


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


def corner_selectors(sizes: dict[str, int], window: int) -> dict[str, dict[str, slice]]:
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


def diff_attrs(attrs_a: dict, attrs_b: dict, ignore_attrs: set[str]) -> list[str]:
    keys_a = set(attrs_a) - ignore_attrs
    keys_b = set(attrs_b) - ignore_attrs

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


def convert_to_tmp(
    input_path: str,
    output_path: str,
) -> str:
    from eopf_geozarr.cpm.writer import register

    register()

    target_store_kwargs = {
        "engine": "geozarr",  # sets processing to geozarr form eopf_geozarr: rechunkerHEHE, geozarr, cpm-zarr
        "keep_scale_offset": True,  # keep uint16 instead of float, actually leads to an error if False
        "spatial_chunk": 1024,  # should feature chunking -> works: should also be set to override default chukning to 256
        "enable_sharding": True,  # adds sharding: one array -> one file
    }

    source_store_kwargs: dict[str, Any] = {"engine": "safe", "mode": "r"}

    convert(
        input_path,
        output_path,
        source_store_kwargs=source_store_kwargs,
        target_store_kwargs=target_store_kwargs,
    )

    return output_path
