# Sentinel-3 OLCI L1 EFR → GeoZarr Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Sentinel-3 OLCI L1 EFR exporter that converts an EOPF OLCI product into a GeoZarr-compliant, multiscale Zarr store, preserving native swath geometry (per-pixel 2-D lat/lon, no reprojection).

**Architecture:** Mirror the existing Sentinel-2 exporter: a self-contained `s3_olci_optimization/` package (band mapping, multiscale, converter), a `data_api/s3_olci.py` pydantic-zarr model for structural product detection, and CLI auto-detection plus a dedicated `convert-s3-olci-optimized` subcommand. Overviews are produced by /2 decimation of the swath grid (radiance bands and 2-D lat/lon/altitude coordinate arrays decimated together).

**Tech Stack:** Python 3.12+, pydantic v2 + pydantic-zarr, zarr v3 (output) / zarr v2 (EOPF input), xarray (DataTree), zarr-cm conventions, pyright (type checker), ruff, pytest.

Design doc: `docs/superpowers/specs/2026-06-21-sentinel3-olci-export-design.md`

## Global Constraints

- Python ≥ 3.12; modern type hints (`|`, `list`, `dict`).
- **Never use `typing.Any`.** Use `object` + narrowing or precise types. (Existing `ArraySpec[Any]` in s2.py is pre-existing; do not copy `Any` into new code — use `ArraySpec[object]` or a precise attrs type.)
- Type checker is **pyright** (`uv run --frozen pyright`); 0 errors required. Lint/format is **ruff** (`uv run ruff check`, `uv run ruff format`).
- Build convention metadata via `zarr_cm` / `eopf_geozarr.conversion.utils.build_convention_attrs`; never hand-assemble `zarr_conventions`.
- Run tools with `uv run`. Tests: `uv run pytest`.
- Commit messages end with the project's `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- Pydantic model members: keep genuinely-optional keys `NotRequired`/`total=False`; never make variant keys Required (real products fail validation otherwise). Property accessors narrow with `.get()` + guard.

---

## File Structure

Create:
- `src/eopf_geozarr/s3_olci_optimization/__init__.py` — package marker.
- `src/eopf_geozarr/s3_olci_optimization/olci_band_mapping.py` — the 21 OLCI band names + per-band metadata; "all bands one resolution" config.
- `src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py` — swath /2 decimation pyramid + GeoZarr metadata.
- `src/eopf_geozarr/s3_olci_optimization/olci_converter.py` — `convert_olci_optimized()` entry point + `is_sentinel3_olci_dataset()`.
- `src/eopf_geozarr/data_api/s3_olci.py` — `Sentinel3OlciRoot` pydantic-zarr model.
- `tests/test_data_api/test_s3_olci.py` — model + detection tests.
- `tests/test_olci_band_mapping.py` — band mapping tests.
- `tests/test_olci_multiscale.py` — decimation + metadata tests.
- `tests/test_olci_integration.py` — synthetic end-to-end + CLI e2e.
- `tests/_test_data/s3_examples/<one real OLCI product>.json` — committed structure dump (Task 8).

Modify:
- `src/eopf_geozarr/cli.py` — auto-detect OLCI in `convert_command`; add `convert-s3-olci-optimized` subcommand.
- `src/eopf_geozarr/s2_optimization/s2_converter.py` — extend the detection `TypeAdapter` union to include `Sentinel3OlciRoot` (or add a dedicated OLCI detector — see Task 4).
- `tests/conftest.py` — add `s3_olci_group_example` fixture + `s3_example_json_paths`.

---

## Task 1: OLCI band mapping

**Files:**
- Create: `src/eopf_geozarr/s3_olci_optimization/__init__.py`
- Create: `src/eopf_geozarr/s3_olci_optimization/olci_band_mapping.py`
- Test: `tests/test_olci_band_mapping.py`

**Interfaces:**
- Produces: `OLCI_BANDS: tuple[str, ...]` (the 21 names `oa01_radiance`..`oa21_radiance`); `OlciBandInfo` dataclass (`name: str`, `data_type: str`, `wavelength_center: float`); `OLCI_BAND_INFO: dict[str, OlciBandInfo]`; `RADIANCE_DTYPE = "uint16"`.

OLCI band central wavelengths (nm), Oa01–Oa21:
`400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75, 753.75, 761.25, 764.375, 767.5, 778.75, 865, 885, 900, 940, 1020`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_olci_band_mapping.py
from eopf_geozarr.s3_olci_optimization.olci_band_mapping import (
    OLCI_BANDS,
    OLCI_BAND_INFO,
    OlciBandInfo,
    RADIANCE_DTYPE,
)


def test_there_are_21_olci_bands() -> None:
    assert len(OLCI_BANDS) == 21
    assert OLCI_BANDS[0] == "oa01_radiance"
    assert OLCI_BANDS[-1] == "oa21_radiance"


def test_every_band_has_info() -> None:
    assert set(OLCI_BAND_INFO) == set(OLCI_BANDS)
    for name, info in OLCI_BAND_INFO.items():
        assert isinstance(info, OlciBandInfo)
        assert info.name == name
        assert info.data_type == RADIANCE_DTYPE
        assert info.wavelength_center > 0


def test_first_band_wavelength() -> None:
    assert OLCI_BAND_INFO["oa01_radiance"].wavelength_center == 400.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_olci_band_mapping.py -v`
Expected: FAIL (ModuleNotFoundError: eopf_geozarr.s3_olci_optimization).

- [ ] **Step 3: Create the package marker**

```python
# src/eopf_geozarr/s3_olci_optimization/__init__.py
"""Sentinel-3 OLCI L1 EFR optimization (GeoZarr export)."""
```

- [ ] **Step 4: Implement the band mapping**

```python
# src/eopf_geozarr/s3_olci_optimization/olci_band_mapping.py
"""Band definitions for Sentinel-3 OLCI L1 EFR.

OLCI has 21 radiance bands (Oa01..Oa21), all delivered at the same full
resolution (~300 m) on a single swath grid.
"""

from dataclasses import dataclass

RADIANCE_DTYPE = "uint16"

# Band index -> central wavelength in nm (OLCI Oa01..Oa21).
_WAVELENGTHS_NM: tuple[float, ...] = (
    400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 665.0, 673.75, 681.25,
    708.75, 753.75, 761.25, 764.375, 767.5, 778.75, 865.0, 885.0, 900.0,
    940.0, 1020.0,
)

OLCI_BANDS: tuple[str, ...] = tuple(f"oa{i:02d}_radiance" for i in range(1, 22))


@dataclass(frozen=True)
class OlciBandInfo:
    """Spectral characterization of a single OLCI radiance band."""

    name: str
    data_type: str
    wavelength_center: float  # nanometers


OLCI_BAND_INFO: dict[str, OlciBandInfo] = {
    name: OlciBandInfo(name=name, data_type=RADIANCE_DTYPE, wavelength_center=wl)
    for name, wl in zip(OLCI_BANDS, _WAVELENGTHS_NM, strict=True)
}
```

- [ ] **Step 5: Run tests + type/lint**

Run: `uv run pytest tests/test_olci_band_mapping.py -v && uv run --frozen pyright src/eopf_geozarr/s3_olci_optimization/olci_band_mapping.py && uv run ruff check src/eopf_geozarr/s3_olci_optimization/ tests/test_olci_band_mapping.py`
Expected: tests PASS; pyright 0 errors; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/eopf_geozarr/s3_olci_optimization/ tests/test_olci_band_mapping.py
git commit -m "feat(s3-olci): add OLCI band mapping

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Data API model + structural detection helper

**Files:**
- Create: `src/eopf_geozarr/data_api/s3_olci.py`
- Test: `tests/test_data_api/test_s3_olci.py`

**Interfaces:**
- Consumes: `OLCI_BANDS` (Task 1); `eopf_geozarr.pyz.v2.{ArraySpec, GroupSpec}`; `eopf_geozarr.data_api.geozarr.common.DatasetAttrs`.
- Produces:
  - `Sentinel3OlciMeasurementsMembers` (TypedDict, closed, total=False): 21 `oaNN_radiance` + `latitude`/`longitude`/`altitude` + optional `orphans`.
  - `Sentinel3OlciMeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel3OlciMeasurementsMembers])`.
  - `Sentinel3OlciRootMembers` (closed, total=False): `measurements` (required), `quality` (NotRequired), `conditions` (NotRequired).
  - `Sentinel3OlciRoot(GroupSpec[Sentinel3OlciRootAttrs, Sentinel3OlciRootMembers])` with `.measurements` accessor.

Use `ArraySpec[object]` (NOT `ArraySpec[Any]`). `quality`/`conditions` members typed as `GroupSpec[object, object]` (we don't model their internals in v1). Detection is structural: a product is OLCI iff it validates as `Sentinel3OlciRoot` (i.e. has `measurements` with the radiance bands).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_api/test_s3_olci.py
from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec


def _olci_arr() -> dict[str, object]:
    # minimal v2 ArraySpec-shaped dict for a 2-D uint16 array
    return ArraySpec(
        shape=(4, 5), chunks=(4, 5), dtype="<u2", fill_value=0,
        attributes={"_ARRAY_DIMENSIONS": ["rows", "columns"]},
    ).model_dump()


def test_validates_minimal_olci_product() -> None:
    radiance = {f"oa{i:02d}_radiance": _olci_arr() for i in range(1, 22)}
    coords = {c: _olci_arr() for c in ("latitude", "longitude", "altitude")}
    root = {
        "zarr_format": 2,
        "node_type": "group",
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {
            "measurements": {
                "zarr_format": 2, "node_type": "group", "attributes": {},
                "members": {**radiance, **coords},
            },
        },
    }
    model = Sentinel3OlciRoot.model_validate(root)
    assert "oa01_radiance" in model.measurements.members


def test_rejects_non_olci_product() -> None:
    import pytest
    from pydantic import ValidationError

    not_olci = {
        "zarr_format": 2, "node_type": "group",
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {"measurements": {
            "zarr_format": 2, "node_type": "group", "attributes": {},
            "members": {"reflectance": {
                "zarr_format": 2, "node_type": "group", "attributes": {}, "members": {}}},
        }},
    }
    with pytest.raises(ValidationError):
        Sentinel3OlciRoot.model_validate(not_olci)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_api/test_s3_olci.py -v`
Expected: FAIL (cannot import `Sentinel3OlciRoot`).

- [ ] **Step 3: Implement the model** (follow `data_api/s2.py` patterns exactly)

```python
# src/eopf_geozarr/data_api/s3_olci.py
"""Pydantic-zarr model for the Sentinel-3 OLCI L1 EFR EOPF Zarr structure.

Mirrors data_api/s2.py: GroupSpec + closed TypedDict members. Used for
structural product detection (an EOPF product is OLCI iff it validates here).
"""

from __future__ import annotations

from pydantic import BaseModel
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec


class Sentinel3OlciRootAttrs(BaseModel):
    """Root-level attributes for an OLCI DataTree (not validated in detail)."""

    other_metadata: dict[str, object]
    stac_discovery: dict[str, object]


class Sentinel3OlciMeasurementsMembers(TypedDict, closed=True, total=False):
    """Members of the OLCI measurements group.

    The 21 radiance bands and the per-pixel geolocation coordinate arrays are
    required in practice but typed optional so partial/variant products still
    validate; the converter checks for the bands it needs.
    """

    latitude: ArraySpec[object]
    longitude: ArraySpec[object]
    altitude: ArraySpec[object]
    orphans: GroupSpec[object, object]
    oa01_radiance: ArraySpec[object]
    oa02_radiance: ArraySpec[object]
    oa03_radiance: ArraySpec[object]
    oa04_radiance: ArraySpec[object]
    oa05_radiance: ArraySpec[object]
    oa06_radiance: ArraySpec[object]
    oa07_radiance: ArraySpec[object]
    oa08_radiance: ArraySpec[object]
    oa09_radiance: ArraySpec[object]
    oa10_radiance: ArraySpec[object]
    oa11_radiance: ArraySpec[object]
    oa12_radiance: ArraySpec[object]
    oa13_radiance: ArraySpec[object]
    oa14_radiance: ArraySpec[object]
    oa15_radiance: ArraySpec[object]
    oa16_radiance: ArraySpec[object]
    oa17_radiance: ArraySpec[object]
    oa18_radiance: ArraySpec[object]
    oa19_radiance: ArraySpec[object]
    oa20_radiance: ArraySpec[object]
    oa21_radiance: ArraySpec[object]


class Sentinel3OlciMeasurementsGroup(
    GroupSpec[DatasetAttrs, Sentinel3OlciMeasurementsMembers]
):
    """OLCI measurements group: 21 radiance bands + 2-D geolocation."""


class Sentinel3OlciRootMembers(TypedDict, closed=True, total=False):
    """Members of the OLCI root group."""

    measurements: Sentinel3OlciMeasurementsGroup
    quality: GroupSpec[object, object]
    conditions: GroupSpec[object, object]


class Sentinel3OlciRoot(GroupSpec[Sentinel3OlciRootAttrs, Sentinel3OlciRootMembers]):
    """Complete Sentinel-3 OLCI L1 EFR EOPF Zarr hierarchy."""

    @property
    def measurements(self) -> Sentinel3OlciMeasurementsGroup:
        group = self.members.get("measurements")
        if group is None:
            raise KeyError("measurements")
        return group
```

NOTE: to make detection meaningful (Task 4), the `measurements` member must be
required for a product to count as OLCI. If `closed=True, total=False` lets an
empty product validate, change `Sentinel3OlciRootMembers` so `measurements` is
required (a separate `closed=True` TypedDict without `total=False` containing
only `measurements`, with `quality`/`conditions` in a `total=False` mixin) OR
add an explicit check in `is_sentinel3_olci_dataset` (Task 4) that
`oa01_radiance` is among `measurements.members`. Implement the explicit check in
Task 4 (simpler, and keeps the model permissive). Adjust the
`test_rejects_non_olci_product` test if needed so it asserts via the Task 4
detector rather than model validation — but since model validation with
`closed=True` rejects the `reflectance` key under `measurements`, the test above
should pass as written. Run it and confirm.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_data_api/test_s3_olci.py -v`
Expected: PASS. If `test_rejects_non_olci_product` does not raise (because the
permissive members allow it), move that assertion into Task 4's detector test
and keep only the positive test here.

- [ ] **Step 5: Type + lint**

Run: `uv run --frozen pyright src/eopf_geozarr/data_api/s3_olci.py && uv run ruff check src/eopf_geozarr/data_api/s3_olci.py tests/test_data_api/test_s3_olci.py`
Expected: pyright 0 errors; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/eopf_geozarr/data_api/s3_olci.py tests/test_data_api/test_s3_olci.py
git commit -m "feat(s3-olci): add Sentinel3OlciRoot data-api model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Swath /2 decimation

**Files:**
- Create: `src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py`
- Test: `tests/test_olci_multiscale.py`

**Interfaces:**
- Consumes: `xarray`, `numpy`.
- Produces: `decimate_swath(ds: xr.Dataset, factor: int = 2) -> xr.Dataset` — returns a dataset with every 2-D `(rows, columns)` variable AND the 2-D coordinate arrays (`latitude`/`longitude`/`altitude`) subsampled `[::factor, ::factor]`, preserving attrs/encoding and CF `coordinates` linkage. 1-D and non-(rows,columns) variables are passed through unchanged.

Decimation (not averaging) is correct for v1: it keeps geolocation exact (an averaged lat/lon would no longer correspond to a real pixel). Radiance is decimated too for consistency with its coordinates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_olci_multiscale.py
import numpy as np
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_multiscale import decimate_swath


def _swath(rows: int = 8, cols: int = 6) -> xr.Dataset:
    rad = xr.DataArray(
        np.arange(rows * cols, dtype="uint16").reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"scale_factor": 0.5, "units": "mW.m-2.sr-1.nm-1"},
    )
    lat = xr.DataArray(
        np.linspace(0, 1, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"), attrs={"standard_name": "latitude"},
    )
    lon = xr.DataArray(
        np.linspace(10, 11, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"), attrs={"standard_name": "longitude"},
    )
    return xr.Dataset(
        {"oa01_radiance": rad},
        coords={"latitude": lat, "longitude": lon},
    )


def test_decimate_halves_each_axis() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].shape == (4, 3)
    assert out["latitude"].shape == (4, 3)
    assert out["longitude"].shape == (4, 3)


def test_decimate_takes_every_other_pixel() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    # top-left pixel is preserved exactly (no averaging)
    assert int(out["oa01_radiance"].values[0, 0]) == 0
    assert float(out["latitude"].values[0, 0]) == 0.0


def test_decimate_preserves_attrs() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].attrs["scale_factor"] == 0.5
    assert out["latitude"].attrs["standard_name"] == "latitude"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_olci_multiscale.py -v`
Expected: FAIL (cannot import `decimate_swath`).

- [ ] **Step 3: Implement decimation**

```python
# src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py
"""Multiscale (overview) generation for OLCI swath data.

OLCI L1 EFR is a curvilinear swath geolocated by per-pixel 2-D lat/lon arrays,
so overviews are produced by /2 decimation of the (rows, columns) grid: every
2-D variable and its 2-D coordinate arrays are subsampled together, keeping
geolocation exact. (Averaging is intentionally avoided — an averaged lat/lon
would not correspond to a real pixel.)
"""

from __future__ import annotations

import xarray as xr

SWATH_DIMS = ("rows", "columns")


def decimate_swath(ds: xr.Dataset, factor: int = 2) -> xr.Dataset:
    """Return *ds* with every (rows, columns) array subsampled by *factor*.

    Both data variables and coordinate variables that span exactly the swath
    dims are decimated `[::factor, ::factor]`; everything else is passed
    through unchanged. Attributes and encoding are preserved by xarray's isel.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return ds
    indexers = {
        dim: slice(None, None, factor) for dim in SWATH_DIMS if dim in ds.sizes
    }
    if not indexers:
        return ds
    return ds.isel(indexers)
```

- [ ] **Step 4: Run tests + type/lint**

Run: `uv run pytest tests/test_olci_multiscale.py -v && uv run --frozen pyright src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py && uv run ruff check src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py tests/test_olci_multiscale.py`
Expected: tests PASS; pyright 0; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py tests/test_olci_multiscale.py
git commit -m "feat(s3-olci): add swath /2 decimation for overviews

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Product detection (`is_sentinel3_olci_dataset`)

**Files:**
- Modify: `src/eopf_geozarr/s3_olci_optimization/olci_converter.py` (create in this task)
- Test: `tests/test_data_api/test_s3_olci.py` (extend)

**Interfaces:**
- Consumes: `Sentinel3OlciRoot` (Task 2); `OLCI_BANDS` (Task 1); `eopf_geozarr.pyz.v2.GroupSpec`; `zarr`.
- Produces: `is_sentinel3_olci_dataset(group: zarr.Group) -> bool` — True iff the group validates as `Sentinel3OlciRoot` AND `measurements` contains `oa01_radiance`.

Pattern mirrors `is_sentinel2_dataset` in `s2_converter.py` (validate `GroupSpec.from_zarr(group).model_dump()`), but the extra `oa01_radiance` check makes detection robust given the permissive model.

- [ ] **Step 1: Write the failing test (extend test_s3_olci.py)**

```python
# append to tests/test_data_api/test_s3_olci.py
def test_detector_accepts_olci_zarr(tmp_path) -> None:
    import zarr
    from eopf_geozarr.pyz.v2 import GroupSpec as PyzGroupSpec
    from eopf_geozarr.s3_olci_optimization.olci_converter import (
        is_sentinel3_olci_dataset,
    )

    # build a minimal OLCI zarr v2 store from the model dict used above
    radiance = {f"oa{i:02d}_radiance": _olci_arr() for i in range(1, 22)}
    coords = {c: _olci_arr() for c in ("latitude", "longitude", "altitude")}
    root_dict = {
        "zarr_format": 2, "node_type": "group",
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {"measurements": {
            "zarr_format": 2, "node_type": "group", "attributes": {},
            "members": {**radiance, **coords}}},
    }
    out = tmp_path / "olci.zarr"
    PyzGroupSpec.model_validate(root_dict).to_zarr(out, path="")  # type: ignore[arg-type]
    group = zarr.open_group(str(out), mode="r")
    assert is_sentinel3_olci_dataset(group) is True


def test_detector_rejects_s2_zarr(s2_group_example) -> None:
    import zarr
    from eopf_geozarr.s3_olci_optimization.olci_converter import (
        is_sentinel3_olci_dataset,
    )

    group = zarr.open_group(str(s2_group_example), mode="r")
    assert is_sentinel3_olci_dataset(group) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_api/test_s3_olci.py -v`
Expected: FAIL (cannot import `is_sentinel3_olci_dataset`).

- [ ] **Step 3: Implement the converter module with the detector**

```python
# src/eopf_geozarr/s3_olci_optimization/olci_converter.py
"""Top-level Sentinel-3 OLCI L1 EFR -> GeoZarr conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot

if TYPE_CHECKING:
    import zarr

log = structlog.get_logger()


def is_sentinel3_olci_dataset(group: "zarr.Group") -> bool:
    """Return True if *group* is a Sentinel-3 OLCI L1 EFR product.

    Detection is structural: the group must validate against
    ``Sentinel3OlciRoot`` and its ``measurements`` group must contain the
    first OLCI radiance band.
    """
    from eopf_geozarr.pyz.v2 import GroupSpec

    try:
        model = Sentinel3OlciRoot.model_validate(GroupSpec.from_zarr(group).model_dump())
    except ValueError as e:
        log.debug("Not an OLCI dataset", error=str(e))
        return False
    return "oa01_radiance" in model.measurements.members
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_data_api/test_s3_olci.py -v`
Expected: PASS (positive detect + S2 rejected).

- [ ] **Step 5: Type + lint**

Run: `uv run --frozen pyright src/eopf_geozarr/s3_olci_optimization/olci_converter.py && uv run ruff check src/eopf_geozarr/s3_olci_optimization/olci_converter.py tests/test_data_api/test_s3_olci.py`
Expected: pyright 0; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/eopf_geozarr/s3_olci_optimization/olci_converter.py tests/test_data_api/test_s3_olci.py
git commit -m "feat(s3-olci): add OLCI product detection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: GeoZarr metadata for a swath group

**Files:**
- Modify: `src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py`
- Test: `tests/test_olci_multiscale.py` (extend)

**Interfaces:**
- Consumes: `eopf_geozarr.conversion.utils.build_convention_attrs`, `zarr_cm` types.
- Produces: `swath_spatial_attrs(dims: tuple[str, str] = ("rows", "columns")) -> SpatialAttrs` — returns the `spatial:` convention data for curvilinear (no-transform) data: `spatial:dimensions = ["rows", "columns"]`, `spatial:registration = "pixel"`, and NO `spatial:transform`/`spatial:bbox` (geolocation lives in the 2-D lat/lon coordinate arrays, not an affine transform).

This isolates the one genuinely OLCI-specific GeoZarr decision (open question in the spec) into a small, tested unit. `build_convention_attrs(spatial=..., crs=None)` is called with `crs=None` because OLCI L1 carries no projected CRS — geolocation is via coordinate arrays. Confirm `build_convention_attrs` accepts `crs=None` (it does: signature is `crs: CRSLike | None`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_olci_multiscale.py
from eopf_geozarr.s3_olci_optimization.olci_multiscale import swath_spatial_attrs


def test_swath_spatial_attrs_has_no_transform() -> None:
    attrs = swath_spatial_attrs()
    assert attrs["spatial:dimensions"] == ["rows", "columns"]
    assert attrs["spatial:registration"] == "pixel"
    assert "spatial:transform" not in attrs
    assert "spatial:bbox" not in attrs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_olci_multiscale.py::test_swath_spatial_attrs_has_no_transform -v`
Expected: FAIL (cannot import `swath_spatial_attrs`).

- [ ] **Step 3: Implement**

```python
# add to src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr_cm import SpatialAttrs


def swath_spatial_attrs(
    dims: tuple[str, str] = SWATH_DIMS,
) -> "SpatialAttrs":
    """Spatial-convention data for curvilinear swath geometry.

    OLCI has no affine transform; geolocation is carried by 2-D lat/lon
    coordinate arrays, so we declare the spatial dimensions and pixel
    registration but no ``spatial:transform``/``spatial:bbox``.
    """
    return {
        "spatial:dimensions": [dims[0], dims[1]],
        "spatial:registration": "pixel",
    }
```

- [ ] **Step 4: Run tests + type/lint**

Run: `uv run pytest tests/test_olci_multiscale.py -v && uv run --frozen pyright src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py && uv run ruff check src/eopf_geozarr/s3_olci_optimization/`
Expected: PASS; pyright 0; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py tests/test_olci_multiscale.py
git commit -m "feat(s3-olci): swath spatial-convention attrs (no transform)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `convert_olci_optimized` entry point

**Files:**
- Modify: `src/eopf_geozarr/s3_olci_optimization/olci_converter.py`
- Test: `tests/test_olci_integration.py`

**Interfaces:**
- Consumes: `decimate_swath`, `swath_spatial_attrs` (Tasks 3/5); `OLCI_BANDS` (Task 1); `build_convention_attrs`; `xarray`, `zarr`; storage/consolidation helpers from `conversion` (`fs_utils`, `geozarr`).
- Produces:
  ```python
  def convert_olci_optimized(
      dt_input: xr.DataTree,
      *,
      output_path: str,
      enable_sharding: bool = False,
      spatial_chunk: int = 1024,
      compression_level: int = 3,
      min_dimension: int = 256,
      keep_scale_offset: bool = False,
  ) -> xr.DataTree: ...
  ```
  Writes a GeoZarr store at `output_path`: the `measurements` group with native-resolution radiance + 2-D coords + GeoZarr convention metadata, plus `/2` decimated overview subgroups (`r1`, `r2`, `r4`, …) down to `min_dimension`; copies `conditions`/`quality` through unchanged. Returns the opened output DataTree.

This is the orchestration task. Build it incrementally with a small synthetic OLCI DataTree (a helper in the test module, reused by Task 7). Keep the function focused; factor any growing helper (e.g. `_write_overviews`, `_copy_group`) into `olci_multiscale.py`.

- [ ] **Step 1: Write the failing test (synthetic OLCI builder + end-to-end)**

```python
# tests/test_olci_integration.py
import numpy as np
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_converter import convert_olci_optimized


def build_synthetic_olci(rows: int = 512, cols: int = 480) -> xr.DataTree:
    """Minimal synthetic OLCI L1 EFR datatree (measurements only)."""
    rng = np.random.default_rng(0)
    lat = np.linspace(40, 41, rows * cols).reshape(rows, cols)
    lon = np.linspace(10, 11, rows * cols).reshape(rows, cols)
    alt = np.zeros((rows, cols), dtype="int16")
    data = {}
    for i in range(1, 22):
        name = f"oa{i:02d}_radiance"
        arr = xr.DataArray(
            rng.integers(0, 6000, (rows, cols)).astype("uint16"),
            dims=("rows", "columns"),
            attrs={"scale_factor": 0.0139, "add_offset": 0.0,
                   "standard_name": "toa_upwelling_spectral_radiance",
                   "coordinates": "latitude longitude altitude"},
        )
        data[name] = arr
    ds = xr.Dataset(
        data,
        coords={
            "latitude": (("rows", "columns"), lat, {"standard_name": "latitude"}),
            "longitude": (("rows", "columns"), lon, {"standard_name": "longitude"}),
            "altitude": (("rows", "columns"), alt, {"standard_name": "altitude"}),
        },
    )
    return xr.DataTree.from_dict({"/measurements": ds})


def test_convert_olci_writes_measurements(tmp_path) -> None:
    dt = build_synthetic_olci()
    out = str(tmp_path / "olci_geozarr.zarr")
    convert_olci_optimized(dt, output_path=out)

    import zarr
    g = zarr.open_group(out, mode="r")
    # native measurements present
    assert "measurements" in g
    # all 21 bands at native res
    meas = g["measurements"]
    for i in range(1, 22):
        assert f"oa{i:02d}_radiance" in meas


def test_convert_olci_creates_overviews(tmp_path) -> None:
    dt = build_synthetic_olci(rows=512, cols=480)
    out = str(tmp_path / "olci_geozarr.zarr")
    convert_olci_optimized(dt, output_path=out, min_dimension=256)
    import zarr
    g = zarr.open_group(out, mode="r")
    # at least one decimated overview level exists under measurements
    meas = g["measurements"]
    subgroups = [k for k in meas.group_keys()]
    assert len(subgroups) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_olci_integration.py -v`
Expected: FAIL (convert_olci_optimized not implemented / missing behavior).

- [ ] **Step 3: Implement `convert_olci_optimized`** (incrementally; minimal to pass)

Implementation outline (write real code — this is the skeleton to flesh out against the tests; mirror `s2_multiscale.create_multiscale_from_datatree` for writing groups, encoding, and `build_convention_attrs` usage):

```python
# add to src/eopf_geozarr/s3_olci_optimization/olci_converter.py
import xarray as xr

from eopf_geozarr.conversion.utils import build_convention_attrs
from eopf_geozarr.s3_olci_optimization.olci_multiscale import (
    decimate_swath,
    swath_spatial_attrs,
)


def _overview_levels(rows: int, cols: int, min_dimension: int) -> int:
    """Number of /2 decimations until min(rows, cols) would drop below min_dimension."""
    levels = 0
    r, c = rows, cols
    while min(r, c) // 2 >= min_dimension:
        r, c = r // 2, c // 2
        levels += 1
    return levels


def convert_olci_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool = False,
    spatial_chunk: int = 1024,
    compression_level: int = 3,
    min_dimension: int = 256,
    keep_scale_offset: bool = False,
) -> xr.DataTree:
    """Convert an EOPF OLCI L1 EFR product to a GeoZarr multiscale store."""
    measurements = dt_input["/measurements"].to_dataset()

    # Attach GeoZarr convention metadata for native-resolution swath data.
    conv = build_convention_attrs(spatial=swath_spatial_attrs(), crs=None)
    measurements.attrs.update(dict(conv))

    # Write native resolution.
    measurements.to_zarr(
        output_path, group="measurements", mode="w",
        consolidated=False, zarr_format=3,
    )

    # Write /2 decimated overviews as subgroups r2, r4, ...
    rows = measurements.sizes["rows"]
    cols = measurements.sizes["columns"]
    current = measurements
    for level in range(1, _overview_levels(rows, cols, min_dimension) + 1):
        current = decimate_swath(current, factor=2)
        current.to_zarr(
            output_path, group=f"measurements/r{2 ** level}", mode="a",
            consolidated=False, zarr_format=3,
        )

    # Copy conditions/quality through unchanged (if present).
    for grp in ("conditions", "quality"):
        try:
            node = dt_input[f"/{grp}"]
        except KeyError:
            continue
        node.to_zarr(output_path, group=grp, mode="a", consolidated=False, zarr_format=3)

    return xr.open_datatree(output_path, engine="zarr", chunks={})
```

NOTE: this skeleton uses xarray's `to_zarr` for simplicity. If the project's
chunk-alignment / encoding helpers (`conversion.utils`,
`s2_multiscale` encoding handling) are needed to match GeoZarr output
conventions (chunking, sharding, scale-offset), adopt them here the way
`s2_multiscale` does. Keep `keep_scale_offset`/`enable_sharding`/`spatial_chunk`/
`compression_level` honored (wire into encoding); if a parameter is not yet used
by the minimal pass, leave a typed parameter and a follow-up note rather than
silently ignoring — but prefer wiring encoding via the existing helpers.

- [ ] **Step 4: Run tests + type/lint**

Run: `uv run pytest tests/test_olci_integration.py -v && uv run --frozen pyright src/eopf_geozarr/s3_olci_optimization/ && uv run ruff check src/eopf_geozarr/s3_olci_optimization/ tests/test_olci_integration.py`
Expected: tests PASS; pyright 0; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/eopf_geozarr/s3_olci_optimization/olci_converter.py tests/test_olci_integration.py
git commit -m "feat(s3-olci): add convert_olci_optimized entry point

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: CLI integration (auto-detect + `convert-s3-olci-optimized`)

**Files:**
- Modify: `src/eopf_geozarr/cli.py`
- Modify: `src/eopf_geozarr/s2_optimization/s2_converter.py` (only if you choose the union-adapter detection approach; otherwise no change — Task 4's standalone detector is used)
- Test: `tests/test_olci_integration.py` (extend with a CLI test)

**Interfaces:**
- Consumes: `convert_olci_optimized`, `is_sentinel3_olci_dataset` (Tasks 4/6); `_is_sentinel2_input` pattern in `cli.py`.
- Produces: CLI behavior — `convert` auto-routes OLCI products to `convert_olci_optimized`; new `convert-s3-olci-optimized` subcommand.

In `convert_command`, add OLCI detection BEFORE the generic path and AFTER S2 (so the order is S2 → OLCI → S1/generic), mirroring the `if _is_sentinel2_input(dt):` block. Add `_is_sentinel3_olci_input(dt)` wrapping `is_sentinel3_olci_dataset(get_zarr_group(dt))` (guard exceptions, like `_is_sentinel2_input`). Add `add_s3_olci_optimization_commands(subparsers)` mirroring `add_s2_optimization_commands` and call it next to `add_s2_optimization_commands(subparsers)`.

- [ ] **Step 1: Write the failing CLI test**

```python
# append to tests/test_olci_integration.py
import subprocess
import sys


def test_cli_convert_s3_olci_optimized(tmp_path) -> None:
    # materialize a synthetic OLCI product to a zarr v2 store on disk
    dt = build_synthetic_olci(rows=300, cols=300)
    src = tmp_path / "olci_src.zarr"
    dt.to_zarr(src, mode="w", consolidated=False)
    out = tmp_path / "olci_out.zarr"
    result = subprocess.run(
        [sys.executable, "-m", "eopf_geozarr", "convert-s3-olci-optimized",
         str(src), str(out), "--spatial-chunk", "256"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    import zarr
    g = zarr.open_group(str(out), mode="r")
    assert "measurements" in g
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_olci_integration.py::test_cli_convert_s3_olci_optimized -v`
Expected: FAIL (unknown subcommand).

- [ ] **Step 3: Add the CLI subcommand + auto-detect**

In `cli.py`, near the top imports add:
```python
from eopf_geozarr.s3_olci_optimization.olci_converter import (
    convert_olci_optimized,
    is_sentinel3_olci_dataset,
)
```

Add the input helper (mirror `_is_sentinel2_input`):
```python
def _is_sentinel3_olci_input(dt: xr.DataTree) -> bool:
    try:
        return is_sentinel3_olci_dataset(get_zarr_group(dt))
    except Exception:  # noqa: BLE001 - detection must never crash convert
        return False
```

In `convert_command`, after the S2 block and before the generic/S1 path:
```python
        if _is_sentinel3_olci_input(dt):
            log.info("Detected Sentinel-3 OLCI product; using OLCI converter")
            dt_geozarr = convert_olci_optimized(
                dt,
                output_path=args.output_path,
                enable_sharding=args.enable_sharding,
                spatial_chunk=args.spatial_chunk,
            )
            # (skip the generic path; mirror how the S2 block returns/continues)
```
Match exactly how the S2 block hands off (return vs. fallthrough) in the current `convert_command`.

Add the subcommand (mirror `add_s2_optimization_commands`):
```python
def add_s3_olci_optimization_commands(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "convert-s3-olci-optimized",
        help="Convert a Sentinel-3 OLCI L1 EFR dataset to optimized GeoZarr",
    )
    p.add_argument("input_path", type=str, help="Path to input OLCI dataset (Zarr)")
    p.add_argument("output_path", type=str, help="Path for output optimized dataset")
    p.add_argument("--spatial-chunk", type=int, default=1024, help="Spatial chunk size")
    p.add_argument("--enable-sharding", action="store_true", help="Enable Zarr v3 sharding")
    p.add_argument("--compression-level", type=int, default=3, choices=range(1, 10),
                   help="Compression level 1-9 (default: 3)")
    p.add_argument("--min-dimension", type=int, default=256,
                   help="Minimum overview dimension (default: 256)")
    p.add_argument("--keep-scale-offset", action="store_true",
                   help="Preserve scale-offset encoding instead of decoding to float")
    p.add_argument("--verbose", action="store_true", help="Enable verbose output")
    p.set_defaults(func=convert_s3_olci_optimized_command)


def convert_s3_olci_optimized_command(args: argparse.Namespace) -> None:
    storage_options = get_storage_options(str(args.input_path))
    dt_input = xr.open_datatree(
        str(args.input_path), engine="zarr", chunks="auto",
        storage_options=storage_options,
    )
    convert_olci_optimized(
        dt_input,
        output_path=args.output_path,
        enable_sharding=args.enable_sharding,
        spatial_chunk=args.spatial_chunk,
        compression_level=args.compression_level,
        min_dimension=args.min_dimension,
        keep_scale_offset=args.keep_scale_offset,
    )
    log.info("✅ S3 OLCI optimization completed", output_path=args.output_path)
```

And register it next to the S2 registration:
```python
    add_s2_optimization_commands(subparsers)
    add_s3_olci_optimization_commands(subparsers)
```

- [ ] **Step 4: Run tests + type/lint**

Run: `uv run pytest tests/test_olci_integration.py -v && uv run --frozen pyright src/eopf_geozarr/cli.py && uv run ruff check src/eopf_geozarr/cli.py`
Expected: tests PASS; pyright 0; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/eopf_geozarr/cli.py tests/test_olci_integration.py
git commit -m "feat(s3-olci): CLI auto-detect + convert-s3-olci-optimized command

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Real-product fixture + round-trip test

**Files:**
- Create: `tests/_test_data/s3_examples/<product>.json`
- Modify: `tests/conftest.py`
- Test: `tests/test_data_api/test_s3_olci.py` (extend with a real-product round-trip)

**Interfaces:**
- Consumes: `create_group_from_json` (existing conftest helper); `is_sentinel3_olci_dataset`.
- Produces: `s3_example_json_paths` tuple + `s3_olci_group_example` fixture in conftest.

**Generating the fixture** (run once, by the implementer, to create the committed JSON). The product is on the EODC EOPF store; use an `_NT_` product (the `_NR_` copies lack metadata). The store's `tenant:bucket` name breaks s3fs, but the consolidated `.zmetadata` is fetchable over HTTPS. Generate the structure dump with pydantic-zarr from the consolidated metadata. Reference product:
`https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202511-s03olcefr-eu/01/products/cpm_v262/S3A_OL_1_EFR____20251101T073957_20251101T074257_20251102T084255_0179_132_149_2160_PS1_O_NT_004.zarr`

Write a one-off script `scripts/dump_olci_example.py` (NOT committed to src; can live under a scratch dir or `scripts/`) that builds a `pydantic_zarr.v2.GroupSpec` from the product's consolidated metadata and writes `model_dump_json(indent=2)` to the fixture path. If opening the remote store with xarray/zarr is blocked by the store's naming, build the `GroupSpec` dict directly from the fetched `.zmetadata` JSON (the `metadata` map contains every `.zgroup`/`.zarray`/`.zattrs`). The committed JSON must be a structure-only dump (chunks not required; shapes/dtypes/attrs are what matter), matching the form of `tests/_test_data/s2_examples/*.json`.

Keep the fixture small if the full product is large: it is acceptable to truncate array shapes in the JSON (the model/detection tests only need structure), but document any truncation in a top-level attribute or a sibling README note.

- [ ] **Step 1: Generate and commit the fixture JSON**

Produce `tests/_test_data/s3_examples/S3A_OL_1_EFR____20251101T073957_..._NT_004.json` via the one-off script. Verify it loads:
```bash
uv run python -c "import json,pathlib; json.loads(pathlib.Path('tests/_test_data/s3_examples/S3A_OL_1_EFR____20251101T073957_20251101T074257_20251102T084255_0179_132_149_2160_PS1_O_NT_004.json').read_text()); print('ok')"
```
Expected: `ok`.

- [ ] **Step 2: Add conftest fixture**

```python
# in tests/conftest.py, near the other *_example_json_paths
s3_example_json_paths = tuple(pathlib.Path("tests/_test_data/s3_examples").glob("*.json"))


@pytest.fixture(params=s3_example_json_paths, ids=get_stem)
def s3_olci_group_example(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> pathlib.Path:
    """Path to a Zarr group with the layout of a Sentinel-3 OLCI product."""
    return create_group_from_json(request.param, tmp_path)
```

- [ ] **Step 3: Write the round-trip / detection test**

```python
# append to tests/test_data_api/test_s3_olci.py
def test_real_olci_product_is_detected(s3_olci_group_example) -> None:
    import zarr
    from eopf_geozarr.s3_olci_optimization.olci_converter import (
        is_sentinel3_olci_dataset,
    )

    group = zarr.open_group(str(s3_olci_group_example), mode="r")
    assert is_sentinel3_olci_dataset(group) is True


def test_real_olci_product_validates_model(s3_olci_group_example) -> None:
    import zarr
    from eopf_geozarr.pyz.v2 import GroupSpec as PyzGroupSpec
    from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot

    group = zarr.open_group(str(s3_olci_group_example), mode="r")
    model = Sentinel3OlciRoot.model_validate(PyzGroupSpec.from_zarr(group).model_dump())
    assert "oa01_radiance" in model.measurements.members
```

- [ ] **Step 4: Run tests + type/lint**

Run: `uv run pytest tests/test_data_api/test_s3_olci.py -v && uv run --frozen pyright tests/conftest.py && uv run ruff check tests/conftest.py tests/test_data_api/test_s3_olci.py`
Expected: tests PASS; pyright 0; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add tests/_test_data/s3_examples/ tests/conftest.py tests/test_data_api/test_s3_olci.py
git commit -m "test(s3-olci): real OLCI product fixture + round-trip detection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Golden-file snapshot of converted OLCI structure

**Files:**
- Create: `tests/_test_data/optimized_olci_examples/<product>.json`
- Test: `tests/test_olci_multiscale.py` (extend) or `tests/test_olci_integration.py`

**Interfaces:**
- Consumes: `s3_olci_group_example` (Task 8); `convert_olci_optimized`; `pydantic_zarr.v3.GroupSpec`.
- Produces: a committed expected-structure snapshot + a test comparing converted output to it (mirrors `test_s2_multiscale.test_create_multiscale_from_datatree`).

- [ ] **Step 1: Write the snapshot comparison test**

```python
# append to tests/test_olci_integration.py
import json
from pathlib import Path

import zarr
from pydantic_zarr.v3 import GroupSpec
from pydantic_zarr.core import tuplify_json


def test_olci_conversion_matches_snapshot(s3_olci_group_example, tmp_path) -> None:
    import xarray as xr

    dt_in = xr.open_datatree(str(s3_olci_group_example), engine="zarr", chunks={})
    out = str(tmp_path / "out.zarr")
    convert_olci_optimized(dt_in, output_path=out, min_dimension=256)

    observed = GroupSpec.from_zarr(zarr.open_group(out, use_consolidated=False)).model_dump()
    expected_path = Path("tests/_test_data/optimized_olci_examples") / (
        Path(str(s3_olci_group_example)).stem + ".json"
    )
    # To (re)generate the snapshot, uncomment:
    # expected_path.parent.mkdir(parents=True, exist_ok=True)
    # expected_path.write_text(json.dumps(observed, indent=2, sort_keys=True))
    expected = tuplify_json(json.loads(expected_path.read_text()))
    observed_flat = GroupSpec(**tuplify_json(observed)).to_flat()
    expected_flat = GroupSpec(**expected).to_flat()
    assert set(observed_flat) == set(expected_flat)
    assert [k for k in observed_flat if observed_flat[k] != expected_flat[k]] == []
```

- [ ] **Step 2: Generate the snapshot**

Temporarily uncomment the regeneration lines, run the test once to write the snapshot, re-comment, and verify it now passes:
```bash
uv run pytest tests/test_olci_integration.py::test_olci_conversion_matches_snapshot -v
```
Expected: PASS after the snapshot is written.

- [ ] **Step 3: Run tests + lint**

Run: `uv run pytest tests/test_olci_integration.py -v && uv run ruff check tests/test_olci_integration.py`
Expected: PASS; ruff clean.

- [ ] **Step 4: Commit**

```bash
git add tests/_test_data/optimized_olci_examples/ tests/test_olci_integration.py
git commit -m "test(s3-olci): golden-file snapshot of converted OLCI structure

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Full verification + docs

**Files:**
- Modify: `README.md` and/or `docs/` (add OLCI to supported products + CLI usage).

- [ ] **Step 1: Whole-suite type + lint + tests**

Run:
```bash
uv run --frozen pyright
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/ -p no:cacheprovider -q -m "not network"
```
Expected: pyright 0 errors; ruff clean; tests green.

- [ ] **Step 2: Document OLCI support**

Add a short section to `README.md` (and/or `docs/converter.md`) noting Sentinel-3 OLCI L1 EFR support, the native-swath (no reprojection) behavior, and the `convert-s3-olci-optimized` command + auto-detection.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/
git commit -m "docs(s3-olci): document Sentinel-3 OLCI export support

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes / deferred (future specs)

- GeoZarr-converting `conditions/geometry` (tie-point grid), `meteorology` (3-D + pressure_level), `instrument` (per-band/detector) — copied through unmodified in v1.
- Reprojection-to-regular-grid option (lossy) — explicitly out of scope.
- SLSTR / SRAL / SYNERGY product types — separate specs.
- If GeoZarr's spatial/proj convention for curvilinear (2-D-coordinate) data needs a specific representation beyond `spatial:dimensions` + coordinate arrays, refine `swath_spatial_attrs` (Task 5) and regenerate the Task 9 snapshot.
