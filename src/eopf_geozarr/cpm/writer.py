"""CPM writer plugin: the ``geozarr`` engine for eopf-cpm.

Importing this module registers :class:`GeoZarrWriter` in CPM's
``EOWriterRegistry`` under the engine name ``"geozarr"``, which makes it
usable from CPM's conversion machinery::

    import eopf_geozarr.cpm.writer  # registers the engine
    from eopf.store import write_datatree

    write_datatree(dtree, "product.zarr", engine="geozarr")

or, end to end from a SAFE product::

    from eopf.store.convert import convert

    convert(
        "S2B_MSIL1C_....SAFE",
        "product.zarr",
        target_store_kwargs={"engine": "geozarr"},
    )

The engine is registered with ``discoverable_by_target=False`` on purpose:
CPM's target-based discovery matches writers by file extension, the built-in
``cpm_zarr`` writer already claims ``.zarr``, and two discoverable matches
raise ``EOAmbiguousWriterError``. The geozarr engine must always be selected
by name.

This module requires the ``eopf`` package (install with
``eopf-geozarr[cpm]``).
"""

# eopf is an optional dependency (the `cpm` extra, Python >= 3.13 only), so it
# is absent from the default type-checking environment.
# pyright: reportMissingImports=false

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from eopf.exceptions.errors import EOStoreProductAlreadyExistsError
from eopf.store.abstract import EOWriter
from eopf.store.writer_registry import EOWriterRegistry

from eopf_geozarr.conversion import create_geozarr_dataset
from eopf_geozarr.cpm.routing import select_pipeline
from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized

if TYPE_CHECKING:
    from collections.abc import Iterable

    import click
    from xarray import DataTree

log = structlog.get_logger()

#: Engine name used with ``write_datatree(..., engine=ENGINE_NAME)``.
ENGINE_NAME = "geozarr"

_SUPPORTED_MODES = ("w", "w-")

#: Default spatial chunk size per pipeline, matching the eopf-geozarr CLI.
_DEFAULT_SPATIAL_CHUNK = {"s2-optimized": 256, "generic": 4096}


class GeoZarrWriter(EOWriter):
    """
    CPM writer producing the GeoZarr layout defined by eopf-geozarr.

    The writer routes to the Sentinel-2 optimized pipeline (flat sibling
    ``r{N}m`` multiscale groups) or the generic pipeline (nested ``r{2**N}``
    overview groups) based on the product's declared ``product:type``; see
    :func:`eopf_geozarr.cpm.routing.select_pipeline`.

    Output is always Zarr format 3. Consolidated metadata is currently always
    written; ``consolidated=False`` is reserved until consolidation becomes a
    parameter of the underlying pipelines.
    """

    EXTENSIONS = (".zarr",)
    DEFAULT_EXTENSION = ".zarr"
    ALLOWED_MODE = _SUPPORTED_MODES

    def write(
        self,
        dtree: DataTree,
        filename_or_obj: str | Path | Any,
        *,
        mode: str = "w",
        zarr_format: int | None = None,
        consolidated: bool = True,
        s2_optimized: bool | None = None,
        spatial_chunk: int | None = None,
        enable_sharding: bool = False,
        max_retries: int = 3,
        groups: Iterable[str] | None = None,
        crs_groups: Iterable[str] | None = None,
        gcp_group: str | None = None,
        min_dimension: int = 256,
        compression_level: int = 3,
        keep_scale_offset: bool = False,
        experimental_scale_offset_codec: bool = False,
        validate_output: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Write ``dtree`` as a GeoZarr store.

        Parameters
        ----------
        dtree
            Product to write. Any ``xarray.DataTree`` works, including trees
            built in memory by the CPM SAFE reader.
        filename_or_obj
            Output zarr store path. Local paths only; for S3 targets use
            CPM's output staging (``stage_target=True`` / ``--stage-output``),
            which writes locally and uploads.
        mode
            ``"w"`` (replace an existing store) or ``"w-"`` (fail if the
            target exists).
        zarr_format
            Only Zarr format 3 is supported; ``None`` means 3.
        consolidated
            Must be True for now; consolidation is always performed.
        s2_optimized
            Force (True) or suppress (False) the Sentinel-2 optimized
            pipeline; None auto-detects from the product type.
        spatial_chunk
            Spatial chunk size; defaults to 256 (S2 optimized) or 4096
            (generic).
        enable_sharding
            Enable Zarr v3 sharding for spatial dimensions.
        max_retries
            Retries for network operations inside the pipelines.
        groups
            Generic pipeline only: DataTree groups to convert as GeoZarr
            datasets (required on that pipeline).
        crs_groups
            Generic pipeline only: groups that get CRS information added on a
            best-effort basis.
        gcp_group
            Generic pipeline only: group holding ground control points
            (required for Sentinel-1).
        min_dimension
            Generic pipeline only: minimum dimension of the coarsest overview
            level.
        compression_level
            S2 optimized pipeline only: blosc-zstd compression level.
        keep_scale_offset
            S2 optimized pipeline only: preserve CF scale/offset encoding
            instead of decoding to float.
        experimental_scale_offset_codec
            S2 optimized pipeline only: push CF scale/offset into the zarr
            codec pipeline.
        validate_output
            S2 optimized pipeline only: run output validation after writing.
        kwargs
            No other options are supported; unknown keys raise
            ``NotImplementedError``.

        Returns
        -------
        Any
            The DataTree returned by the selected conversion pipeline.
        """
        self._validate_options(
            mode=mode,
            zarr_format=zarr_format,
            consolidated=consolidated,
            **kwargs,
        )
        output_path = self._prepare_target(filename_or_obj, mode=mode)
        pipeline = select_pipeline(dtree, force_s2_optimized=s2_optimized)
        resolved_spatial_chunk = (
            spatial_chunk if spatial_chunk is not None else _DEFAULT_SPATIAL_CHUNK[pipeline]
        )
        log.info(
            "Writing GeoZarr product",
            engine=ENGINE_NAME,
            pipeline=pipeline,
            output_path=output_path,
            spatial_chunk=resolved_spatial_chunk,
        )

        if pipeline == "s2-optimized":
            return convert_s2_optimized(
                dt_input=dtree,
                output_path=output_path,
                enable_sharding=enable_sharding,
                spatial_chunk=resolved_spatial_chunk,
                compression_level=compression_level,
                validate_output=validate_output,
                keep_scale_offset=keep_scale_offset,
                experimental_scale_offset_codec=experimental_scale_offset_codec,
                max_retries=max_retries,
            )

        if groups is None:
            raise ValueError(
                "The generic GeoZarr pipeline requires the 'groups' option naming the "
                "DataTree groups to convert (e.g. groups=['/measurements']). Sentinel-1 "
                "products additionally require 'gcp_group' (e.g. '/conditions/gcp').",
            )
        return create_geozarr_dataset(
            dt_input=dtree,
            groups=list(groups),
            output_path=output_path,
            spatial_chunk=resolved_spatial_chunk,
            min_dimension=min_dimension,
            max_retries=max_retries,
            crs_groups=list(crs_groups) if crs_groups is not None else None,
            gcp_group=gcp_group,
            enable_sharding=enable_sharding,
        )

    def validate_write_options(
        self,
        filename_or_obj: str | Path | Any,
        **kwargs: Any,
    ) -> None:
        """Validate writer options without writing data."""
        super().validate_write_options(filename_or_obj, **kwargs)
        option_names = (
            "mode",
            "zarr_format",
            "consolidated",
            "s2_optimized",
            "spatial_chunk",
            "enable_sharding",
            "max_retries",
            "groups",
            "crs_groups",
            "gcp_group",
            "min_dimension",
            "compression_level",
            "keep_scale_offset",
            "experimental_scale_offset_codec",
            "validate_output",
        )
        unknown = {key: value for key, value in kwargs.items() if key not in option_names}
        self._validate_options(
            mode=kwargs.get("mode", "w"),
            zarr_format=kwargs.get("zarr_format"),
            consolidated=kwargs.get("consolidated", True),
            **unknown,
        )

    def _validate_options(
        self,
        *,
        mode: str,
        zarr_format: int | None,
        consolidated: bool,
        **kwargs: Any,
    ) -> None:
        """Reject unsupported writer options loudly."""
        if kwargs:
            unsupported = ", ".join(sorted(str(key) for key in kwargs))
            raise NotImplementedError(
                f"The {ENGINE_NAME} engine does not support these options: {unsupported}. "
                "S3 credentials are read from the environment, not from storage_options.",
            )
        if mode not in _SUPPORTED_MODES:
            supported = ", ".join(repr(supported_mode) for supported_mode in _SUPPORTED_MODES)
            raise ValueError(
                f"The {ENGINE_NAME} engine only supports mode {supported}; got {mode!r}.",
            )
        if zarr_format not in (None, 3):
            raise NotImplementedError(
                f"The {ENGINE_NAME} engine writes Zarr format 3 exclusively; "
                f"got zarr_format={zarr_format!r}.",
            )
        if consolidated is not True:
            raise NotImplementedError(
                "Consolidated metadata is currently always written by the "
                f"{ENGINE_NAME} engine; consolidated=False is not supported yet.",
            )

    @staticmethod
    def _prepare_target(filename_or_obj: str | Path | Any, *, mode: str) -> str:
        """Resolve the output target to a local path string and apply mode semantics."""
        if not isinstance(filename_or_obj, (str, Path)):
            raise TypeError(
                f"The {ENGINE_NAME} engine expects a str or Path target; "
                f"got {type(filename_or_obj).__name__}.",
            )
        path_str = str(filename_or_obj)
        # CPM's convert() wraps targets in pathlib.Path, which collapses the
        # double slash of URL schemes; catch both spellings.
        if path_str.startswith(("s3:", "gs:", "http:", "https:")):
            raise NotImplementedError(
                f"The {ENGINE_NAME} engine does not write directly to remote targets yet. "
                "Use CPM's output staging (stage_target=True / --stage-output) to write "
                "locally and upload.",
            )
        target = Path(path_str)
        if target.exists():
            if mode == "w-":
                raise EOStoreProductAlreadyExistsError(f"Product already exists in {target}")
            # Matching zarr's mode="w" semantics: replace the existing store
            # entirely rather than overwriting in place, which would leave
            # stale chunks and metadata from the previous product.
            log.info("Removing existing store before write", output_path=path_str)
            shutil.rmtree(target)
        return path_str


def register() -> None:
    """Register the ``geozarr`` engine with CPM's writer registry (idempotent)."""
    if not EOWriterRegistry.contains(ENGINE_NAME):
        EOWriterRegistry.register(ENGINE_NAME, discoverable_by_target=False)(GeoZarrWriter)


def get_cli_command() -> click.Command:
    """
    Build the ``eopf convert-geozarr`` click command.

    Exposed to CPM's CLI through the ``eopf.cli`` entry-point group, so the
    command shows up in ``eopf --help`` whenever eopf-geozarr is installed
    next to eopf-cpm.
    """
    import click
    from eopf.store.convert import convert

    @click.command(
        name="convert-geozarr",
        help="Convert a product to GeoZarr using the eopf-geozarr engine.",
    )
    @click.argument("source_path", type=click.Path())
    @click.argument("target_path", type=click.Path())
    @click.option("--spatial-chunk", type=int, default=None, help="Spatial chunk size.")
    @click.option("--enable-sharding", is_flag=True, help="Enable Zarr v3 sharding.")
    @click.option(
        "--groups",
        type=str,
        multiple=True,
        help="Generic pipeline: DataTree groups to convert (repeatable).",
    )
    @click.option(
        "--crs-groups",
        type=str,
        multiple=True,
        help="Generic pipeline: groups that get CRS information added (repeatable).",
    )
    @click.option(
        "--gcp-group",
        type=str,
        default=None,
        help="Generic pipeline: group holding ground control points (Sentinel-1).",
    )
    @click.option(
        "--min-dimension",
        type=int,
        default=256,
        help="Generic pipeline: minimum dimension of the coarsest overview level.",
    )
    @click.option(
        "--no-s2-optimized",
        is_flag=True,
        help="Force the generic pipeline for Sentinel-2 inputs.",
    )
    @click.option(
        "--stage-source",
        is_flag=True,
        help="Download the source product to a local temporary folder before converting.",
    )
    @click.option(
        "--stage-output",
        is_flag=True,
        help="Write locally first, then upload to the target path (required for S3 targets).",
    )
    def convert_geozarr(
        source_path: str,
        target_path: str,
        spatial_chunk: int | None,
        enable_sharding: bool,
        groups: tuple[str, ...],
        crs_groups: tuple[str, ...],
        gcp_group: str | None,
        min_dimension: int,
        no_s2_optimized: bool,
        stage_source: bool,
        stage_output: bool,
    ) -> None:
        register()
        target_store_kwargs: dict[str, Any] = {"engine": ENGINE_NAME}
        if spatial_chunk is not None:
            target_store_kwargs["spatial_chunk"] = spatial_chunk
        if enable_sharding:
            target_store_kwargs["enable_sharding"] = True
        if groups:
            target_store_kwargs["groups"] = list(groups)
        if crs_groups:
            target_store_kwargs["crs_groups"] = list(crs_groups)
        if gcp_group is not None:
            target_store_kwargs["gcp_group"] = gcp_group
        if min_dimension != 256:
            target_store_kwargs["min_dimension"] = min_dimension
        if no_s2_optimized:
            target_store_kwargs["s2_optimized"] = False
        convert(
            source_path=source_path,
            target_path=target_path,
            target_store_kwargs=target_store_kwargs,
            stage_source=stage_source,
            stage_target=stage_output,
        )

    return convert_geozarr


register()
