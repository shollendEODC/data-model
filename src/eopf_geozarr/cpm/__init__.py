"""Integration of eopf-geozarr with the EOPF CPM (the ``eopf`` package).

This package provides the ``geozarr`` writer engine for CPM's writer
registry, plus an ``eopf convert-geozarr`` CLI command exposed through the
``eopf.cli`` entry-point group.

Importing :mod:`eopf_geozarr.cpm.writer` (or accessing ``GeoZarrWriter`` /
``register`` here) requires the ``eopf`` package; install it with the
``eopf-geozarr[cpm]`` extra. :mod:`eopf_geozarr.cpm.routing` is importable
without eopf-cpm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import click

    from eopf_geozarr.cpm.writer import GeoZarrWriter, register

#: Engine name used with ``write_datatree(..., engine=ENGINE_NAME)``.
ENGINE_NAME = "geozarr"

__all__ = ["ENGINE_NAME", "GeoZarrWriter", "get_cli_command", "register"]


def get_cli_command() -> click.Command:
    """Build the ``eopf convert-geozarr`` command (entry-point hook for CPM's CLI)."""
    from eopf_geozarr.cpm.writer import get_cli_command as build

    return build()


def __getattr__(name: str) -> Any:
    if name in {"GeoZarrWriter", "register"}:
        from eopf_geozarr.cpm import writer

        return getattr(writer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
