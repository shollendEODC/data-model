#!/usr/bin/env python3
"""
Command-line interface for eopf-geozarr.

This module provides CLI commands for converting EOPF datasets to GeoZarr compliant format.
"""

import structlog

log = structlog.get_logger()


def main() -> None:
    """Execute main entry point for the CLI."""
    # this needs to be redone to act only as an entrypoint for cpm cli with the addition of registering the geozarr cpm writer


if __name__ == "__main__":
    main()
