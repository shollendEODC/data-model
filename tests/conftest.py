# tests/conftest.py
"""Shared pytest fixtures for the tests/ suite."""

from __future__ import annotations

import pytest
from test_s2 import S2_STORE_CONFIGS
from utils import TestFiles, cleanup_tmp_root


@pytest.fixture(scope="session", params=sorted(S2_STORE_CONFIGS))
def s2_product_files(request: pytest.FixtureRequest) -> TestFiles:
    """Build (and, for .SAFE/.SEN3 inputs, convert) one S2 product's files.

    Session-scoped + parametrized: pytest builds and caches one instance per
    `request.param` ("L2A", "L1C", ...) for the whole session, so every test
    needing that pair reuses the same converted store instead of re-converting
    per test. Conversion itself (and its tmp dir) is managed by TestFiles /
    utils._tmp_root - this fixture just supplies the config per pair.
    """
    cfg = S2_STORE_CONFIGS[request.param]
    return TestFiles(
        sensor="S2",
        mode=request.param,
        ref_input_path=cfg["ref_input_path"],
        geozarr_path=cfg["geozarr_path"],
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Remove every SAFE/SEN3 -> zarr conversion this session produced.

    Deliberately a pytest_sessionfinish hook rather than an autouse fixture:
    the conversions happen inside pytest_generate_tests (a collection-time
    hook), which runs even for `--collect-only` or a run that errors before
    any test executes - an autouse fixture's teardown would never fire in
    those cases (fixtures only set up/tear down around actual test runs),
    silently leaving multi-hundred-MB converted stores behind in /tmp.
    pytest_sessionfinish runs at the end of every pytest process regardless.
    """
    cleanup_tmp_root()
