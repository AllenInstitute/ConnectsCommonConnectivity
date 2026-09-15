from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import connects_common_connectivity as ccc
from connects_common_connectivity.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Each test gets isolated cwd/env and a fresh get_settings cache."""
    monkeypatch.delenv("CCC_OUTPUT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def models() -> dict:
    """Generate pydantic models once per session (expensive)."""
    return ccc.generate_pydantic_models()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(output_root=tmp_path)


@pytest.fixture
def read_delta():
    def _read(path: str | Path) -> pl.DataFrame:
        return pl.read_delta(str(path))

    return _read
