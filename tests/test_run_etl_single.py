from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from connects_common_connectivity.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def single_runner(monkeypatch, tmp_path):
    """Load the single runner with isolated configuration and no notebook execution."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_etl_single", REPO_ROOT / "scripts/run_etl_single.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "get_settings", lambda: Settings(output_root=tmp_path / "release"),
    )
    return module


def test_single_runner_dry_run_has_no_side_effects(single_runner, monkeypatch, tmp_path, capsys):
    """Dry-run selects an existing V1DD notebook and suffixed root without writing or executing."""
    run = Mock()
    monkeypatch.setattr(single_runner, "run_notebooks", run)
    output = tmp_path / "execution"

    assert single_runner.main(["--dry-run", "--output-dir", str(output)]) == 0

    run.assert_not_called()
    assert not output.exists()
    assert "CCC_OUTPUT_ROOT" not in os.environ
    text = capsys.readouterr().out
    assert f"CCC data destination: {tmp_path / 'release_v1dd'}" in text
    assert single_runner.NOTEBOOKS == ("examples/ETL/EM/etl_v1dd1196_01.ipynb",)
    assert (REPO_ROOT / single_runner.NOTEBOOKS[0]).is_file()
    assert single_runner.NOTEBOOKS[0] in text


@pytest.mark.parametrize("suffix", ["_custom", ""])
@pytest.mark.parametrize("manual", [False, True])
def test_single_runner_configurable_suffix(
    single_runner, monkeypatch, tmp_path, capsys, suffix, manual,
):
    """The configured suffix controls default paths while an explicit data destination wins."""
    monkeypatch.setattr(single_runner, "OUTPUT_SUFFIX", suffix)
    run = Mock()
    monkeypatch.setattr(single_runner, "run_notebooks", run)
    args = ["--dry-run"]
    data_root = tmp_path / f"release{suffix}"
    if manual:
        data_root = tmp_path / "manual"
        args += ["--data-output-root", str(data_root)]

    assert single_runner.main(args) == 0

    run.assert_not_called()
    text = capsys.readouterr().out
    assert f"CCC data destination: {data_root}\n" in text
    execution_root = single_runner.REPO_ROOT / "scratch" / "etl_runs"
    assert re.search(
        rf"Executed notebooks: {re.escape(str(execution_root))}/"
        rf"\d{{8}}T\d{{6}}_\d{{6}}Z{re.escape(suffix)}\n",
        text,
    )


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_single_runner_overrides_and_restores_environment(
    single_runner, monkeypatch, tmp_path, manual, fails,
):
    """Default or manual roots reach the runner, and success or failure restores the environment."""
    previous = str(tmp_path / "previous")
    monkeypatch.setenv("CCC_OUTPUT_ROOT", previous)
    data_root = tmp_path / ("manual" if manual else "release_v1dd")
    output = tmp_path / "execution"
    calls = []

    def run(repo_root, output_dir, notebooks):
        assert os.environ["CCC_OUTPUT_ROOT"] == str(data_root)
        calls.append((repo_root, output_dir, notebooks))
        if fails:
            raise RuntimeError("execution failed")

    monkeypatch.setattr(single_runner, "run_notebooks", run)
    args = ["--output-dir", str(output)]
    if manual:
        args += ["--data-output-root", str(data_root)]
    if fails:
        with pytest.raises(RuntimeError, match="execution failed"):
            single_runner.main(args)
    else:
        assert single_runner.main(args) == 0
    assert calls == [(single_runner.REPO_ROOT, output, single_runner.NOTEBOOKS)]
    assert os.environ["CCC_OUTPUT_ROOT"] == previous
