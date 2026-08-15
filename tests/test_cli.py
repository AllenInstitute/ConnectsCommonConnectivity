from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "connects_common_connectivity.cli", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )


def test_cli_help():
    """The CLI help command must succeed and display usage text."""
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_info_shows_version():
    """The CLI info command must report the package version."""
    result = _run_cli("info")
    assert result.returncode == 0
    assert "Package version:" in result.stdout


def test_cli_bundle_happy_path(tmp_path):
    """The bundle command must create an archive containing schemas."""
    out = tmp_path / "connectivity_bundle.tar.gz"
    result = _run_cli("bundle", "--output", str(out), cwd=tmp_path)
    assert result.returncode == 0
    assert out.exists()
    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    assert any(name.startswith("schemas/") for name in names)


def test_cli_bad_subcommand_exits_nonzero():
    """An unknown CLI subcommand must fail with an invalid-choice error."""
    result = _run_cli("not-a-command")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()
