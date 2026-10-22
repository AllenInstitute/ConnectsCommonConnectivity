from __future__ import annotations

import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    """Import the runner without entering its CLI or starting any notebook kernels."""
    spec = importlib.util.spec_from_file_location("run_etl", REPO_ROOT / "scripts/run_etl.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_notebook(path: Path) -> bytes:
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(
            "print('fixture')",
            execution_count=9,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="stale output")],
        ),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path.read_bytes()


def test_runner_covers_active_etls_in_dependency_order(runner):
    """The plan includes all ten active ETLs and places shared taxonomies before cohorts."""
    active = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "code").rglob("etl_*.ipynb")
        if "old" not in path.parts and ".ipynb_checkpoints" not in path.parts
    }
    assert set(runner.NOTEBOOKS) == active
    assert len(runner.NOTEBOOKS) == len(active) == 10
    order = {Path(name).name: index for index, name in enumerate(runner.NOTEBOOKS)}
    for cohort in (
        "etl_patchseq_inh_01.ipynb", "etl_patchseq_exc_01.ipynb", "etl_wnm_exc_01.ipynb",
    ):
        assert order["etl_ttype_01.ipynb"] < order[cohort]
        assert order["etl_met_type_01.ipynb"] < order[cohort]
    for before, after in (
        ("etl_ttype_01.ipynb", "etl_barseq_01.ipynb"),
        ("etl_abc_01.ipynb", "etl_barseq_01.ipynb"),
        ("etl_patchseq_exc_01.ipynb", "etl_wnm_exc_01.ipynb"),
        ("etl_minnie1822_01.ipynb", "etl_minnie1822_02_umap_features.ipynb"),
    ):
        assert order[before] < order[after]


@pytest.mark.parametrize("fail_first", [False, True])
def test_runner_saves_outputs_without_modifying_sources(
    runner, tmp_path, monkeypatch, fail_first,
):
    """Success and failure save output and collect runner memory before any next notebook."""
    notebooks = ("code/first.ipynb", "code/second.ipynb")
    originals = {name: _write_notebook(tmp_path / name) for name in notebooks}
    output = tmp_path / "executed"
    clients = []
    collect = Mock()
    monkeypatch.setattr(runner.gc, "collect", collect)

    def fake_client(notebook, **options):
        assert collect.call_count == len(clients)
        assert options == {
            "kernel_name": "python3", "timeout": None,
            "allow_errors": False, "force_raise_errors": True,
        }
        assert notebook.cells[0].outputs == []
        assert notebook.cells[0].execution_count is None
        spec = SimpleNamespace(argv=["another-python"])

        def execute(**kwargs):
            assert kwargs == {"cwd": str(tmp_path), "cleanup_kc": True}
            assert spec.argv == [
                sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
            ]
            notebook.cells[0].outputs = [
                nbformat.v4.new_output("stream", name="stdout", text="fresh output"),
            ]
            if fail_first:
                raise RuntimeError("simulated cell failure")

        client = SimpleNamespace(
            create_kernel_manager=lambda: SimpleNamespace(kernel_spec=spec),
            execute=execute,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(runner, "NotebookClient", fake_client)
    if fail_first:
        with pytest.raises(RuntimeError, match="simulated cell failure"):
            runner.run_notebooks(tmp_path, output, notebooks)
    else:
        runner.run_notebooks(tmp_path, output, notebooks)
    assert len(clients) == (1 if fail_first else 2)
    assert collect.call_count == len(clients)
    for name, original in originals.items():
        assert (tmp_path / name).read_bytes() == original
    saved = nbformat.read(output / notebooks[0], as_version=4)
    assert saved.cells[0].outputs[0].text == "fresh output"
    assert (output / notebooks[1]).exists() is not fail_first


def test_runner_releases_client_cycles_between_notebooks(runner, tmp_path, monkeypatch):
    """Client cycles and notebook references are collected before the next client is created."""
    notebooks = ("first.ipynb", "second.ipynb")
    for name in notebooks:
        _write_notebook(tmp_path / name)
    references = []

    class TrackingClient:
        def __init__(self, notebook, **options):
            assert all(reference() is None for reference in references)
            self.notebook = notebook
            self.cycle = self
            references.append(weakref.ref(self))

        def create_kernel_manager(self):
            return SimpleNamespace(kernel_spec=SimpleNamespace(argv=[]))

        def execute(self, **kwargs):
            assert kwargs["cleanup_kc"] is True

    monkeypatch.setattr(runner, "NotebookClient", TrackingClient)
    runner.run_notebooks(tmp_path, tmp_path / "executed", notebooks)
    assert len(references) == 2
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("fails", [False, True])
def test_runner_shuts_down_real_kernel(runner, tmp_path, monkeypatch, fails):
    """A real kernel is shut down after either successful execution or a cell error."""
    source = tmp_path / "notebook.ipynb"
    code = "raise ValueError('intentional failure')" if fails else "print('done')"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), source,
    )
    managers = []
    client_class = runner.NotebookClient

    class TrackingClient(client_class):
        def create_kernel_manager(self):
            manager = super().create_kernel_manager()
            managers.append(manager)
            return manager

    monkeypatch.setattr(runner, "NotebookClient", TrackingClient)
    if fails:
        from nbclient.exceptions import CellExecutionError

        with pytest.raises(CellExecutionError, match="intentional failure"):
            runner.run_notebooks(tmp_path, tmp_path / "executed", ("notebook.ipynb",))
    else:
        runner.run_notebooks(tmp_path, tmp_path / "executed", ("notebook.ipynb",))
    assert managers and all(not manager.has_kernel for manager in managers)
    assert (tmp_path / "executed/notebook.ipynb").is_file()


def test_runner_preflights_missing_notebooks_before_execution(runner, tmp_path, monkeypatch):
    """A missing later notebook prevents all execution and output-directory creation."""
    _write_notebook(tmp_path / "first.ipynb")
    client = Mock()
    monkeypatch.setattr(runner, "NotebookClient", client)
    with pytest.raises(FileNotFoundError, match="missing.ipynb"):
        runner.run_notebooks(
            tmp_path, tmp_path / "outputs", ("first.ipynb", "missing.ipynb"),
        )
    client.assert_not_called()
    assert not (tmp_path / "outputs").exists()


def test_runner_rejects_existing_output_directory(runner, tmp_path, monkeypatch):
    """An existing output directory is rejected without overwriting records or starting kernels."""
    _write_notebook(tmp_path / "first.ipynb")
    client = Mock()
    monkeypatch.setattr(runner, "NotebookClient", client)
    with pytest.raises(FileExistsError):
        runner.run_notebooks(tmp_path, tmp_path, ("first.ipynb",))
    client.assert_not_called()


@pytest.mark.parametrize(
    "start",
    ["etl_wnm_exc_01.ipynb", "code/EXC_WNM_patchseq/etl_wnm_exc_01.ipynb"],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_from_selects_inclusive_suffix(runner, tmp_path, monkeypatch, capsys, start, dry_run):
    """A filename or relative path selects an inclusive suffix for preview or execution."""
    execute = Mock()
    monkeypatch.setattr(runner, "run_notebooks", execute)
    output = tmp_path / "executed"
    args = ["--from", start, "--output-dir", str(output)]
    if dry_run:
        args.append("--dry-run")

    assert runner.main(args) == 0

    selected = runner.NOTEBOOKS[6:]
    printed = capsys.readouterr().out
    assert all(name in printed for name in selected)
    assert all(name not in printed for name in runner.NOTEBOOKS[:6])
    if dry_run:
        execute.assert_not_called()
    else:
        execute.assert_called_once_with(runner.REPO_ROOT, output, selected)
    assert not output.exists()


@pytest.mark.parametrize(
    "start",
    ["missing.ipynb", "code/old/etl_wnm_exc_01.ipynb"],
)
def test_from_rejects_unknown_notebooks(runner, tmp_path, monkeypatch, start):
    """An unknown start notebook fails before execution or output creation."""
    execute = Mock()
    monkeypatch.setattr(runner, "run_notebooks", execute)
    output = tmp_path / "executed"
    with pytest.raises(SystemExit) as error:
        runner.main(["--from", start, "--output-dir", str(output)])
    assert error.value.code == 2
    execute.assert_not_called()
    assert not output.exists()


def test_without_from_runs_complete_plan(runner, tmp_path, monkeypatch):
    """Omitting --from preserves execution of every active notebook."""
    execute = Mock()
    monkeypatch.setattr(runner, "run_notebooks", execute)
    output = tmp_path / "executed"
    assert runner.main(["--output-dir", str(output)]) == 0
    execute.assert_called_once_with(runner.REPO_ROOT, output, runner.NOTEBOOKS)
