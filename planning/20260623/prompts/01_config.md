# Agent prompt — Config module (discovered config file)

> Prepend `00_shared_context.md`.

## Goal
Create `src/connects_common_connectivity/config.py` — at the **package root**, next to
`models.py` and `cli.py`, NOT in `io/`. Configuration is package-wide (cli and future
plotting/analysis read it too), so the general name belongs in the general namespace.
Settings live in **one declarative, version-controlled file** (`ccc_config.yaml`) that every
entry point discovers automatically — no per-notebook setup, no `%run`, no process-global
mutation. The library holds the *mechanism* and validates the file via pydantic; the
*values* live in `ccc_config.yaml` at the repo root. **No new dependency** — plain pydantic
`BaseModel` + PyYAML (already in the tree via LinkML).

## Requirements
1. A `Settings(BaseModel)`:
   - `output_root: Path` (required, no default).
   - `dry_run: bool = False`, and room for more knobs (`schema_version_pin`, ...) later.
   - **No built-in default output path.** The value comes from the config file.
   - `describe()` / `__repr__` printing the resolved config.
2. **File discovery + typed load (the key piece):**
   - `find_config_file(start: Path | None = None) -> Path | None` walks up from `cwd`
     (or `start`) to the filesystem root looking for `ccc_config.yaml` — same pattern as
     `pyproject.toml`/`ruff`/`pytest`. This is what lets a notebook in `code/` find the
     repo-root config with zero config code.
   - `get_settings() -> Settings` (cache with `functools.lru_cache`):
     1. find `ccc_config.yaml`; if none found, **raise a clear, actionable error**
        (`"No ccc_config.yaml found — create one at the repo root with output_root: ..."`).
     2. `yaml.safe_load` it and construct `Settings(**data)` (pydantic validates here).
     3. **Developer escape hatch:** if `CCC_OUTPUT_ROOT` env is set, override
        `output_root` with it (env wins over the file, for the path only — it cannot
        express other knobs). Document it as override-only, not the primary path.
   - Precedence overall: **explicit `settings=` arg (handled by callers) > `CCC_OUTPUT_ROOT`
     env > `ccc_config.yaml` > error.**
   - Provide a way to clear the cache for tests (e.g. expose `get_settings.cache_clear`).
3. `table_path(settings: Settings, table: str) -> Path` joins `output_root / table` (e.g.
   `"dataset"`, `"dataitem"`, `"dataitem_dataset_association"`) using the exact subdir names
   in the notebooks, so nothing concatenates path strings.
4. Export `Settings`, `get_settings`, `table_path` from `config.py` (and re-exported from
   `io/__init__.py` for convenience). `io/` imports them via `from ..config import ...`.
   Do NOT add a `configure()` process-global setter — discovery replaces it.

## Tests (`tests/test_config.py`)
- `get_settings()` raises the actionable error when no `ccc_config.yaml` is discoverable.
- A `ccc_config.yaml` in a tmp dir is discovered from a nested cwd and loaded/validated.
- `CCC_OUTPUT_ROOT` env overrides only `output_root`; `dry_run` still comes from the file.
- An explicit `settings=` passed to a caller wins over both.
- `table_path` joins correctly and returns a `Path`.

## Do not
- Add a built-in default output path, a `configure()` global, or `%run`-style coupling. Add
  any dependency beyond pydantic + PyYAML. Touch `models.py` or schemas.

## Report
List the subdir names found in the notebooks (for `table_path`) and the `output_root`
value you put in `ccc_config.yaml`.
