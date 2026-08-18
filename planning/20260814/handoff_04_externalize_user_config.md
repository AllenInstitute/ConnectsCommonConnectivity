# Handoff 04 — Externalize user configuration

## Context

The repository tracks `ccc_config.yaml`, but its `output_root` is expected to be edited for a
user's local environment. The reviewer warned that this creates dirty worktrees and merge or
pull conflicts, while the author noted a separate desire to retain output-data version history.

## Working agreement

This is a prompt for planning, not an implementation plan. Treat configuration ownership and
data-version provenance as a design decision to make with the user; inspect all current config
consumers before proposing file changes, and do not edit or untrack files until the user agrees.

## Relevant files

- `ccc_config.yaml`
- `.gitignore`
- `README.md`
- `CHANGELOG.md`
- `.github/instructions/changelog.instructions.md`
- `src/connects_common_connectivity/config.py`
- `src/connects_common_connectivity/io/writers.py`
- `tests/test_config.py`
- `tests/conftest.py`
- `etl_example_prompt.md`
- `planning/20260623/ARCHITECTURE.md`
- `planning/20260623/prompts/01_config.md`
- `planning/20260623/prompts/06_notebook_migration.md`

## Reviewer comment

- [Do not track the user-edited config; document how to create it](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483932698)

## Tasks to plan with the user

1. **Decide which configuration artifacts are repository-owned.** Compare a tracked
   `ccc_config.example.yaml` plus ignored `ccc_config.yaml`, an environment-only setup, and any
   other local/project split that fits notebook discovery. The motivation is to prevent normal
   user configuration from appearing as source-code changes.
2. **Separate output location from data-version provenance.** Determine what the author wants
   to recover from Git history and whether that belongs in a dataset manifest, release metadata,
   or another stable file rather than a local filesystem path. The motivation is to preserve
   provenance without coupling it to machine-specific configuration.
3. **Audit discovery and first-run behavior.** Trace `find_config_file`, `get_settings`,
   `CCC_OUTPUT_ROOT`, explicit `settings=`, and explicit `output_root=` across scripts and
   notebooks. The motivation is to ensure a fresh clone remains understandable and usable once
   the populated config is no longer tracked.
4. **Plan documentation changes.** Specify README instructions for creating the local config,
   choosing relative or absolute paths, using the environment override, and understanding
   precedence and errors. The motivation is to replace implicit repository setup with an
   explicit onboarding contract.
5. **Plan repository and test changes.** Identify the exact rename/add/ignore operations,
   tests that must use temporary configs, notebook or planning references that need correction,
   and any user-visible changelog entry. The motivation is to avoid leaving documentation,
   tests, or discovery behavior dependent on a tracked local config.

## Planning outcome

Present the viable ownership options and a recommendation, including provenance tradeoffs,
migration steps, affected references, and validation commands. Stop for user approval before
changing tracked configuration or documentation.