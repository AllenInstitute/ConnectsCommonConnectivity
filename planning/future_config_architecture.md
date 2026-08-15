# Handoff 06 — Future configuration architecture

## Context

This repository currently combines schema development, an IO package, and executable ETL
notebooks. In that combined role, the tracked `ccc_config.yaml` is repository-owned workflow
configuration: it gives the notebooks a shared default output namespace whose changes remain
visible in Git history. Machine-specific destinations can use `CCC_OUTPUT_ROOT` or explicit
writer arguments instead of changing that tracked default.

The longer-term architecture will separate the schemas, IO implementation, and demonstration
notebooks into different repositories. Configuration ownership should be redesigned as part of
that split rather than inherited accidentally from the current monorepo.

## Working agreement

This handoff covers future configuration design and migration. It does not require untracking or
renaming the current `ccc_config.yaml`, changing the current resolution behavior, or preparing
responses to review comments. Preserve the current tracked project configuration until a package
split or an explicitly approved configuration migration begins.

## Intended ownership after the repository split

### Schema package

- Own schema definitions, generated model contracts, and schema-version metadata.
- Do not own output paths, local filesystem discovery, or writer runtime settings.
- Expose enough version information for IO and dataset manifests to record which schema produced
  an output release.

### IO package

- Own the typed settings model, YAML loading and validation, configuration precedence, and path
  normalization.
- Provide a safe config-initialization API and a corresponding CLI command.
- Ship an example or template config as package data so documentation and generated files cannot
  drift independently.
- Remain usable without a notebook repository by accepting explicit settings and environment
  overrides.

### Demo notebook repository

- Own configuration that defines a canonical, reproducible notebook workflow.
- Track a populated config only when its values are genuinely project-owned defaults that users
  are not expected to edit.
- Otherwise track an example config and ignore the generated user config.
- Own dataset manifests or release metadata describing the outputs created by the notebooks.

## Configuration artifacts

Plan for three distinct concepts rather than asking one file to serve all of them:

1. **Packaged template:** an IO-package resource containing documented, valid defaults and all
   supported keys. It is the source used by the initialization helper.
2. **Project or user config:** a validated YAML file consumed at runtime. Whether it is tracked is
   decided by the repository that owns it: canonical workflow configs may be tracked; local
   configs should be ignored.
3. **Dataset manifest:** tracked provenance describing a produced dataset or release. It should
   carry dataset version, schema version, source versions, generation metadata, and logical
   assets. An output directory name alone is not sufficient provenance.

## Initialization API and CLI

Design a small IO-package API, with a CLI as the normal user-facing entry point:

```python
def create_config(
    path: Path,
    *,
    output_root: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Path:
    ...
```

```text
ccc init-config [PATH] --output-root PATH [--dry-run] [--force]
```

The final contract should:

- generate YAML through structured serialization rather than string substitution;
- validate generated content with the same `Settings` model used during loading;
- refuse to overwrite an existing file unless the caller explicitly opts in;
- create deterministic output suitable for documentation and snapshot tests;
- return or print the created path and the next command needed to use it;
- avoid guessing a machine-specific output location when none was supplied; and
- keep the Python helper independent of CLI prompting so notebooks and other tools can call it.

Decide whether the packaged template is a literal YAML resource read by `create_config` or is
rendered from typed defaults. Whichever representation is selected must have a drift test against
`Settings`.

## Discovery and precedence decisions

Retain explicit dependency injection as the highest-priority path. Before extraction, decide and
document the remaining precedence as one stable contract, including whether the installed IO
package needs an explicit config-path variable in addition to walk-up discovery. A candidate is:

1. Explicit `Settings` or per-call writer arguments.
2. Explicit config path such as `CCC_CONFIG`, if introduced.
3. Field-specific environment overrides such as `CCC_OUTPUT_ROOT`.
4. A discovered project config.
5. A clear error with the `ccc init-config` command.

Clarify whether environment fields merge onto file settings or replace the complete settings
object, how relative paths anchor for explicit and discovered configs, and how cached settings are
reset in long-running notebook sessions.

## Provenance contract

Move dataset identity out of `output_root` before treating the IO package as a general-purpose
library. Extend or formalize the existing dataset-manifest example to record at least:

- dataset identifier and release version;
- schema package name and version;
- IO package name and version;
- source dataset or upstream release versions;
- generation timestamp and, where available, source Git commit;
- logical assets, formats, and relative paths; and
- integrity hashes or another immutable-content identifier when releases become publishable.

The output root remains an operational destination. It may include a readable version label, but
that label must not be the only record connecting generated data to source and schema versions.

## Migration stages

1. **Stabilize the current contract.** Keep the tracked repository config, treat it as the
   notebook workflow default, and direct machine-specific overrides through the existing public
   mechanisms.
2. **Formalize provenance.** Turn the dataset-manifest example into an agreed release contract and
   record output versions independently of filesystem location.
3. **Extract the IO package.** Move `Settings`, config loading, template package data,
   `create_config`, and `ccc init-config` together. Preserve explicit settings support so callers
   do not depend on discovery.
4. **Extract the demo notebooks.** Choose tracked canonical config versus tracked example plus
   ignored local config based on whether notebook users should edit the values. Generate local
   config through the IO CLI when appropriate.
5. **Extract the schema package.** Remove runtime IO configuration from its dependency surface and
   expose only schema/model version information needed by manifests.
6. **Deprecate compatibility behavior deliberately.** If config filenames, discovery, or
   precedence change, provide warnings and a documented migration window rather than silently
   selecting a different destination.

## Relevant current files

- `ccc_config.yaml`
- `src/connects_common_connectivity/config.py`
- `src/connects_common_connectivity/io/__init__.py`
- `src/connects_common_connectivity/io/writers.py`
- `tests/test_config.py`
- `tests/conftest.py`
- `examples/dataset_manifest_example.yaml`
- `README.md`
- `etl_example_prompt.md`
- `planning/20260623/ARCHITECTURE.md`
- `planning/handoff_04_externalize_user_config.md`

## Tests to plan

- Template or generated config validates against `Settings`.
- Initialization creates the requested file with deterministic YAML.
- Initialization refuses accidental overwrite and permits explicit overwrite.
- CLI and Python API produce equivalent settings.
- Explicit settings, explicit config path, environment overrides, and discovery follow the agreed
  precedence.
- Relative paths anchor consistently for every loading route.
- Missing-config errors point to the initialization command.
- Packaged template remains available from an installed wheel, not only a source checkout.
- Manifest validation rejects missing version and provenance fields once a formal schema exists.

## Documentation to plan

- IO-package configuration reference covering every setting, precedence, path anchoring, caching,
  environment variables, and initialization.
- Demo-repository setup instructions that state whether its config is project-owned or user-owned.
- A short migration guide for users moving from the current monorepo config.
- Dataset-release documentation that distinguishes runtime destination from durable provenance.

## Planning outcome

Before implementation, produce an architecture decision record that fixes artifact ownership,
the initialization API and CLI, discovery precedence, template packaging, and the manifest
contract. Then prepare separate implementation plans for the IO-package extraction and demo-repo
migration, with compatibility tests and release sequencing for each repository.