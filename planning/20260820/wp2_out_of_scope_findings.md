# WP2 out-of-scope findings

These items were identified while implementing merge-scoped writes. They are not
required to close the WP2 milestone and were not folded into its implementation.

## Concurrent commit retry policy

Delta MERGE makes each upsert atomic, preventing the read-union-rewrite data-loss
pattern addressed by WP2. Two writers that commit to the same Delta table at the
same instant can still encounter an optimistic-concurrency conflict. WP2 does not
add retries or distributed concurrency tests.

Future decision: define whether the IO layer should expose conflicts directly or
retry selected conflict types with bounded backoff. Cover the policy against the
object-store backend used in deployment, not only a local filesystem.

## Repository-wide Ruff backlog

The WP2-touched Python files pass targeted Ruff checks. A repository-wide
`uv run ruff check src tests` still reports 251 existing diagnostics, primarily
line-length and import-order violations. Ruff also warns that the top-level
linter settings in `pyproject.toml` are deprecated in favor of `[tool.ruff.lint]`.

Future work: create a dedicated formatting/lint cleanup issue so these unrelated
changes do not obscure the WP2 behavioral review.

## Development-environment dependency warning

The full test suite passes, but importing `requests` emits a
`RequestsDependencyWarning` because the installed `urllib3`, `chardet`, or
`charset_normalizer` versions do not match its supported range. This is unrelated
to the WP2 writer behavior and was not addressed here.

Future work: reconcile the environment lock and installed packages so test runs do
not rely on an unsupported requests dependency combination.