---
description: "Use when editing CHANGELOG.md, drafting release notes, or summarizing user-visible changes. Enforces Keep a Changelog format, SemVer scope, and the user-voice rule."
applyTo: "CHANGELOG.md"
---
# Changelog rules

The changelog is the user-facing log of what changed in
`connects_common_connectivity`. It is **not** an internal work journal.

## Format
- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) +
  [SemVer](https://semver.org/spec/v2.0.0.html).
- All new entries go under `## [Unreleased]` until a release is cut.
- Use only the standard sections: `Added`, `Changed`, `Deprecated`, `Removed`,
  `Fixed`, `Security`. Omit empty sections in released versions; keep them as
  empty headers under `[Unreleased]` so contributors see the slots.
- Newest version on top. Releases are `## [X.Y.Z] - YYYY-MM-DD`.

## Voice and scope (the rule that actually matters)
- Write in **user voice**: what changed for someone who imports
  `connects_common_connectivity`, runs the `ccc` CLI, or follows the README.
- One bullet per change. Past tense, present-perfect-style is fine
  (`Added …`, `Moved …`, `Fixed …`). No first person, no narrative.
- **Include**: new public names, removed public names, moved import paths,
  changed signatures, changed defaults, behavior fixes a user could observe,
  new CLI flags, new config keys, dropped Python versions.
- **Exclude**: internal refactors, test-only changes, planning-doc edits,
  prompt/agent-customization edits, dev-tooling tweaks, comment-only changes.
  If a user couldn't notice it, it doesn't belong here.
- If a change has both an internal and a user-visible side, log only the
  user-visible side.

## Linking
- Reference public names in backticks: `` `write_models` ``, `` `io.writers` ``.
- Link to issues/PRs only when they add information a user would want
  (`#123`); do not link to internal planning docs.

## Deprecations and removals
- Announce in `Deprecated` first (one release minimum) before moving to
  `Removed`, except for genuinely unused or never-released names.
- Name the replacement when there is one: "Deprecated `X`; use `Y` instead."

## Releasing (manual for now)
1. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (today's date).
2. Drop empty subsections from the released block.
3. Add a fresh `## [Unreleased]` at the top with all six empty sub-headers.
4. Bump the version in `pyproject.toml` in the same commit.

## Anti-patterns
- "Refactored internals." — internal, drop it.
- "Updated planning docs." — internal, drop it.
- "Various fixes." — split into specific bullets or drop.
- "Added new feature." — name the public symbol or describe the behavior.
- Long prose paragraphs — one bullet, one change.
