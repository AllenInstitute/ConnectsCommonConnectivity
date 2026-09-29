---
name: write-pr-message
description: "Draft or update a pull request message using the repository PR template, with a numbered change list, symbol anchors, and verified test evidence. Use when asked to write, revise, shorten, or fact-check a PR description or PR message draft."
argument-hint: "[optional base branch, draft path, or scope note]"
---

# Write PR Message

Produce a reviewer-friendly pull request message that follows
`.github/PULL_REQUEST_TEMPLATE.md` and states only claims verified against the
branch.

## Required Shape

Keep the template's four headings, in order and unrenamed: `## What changed`,
`## Why`, `## How to test`, `## Reviewer focus (optional)`. Delete the template
comments. Add a short lead paragraph above `## What changed`.

**Lead.** Two to four lines: what the branch replaces or adds, the resulting
state, and the base branch to review against. No heading.

**What changed.** A numbered list, one entry per behavior change. Bold the
change in a few words, then anchor it to the code as `file.py::symbol`
(`write_spec.py::REGISTRY`, `writers.py::_dispatch_merge_scoped`). Prefer
`file::symbol` over line numbers, which shift during review; use permalinks only
when pinned to a pushed SHA. One or two sentences of detail at most. Group
tightly related edits into a single numbered item rather than splitting them.

Follow the list with short footnote paragraphs for: code deliberately left in
place that a reviewer may question, and any files in the diff that are unrelated
to the stated scope. Never hide unrelated changes.

Close the section with an issue-to-change table so a reviewer can see which code
closes which issue:

```markdown
| Issue | Closed by |
|---|---|
| Closes #13 — short issue title | 1, 2 |
```

**Why.** Labelled paragraphs (`**Data loss (#15).**`) that reference the
numbered changes rather than restating them: "Change 2 makes each contribution
an atomic upsert." State the problem and its observed symptom, then the fix.
Add a `**Scope limit.**` paragraph whenever a closed issue is only partly
addressed, so the `Closes` lines do not overclaim.

**How to test.** One command block with the actual commands run and their real
results as comments. Then a short paragraph on non-automated verification, and
one paragraph per body of manual or external evidence. Put any reason the
evidence is weaker than it looks in a blockquote callout.

**Reviewer focus.** Five to eight bullets. Each names a symbol, a decision, or a
risk — not a summary of the change. Include open questions and deferred work
with a pointer to where it is tracked.

## Style

- Plain, specific language. Cut adjectives, marketing, and restated diffs.
- Say each thing once. If a rationale belongs in `Why`, do not repeat it in
  `What changed` or `Reviewer focus`.
- Wrap prose at roughly 80 columns so the diff of the draft stays readable.
- No emoji. No Conventional Commit prefixes. No "this PR ..." throat-clearing.
- Aim to halve a first draft. Length is a reviewer cost.

## Procedure

1. Determine the base branch (ask if ambiguous) and read the full branch diff:
   `git diff --stat <base>...HEAD` plus the commit subjects. Cover every file in
   the diff, including unrelated ones.
2. When revising an existing draft, diff the draft's claims against the commits
   made since it was written. Post-review commits are the usual source of stale
   or missing entries.
3. Read the linked milestone and issues when available, and map each `Closes`
   line to the numbered changes that close it.
4. Verify every factual claim before writing it:
   - Run the test and lint commands and quote the real counts. Never carry over
     a previous run's numbers.
   - If a quoted command now fails, fix the underlying problem or change the
     claim. Do not publish a passing claim for a failing command.
   - Trace measurements to their assertion. Quote what a test actually asserts,
     not a remembered ad-hoc experiment.
   - Confirm referenced symbols, paths, config values, and counts still exist.
     Reverted commits leave stale claims behind.
   - For notebooks, check for cells with `execution_count: null` beside stored
     outputs. Those outputs predate the code and cannot be cited as evidence
     without a caveat.
5. Write the draft to the path the user names, or alongside any existing draft
   (for example `planning/<YYYYMMDD>/<scope>_PR_message.md`). Edit the existing
   file in place rather than creating a parallel copy.
6. Report the claims that were corrected, the claims that could not be verified,
   and any evidence that needs a rerun before merge.

## Do Not

- Assert test, lint, or acceptance results that were not observed in this
  session.
- Drop a caveat because it weakens the PR. Flag it and let the reviewer decide.
- Expand line numbers, function signatures, or diff hunks into the message; link
  to the code instead.
