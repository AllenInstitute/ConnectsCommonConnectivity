---
name: construct-commit-message
description: "Construct a concise, copy-pasteable Git commit message in the user's preferred style without inspecting Git status or diffs. Use when asked for a commit message, commit subject, or copyable message for described changes."
argument-hint: "[change summary]"
---

# Construct Commit Message

Turn the user's supplied change summary or the current conversation's completed
task into one copy-pasteable Git commit subject.

## Message Style

- Use a concise, lowercase, past-tense description.
- Do not use Conventional Commit prefixes such as `feat:`, `fix:`, or `docs:`.
- State what changed first. When useful, add a colon followed by the user-visible behavior, outcome, or reason.
- Prefer concrete domain terms over generic phrases such as "updated files" or "made changes."
- Keep the subject self-contained and omit a trailing period.
- Correct obvious spelling errors without changing the intended meaning.

Preferred shape:

```text
<what changed>: <user-visible behavior, outcome, or reason>
```

Example:

```text
added to changelog user-visible behavior change: rerunning with fewer taxonomy rows no longer deletes omitted rows
```

## Procedure

1. Use only the change description supplied by the user and relevant context already present in the conversation.
2. Do not inspect `git status`, diffs, branches, remotes, files, or repository history. Do not run tools unless the user explicitly asks for repository-based analysis.
3. Construct one commit subject that follows the style above.
4. Return only the copy-pasteable message in a `text` code block, with no explanation or command wrapper.