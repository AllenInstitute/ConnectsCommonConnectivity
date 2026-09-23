---
name: commit-and-push
description: "Commit relevant changes with the user's preferred descriptive message style, then push the current branch. Use when asked to commit and push, write a commit message and push, or run the git commit workflow."
argument-hint: "[optional change summary or message]"
---

# Commit and Push

Commit the changes relevant to the current task and push the current branch.

## Commit Message Style

- Use a concise, lowercase, past-tense description.
- Do not use Conventional Commit prefixes such as `feat:`, `fix:`, or `docs:`.
- State what changed first. When useful, add a colon followed by the user-visible behavior, outcome, or reason.
- Prefer concrete domain terms over generic phrases such as "updated files" or "made changes."
- Keep the subject self-contained and omit a trailing period.

Preferred shape:

```text
<what changed>: <user-visible behavior, outcome, or reason>
```

Example:

```text
added to changelog user-visible behavior change: rerunning with fewer taxonomy rows no longer deletes omitted rows
```

## Workflow

1. Inspect `git status`, the staged diff, and the unstaged diff. Identify only the changes belonging to the current task.
2. Check the current branch and its upstream. Never switch branches as part of this workflow.
3. If unrelated changes exist, leave them untouched. Stage only task-relevant paths or hunks. If relevant and unrelated edits cannot be separated safely, ask the user before committing.
4. Derive the commit message from the actual diff and any message or summary supplied by the user. Follow the style above and correct obvious spelling errors without changing the intended meaning.
5. Treat an explicit invocation of this skill as authorization to commit the identified task changes and push them. Ask for confirmation only when the intended files, message, remote, or branch are ambiguous.
6. Create one commit. Do not amend an existing commit unless explicitly requested.
7. Push normally to the configured upstream. If no upstream exists, use `git push -u origin <current-branch>` after confirming that `origin` is the intended remote. Never force-push.
8. Verify the push succeeded, then report the commit hash, exact message, branch, remote, and any remaining uncommitted changes.

## Safety

- Never commit credentials, tokens, private keys, environment secrets, or obvious generated artifacts that are not intentionally part of the task.
- Never discard, overwrite, or include unrelated user changes.
- Stop and explain the blocker if commit hooks or the remote reject the operation; do not bypass checks unless explicitly requested.