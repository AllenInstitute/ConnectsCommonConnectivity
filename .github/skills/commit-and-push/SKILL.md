---
name: commit-and-push
description: "Commit relevant changes with the user's preferred descriptive message style, then push the current branch. Use when asked to commit and push, write a commit message and push, or run the git commit workflow."
argument-hint: "[optional change summary or message]"
---

# Commit and Push

Commit the changes relevant to the current task and push the current branch.

## Commit Message

Use the [construct-commit-message skill](../construct-commit-message/SKILL.md)
to derive the commit subject from the actual diff and any message or summary
supplied by the user.

## Workflow

1. Inspect `git status`, the staged diff, and the unstaged diff. Identify only the changes belonging to the current task.
2. Check the current branch and its upstream. Never switch branches as part of this workflow.
3. If unrelated changes exist, leave them untouched. Stage only task-relevant paths or hunks. If relevant and unrelated edits cannot be separated safely, ask the user before committing.
4. Follow the linked `construct-commit-message` skill to derive the exact commit subject.
5. Before committing, show the exact subject in a copy-pasteable `text` code block labeled `Commit message`.
6. Treat an explicit invocation of this skill as authorization to commit the identified task changes and push them. Ask for confirmation only when the intended files, message, remote, or branch are ambiguous.
7. Create one commit using the displayed subject. Do not amend an existing commit unless explicitly requested.
8. Push normally to the configured upstream. If no upstream exists, use `git push -u origin <current-branch>` after confirming that `origin` is the intended remote. Never force-push.
9. Verify the push succeeded, then report the commit hash, exact message, branch, remote, and any remaining uncommitted changes.

## Safety

- Never commit credentials, tokens, private keys, environment secrets, or obvious generated artifacts that are not intentionally part of the task.
- Never discard, overwrite, or include unrelated user changes.
- Stop and explain the blocker if commit hooks or the remote reject the operation; do not bypass checks unless explicitly requested.