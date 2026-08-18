---
name: docstring-writing
description: "Write, improve, audit, or review Python docstrings and API documentation. Use for docstring-writing requests, docstring review comments, consistency passes, and documentation-only contract corrections."
argument-hint: "[files, symbols, or review comments]"
---

# Docstring Writing

Write accurate Python docstrings from observable behavior. A caller should not
need to reconstruct the implementation to understand the API contract.

## Principles

- Base behavioral claims on the implementation. Use tests, annotations, and
  existing documentation to distinguish the supported API contract from
  incidental permissiveness or apparent bugs. When sources conflict, document
  only claims supported by the implementation and report the mismatch.
- Describe purpose, semantic roles, interactions, and guarantees rather than
  narrating implementation or repeating annotations.
- Follow nearby docstring style and project terminology. When no convention is
  clear, use NumPy-style sections.
- Keep detail proportional to ambiguity, risk, and API visibility.
- Never claim validation, safety, atomicity, or idempotency that the code does
  not guarantee.

## Procedure

### 1. Establish Scope

Identify the requested files and symbols. For a consistency pass, enumerate the
explicitly requested scope before editing. Keep documentation-only work free of
runtime changes unless the user expands the task.

### 2. Read the Controlling Behavior

Start with the implementation that directly computes, validates, mutates, or
performs IO. If the visible API only forwards or registers behavior, follow it
to that implementation. Consult callers and tests only when the contract remains
ambiguous.

Determine the relevant parts of the contract:

| Category | Contract question |
|---|---|
| Purpose | What responsibility and boundary does the API own? |
| Inputs | What is accepted, required, normalized, or rejected? |
| Parameters | What does each input mean, and how do inputs interact? |
| Results | What is returned or yielded, including identity and sentinel values? |
| Side effects | What is mutated, persisted, cached, logged, or transmitted? |
| Errors | Which stable failures can callers reasonably handle? |
| Guarantees | What ordering, isolation, retry, or concurrency behavior exists? |

### 3. Write the Contract

- Start with the API's purpose and responsibility.
- Define non-trivial parameters by semantic role. State relationships,
  precedence, mutual exclusions, and consumed configuration fields.
- For structured data, state required shapes, columns, keys, ordering, or
  readiness assumptions that the implementation enforces.
- For paths, distinguish roots, directories, files, URIs, table locations,
  temporary locations, and final destinations.
- Explain the meaning of results, tuple members, structured fields, optional
  values, and whether objects are originals, copies, views, or cached instances.
- Use `Yields` for generators.
- Document explicit, stable, caller-actionable exceptions by condition. Omit
  incidental dependency failures unless they are part of the API contract.
- Use `Notes` only for relevant side effects, invariants, caching, ordering,
  dry-run behavior, concurrency limits, or important non-guarantees.

### 4. Check Accuracy and Consistency

Challenge absolute terms such as `always`, `safe`, `atomic`, `idempotent`,
`validates`, `unique`, `preserves`, and `thread-safe` against the implementation.
Annotations and intended shapes are not runtime validation.

Within a requested consistency scope, equivalent APIs should cover equivalent
contract categories and use consistent terminology. They do not need identical
section counts or length. Leave concise, adequate docstrings concise.

## Special Cases

Apply only the relevant guidance:

- **Validation helpers:** accepted types or subclasses, normalization, identity,
  controlling policy, and whether validation performs IO or mutation.
- **IO helpers:** input readiness, full destination semantics, write mode,
  replacement scope, return counts, and transaction or concurrency limits.
- **Configuration resolvers:** precedence, mutual exclusions, explicit versus
  discovered values, and every member of structured results.
- **Collections and tables:** required columns or keys, empty inputs, duplicates,
  grouping, ordering, and row-order preservation.
- **Cached factories:** cache identity and bounds, object reuse, and mutation.
- **Placeholders:** why the API is unavailable, whether arguments are ignored,
  and the exception that is always raised.

## Verification

For documentation-only changes:

1. Run configured syntax and lint checks for touched modules; do not treat lint
   success as proof that behavioral claims are accurate.
2. Run focused tests when available and proportionate to the change.
3. Inspect the diff for unrelated churn and confirm signatures and executable
   statements did not change. Use an existing AST-comparison tool if provided;
   do not invent one solely for this task.

Report unavailable checks and use the nearest practical validation.