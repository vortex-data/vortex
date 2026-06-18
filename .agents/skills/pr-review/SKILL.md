---
name: pr-review
description: Review Vortex pull requests for correctness, Rust soundness, performance, API compatibility, and test coverage. Use when reviewing PRs, reviewing code changes, or when the user mentions "/pr-review".
---

# Vortex PR Review Skill

Review Vortex changes for issues that CI may miss: semantic correctness, Rust soundness,
zero-copy and alignment invariants, performance, API compatibility, and missing regression
coverage.

## Usage Modes

### GitHub Actions Mode

When invoked by a PR automation, the prompt may already include PR metadata, issue body,
comments, and changed files. In this mode, use git commands for the diff and history:

```bash
git diff origin/<base-branch>...HEAD
git diff --stat origin/<base-branch>...HEAD
git log origin/<base-branch>..HEAD --oneline
```

If the base ref is missing, fetch only that ref:

```bash
git fetch origin <base-branch> --depth=1
```

Do not refetch data already provided in the prompt. Use the supplied comments and metadata as
review context.

### Local CLI Mode

When the user provides a PR number or URL, use `gh` to fetch the PR metadata, comments, and diff:

```bash
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits
gh pr view <PR_NUMBER> --json comments,reviews
gh pr diff <PR_NUMBER>
```

If `gh` cannot connect to `api.github.com` because of sandbox networking, rerun with escalated
network permissions.

## Review Workflow

1. Read `AGENTS.md` and any nested `AGENTS.md` for project conventions.
2. Identify the change intent from the PR title, body, commits, and tests.
3. Group changed files by area: arrays, encodings, buffers, file/layout, integrations, bindings,
   docs, or CI.
4. Trace changed behavior through callers, trait implementations, dtype/nullability handling,
   validity masks, and tests.
5. Focus findings on actionable defects. Avoid commenting on formatting or issues already covered
   by clippy, rustfmt, or generated API checks.
6. Scope verification to the change. Rust/API changes need Rust checks; docs-only, agent-only,
   symlink-only, and metadata-only changes should use targeted validation such as Markdown review,
   `ls`, `find`, `git status`, or relevant config linters.

## Review Areas

| Area               | Focus                                                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness        | Length and dtype invariants, nullability, validity masks, offset math, canonicalization, boundary conditions, empty arrays, scalar vs array behavior |
| Rust soundness     | `unsafe` blocks, aliasing, lifetimes, alignment, FFI boundaries, panic safety, ownership of buffers and arrays                                       |
| Compression and IO | Encoding metadata, statistics, layout evolution, file compatibility, scan projection/filter behavior, async IO edge cases                            |
| Performance        | Unnecessary copies, lost zero-copy behavior, avoidable allocations, poor cache locality, quadratic loops, excessive dynamic dispatch in hot paths    |
| Error handling     | Correct `vortex_err!` and `vortex_bail!` usage, useful messages, no accidental panics on user data                                                   |
| API compatibility  | Public API docs, feature flags, crate boundaries, Python/Java binding impacts                                                                        |
| Tests              | Regression coverage, edge cases, parameterized cases with `rstest`, use of `assert_arrays_eq!`, docs doctests when docs change                       |
| Verification scope | Avoid requesting or running expensive workspace checks when the PR only changes docs, agent files, symlinks, or metadata                             |

## Output

Lead with findings, ordered by severity. For each finding include:

- File and line reference.
- Why the issue is a real bug or material risk.
- A concrete fix or verification path when possible.

Use inline review comments when the environment supports them and a precise changed line is the
best place for the feedback. Keep broad design feedback in the summary.

If no issues are found, say so explicitly and mention any residual risk or tests not run.

## Principles

- Review the code that changed, but inspect enough surrounding code to validate invariants.
- Do not infer causation from commit messages alone. Verify with code, tests, or logs.
- Do not ask for broad rewrites when a narrow fix would address the risk.
- Do not downgrade a correctness or soundness issue to a nit because it is inconvenient.
- Be specific and proportionate.
