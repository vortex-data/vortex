---
name: ci-failure-analysis
description: Analyze Vortex GitHub Actions CI failures. Use when asked to investigate failed CI runs, failed jobs, or when the user mentions "/ci-failure-analysis".
---

# Vortex CI Failure Analysis Skill

Analyze failed GitHub Actions runs for the Vortex repository and identify whether the failure is
caused by the PR, pre-existing flakiness, infrastructure, or an unrelated main-branch issue.

## Inputs

Use any PR number, repository, run ID, failed job metadata, or log snippets supplied by the user or
automation prompt. If a needed value is missing, discover it with the narrowest `gh` command that
can answer the question.

## Workflow

1. List failed jobs for the workflow run:

   ```bash
   gh run view <run-id> --repo <owner/repo> --json jobs
   ```

2. Fetch only failed job logs first:

   ```bash
   gh run view <run-id> --repo <owner/repo> --job <job-id> --log-failed
   ```

   If that fails, use the Actions API:

   ```bash
   gh api repos/<owner/repo>/actions/jobs/<job-id>/logs
   ```

3. If any `gh` command fails with `error connecting to api.github.com` in a sandbox, rerun it with
   escalated network permissions immediately.

4. Classify each failure:

   - Rust build errors: compiler diagnostics, spans, trait bound failures, feature-gate issues.
   - Rust test failures: failing test name, panic/assertion output, expected vs actual values,
     source path and line.
   - Clippy failures: lint name, file path, line, and suggested fix if shown.
   - Formatting or public API failures: changed files and commands needed to regenerate output.
   - Python/docs failures: pytest, maturin, Sphinx, doctest, or packaging output.
   - Infrastructure failures: toolchain download, cache, runner, network, disk, timeout, or
     service issues.

5. Fetch the PR diff and metadata only after the failing log section is understood:

   ```bash
   gh pr view <pr-number> --repo <owner/repo> --json title,body,baseRefName,headRefName,files,commits
   gh pr diff <pr-number> --repo <owner/repo>
   ```

6. Reproduce narrowly when practical:

   ```bash
   cargo test -p <crate-name> <test-name>
   cargo clippy -p <crate-name> --all-targets --all-features
   make -C docs doctest
   uv run --all-packages pytest <path>
   ```

7. Check whether the same failure appears on recent main-branch runs or open issues before calling
   it PR-caused.

## Report Format

Post or return one concise Markdown report:

````markdown
## CI Failure Analysis

### Status
<PR-caused | likely pre-existing | infrastructure | inconclusive>

### Failed Jobs
- `<job name>`: <build | test | clippy | fmt | docs | infra>

### Relevant Log Output
```text
<only the failing lines needed to understand the issue>
```

### Correlation With PR Changes
<Explain whether the diff touches the failing area and cite files/functions.>

### Recommended Next Step
<One or two concrete commands or code fixes.>
````

## Rules

- Show relevant failure excerpts, not full logs.
- If many tests fail, detail the first few distinct failures and summarize the rest.
- Do not guess. If causation is unclear, say what was checked and what would resolve it.
- Prefer one PR comment or one final report over multiple fragmented updates.
