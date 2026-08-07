# AGENTS.md

Guidance for Claude, Codex and other coding agents working in the Vortex repository.

## Task Routing

- When asked a question about the PR or codebase, especially via `/query`, use the
  `.agents/skills/query` skill.
- When asked to investigate a CI failure, especially via `/ci-failure-analysis`, use the
  `.agents/skills/ci-failure-analysis` skill.

## Overview

Vortex is a Rust monorepo for columnar array processing, compression encodings, and file IO.
The workspace also contains Java bindings in `java/`, Python bindings in `vortex-python/`,
documentation in `docs/`, and benchmark tooling in `vortex-bench/` and `benchmarks/`.

## Repository Layout

- `vortex-buffer` defines zero-copy aligned `Buffer<T>` and `BufferMut<T>`, guaranteed to
  be aligned to `T` or to a requested runtime alignment.
- `vortex-array/src/dtype` contains the `DType` logical type system used throughout Vortex.
- `vortex-array` contains the core `Array` trait and the base encodings, including most
  Apache Arrow-style encodings.
- `encodings/*` contains more specialized compressed encodings.
- `vortex-file` implements file IO. It uses `LayoutReader` from `vortex-layout`.
- `vortex-io` holds the core async and blocking IO traits, plus the generic `object_store`
  adapters that implement them.
- `vortex-cloud` holds the cloud object store integration: the URL-to-`ObjectStore` registry
  and the OpenDAL-backed services (`cos://`, `oss://`). Every binding resolves URLs through it.
- `vortex-scan`, `vortex-session`, `vortex-datafusion`, and `vortex-duckdb` contain scan
  and execution integrations.
- `vortex-python` contains Python bindings. RST-flavored project docs live in `docs/`.

## Scoped Guidance

Before changing files in a subtree, read the closest nested `AGENTS.md`. In particular:

- `.github/AGENTS.md` covers workflows and other GitHub configuration.
- `docs/AGENTS.md` covers Sphinx documentation.
- `vortex-python/AGENTS.md` covers Python and PyO3 binding work.

## Build

Prefer narrow crate builds while iterating:

```bash
cargo build -p <crate-name>
```

Use workspace-wide builds only when the change spans crate boundaries or before handing off a
broad refactor:

```bash
cargo build --workspace
```

## Testing

Run tests for the crate or binding you touched before broader checks:

```bash
cargo nextest run -p <crate-name>
```

If cargo-nextest is not available, you can install it with:

```bash
cargo install --locked cargo-nextest
```

For Rust doc comments or crate documentation, run doctests for the affected crate:

```bash
cargo test --doc -p <crate-name>
```

## Linting, Formatting, and Generated Files

Run verification that matches the files changed. Do not run expensive Rust checks for changes that
only touch Markdown, agent configuration, comments outside Rust code, symlinks, or other metadata
with no Rust/API behavior impact. For docs/config-only changes, validate formatting by inspection
or with a targeted doc/config command, and verify symlink or path changes with `ls`, `find`, and
`git status`.

For Rust code, public API, feature flag, or generated-file changes, run these before stopping:

```bash
cargo +nightly fmt --all
cargo clippy --all-targets --all-features
```

Notes:

- For `.github/` changes, follow `.github/AGENTS.md` and run
  `yamllint --strict -c .yamllint.yaml` on changed workflow files.
- If cargo fails with exactly `sccache: error: Operation not permitted`, rerun that command
  with `RUSTC_WRAPPER=` so rustc runs directly. Only do this for that exact error.

## CI Investigation

- When iterating on CI failures, fetch only failed job logs first:
  `gh run view <run-id> --job <job-id> --log-failed`.
- Run narrow local repro commands for the affected crate, test, docs target, or binding before
  running workspace-wide checks.
- If a `gh` command fails with `error connecting to api.github.com` in the sandbox, immediately
  rerun it with escalated network permissions instead of retrying in the sandbox.
- Verify causation from logs, diffs, and local repros before attributing a failure to a PR.

## Rust Code Style

- Follow `STYLE.md` for Rust formatting, documentation, API, error-handling, import, and safety
  conventions.
- Only write comments that explain non-obvious logic or important context. Do not comment
  self-explanatory code.
- Keep public APIs small and consistent with neighboring crates.

## Performance

Avoid hidden-cost per-element accessors in hot loops, follow the performance guidance in
`STYLE.md`, and benchmark changes to hot paths.

## Tests

- Strongly consider `rstest` cases when parameterizing repetitive test logic.
- Prefer test functions that return `VortexResult<()>` and use `?` instead of `unwrap`.
- Prefer test module names `tests`, not `test`.
- Use `assert_arrays_eq!` for array comparisons instead of element-by-element assertions.
- Keep tests concise and focused on behavior, edge cases, and regressions.
- If a bug fix is requested, add or identify a failing test first when practical. A test that
  passes before and after the fix does not prove the fix.
- If clippy lints in tests prohibit patterns that are acceptable only in test code, consider
  allowing the lint at the test module level.
- If an existing `foo.rs` module needs many tests, promote it to a directory module:
  `foo/mod.rs` plus `foo/tests.rs`, included from `foo/mod.rs` behind the appropriate test
  configuration.

## Common Mistakes

Check new and modified lines against this list before finishing:

- Running broad CI-style commands before trying a narrow local repro.
- Using `unwrap`, `expect`, or panic-oriented assertions in tests where `VortexResult<()>` and
  `?` would be clearer.
- Comparing arrays element by element instead of using `assert_arrays_eq!`.
- Adding imports inside functions when module-level imports would work.
- Introducing `unsafe` without proving that safe Rust cannot express the same operation.
- Updating expected test output to match buggy behavior without independently verifying the
  intended semantics.
- Silently reducing the scope of an approved plan when implementation is harder than expected.
- Calling a hidden-cost per-element accessor (`Validity::is_valid`, `scalar_at`, `BitBuffer::
  value` accumulation) inside a hot loop instead of materializing once.

## Summaries

When summarizing work, write valid Markdown that can be copied into GitHub. Include the checks
you ran and call out any checks you could not run.

## Commits

All commits must be signed off by the committers in this form:

```text
Signed-off-by: "COMMITTER" <COMMITTER_EMAIL>
```
