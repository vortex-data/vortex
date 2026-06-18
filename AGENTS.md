# AGENTS.md

Guidance for Claude, Codex and other coding agents working in the Vortex repository.

## Task Routing

- When asked to review a PR, especially via `/pr-review`, use the
  `.agents/skills/pr-review` skill.
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
- `vortex-scan`, `vortex-session`, `vortex-datafusion`, and `vortex-duckdb` contain scan
  and execution integrations.
- `vortex-python` contains Python bindings. RST-flavored project docs live in `docs/`.

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

if cargo-nextest is not available you can install it with
```bash
cargo install --locked cargo-nextest
```

Examples:

```bash
cargo nextest run -p vortex-array
make -C docs doctest
uv run --all-packages pytest vortex-python/test
cargo test --doc
```

Run docs doctests from the docs directory with `make -C docs doctest` so the correct Sphinx
Makefile target is used.

If you touch documentation run doc tests via `cargo test --doc`.

For Python binding changes under `vortex-python/`, run the narrow Python checks that match the
files touched before broader test suites. Useful checks include:

```bash
python -m py_compile <changed-python-files>
uv run --all-packages --reinstall-package vortex-data pytest <changed-python-tests>
```

If Python docstrings, `docs/api/python/`, or Sphinx configuration change, also run the docs checks
from a clean Sphinx environment:

```bash
uv run --all-packages make -C docs clean html
uv run --all-packages make -C docs clean doctest
```

## Linting, Formatting, and Generated Files

Run verification that matches the files changed. Do not run expensive Rust checks for changes that
only touch Markdown, agent configuration, comments outside Rust code, symlinks, or other metadata
with no Rust/API behavior impact. For docs/config-only changes, validate formatting by inspection
or with a targeted doc/config command, and verify symlink or path changes with `ls`, `find`, and
`git status`.

For Python binding changes under `vortex-python/`, run the relevant Python lint and type checks:

```bash
uv run basedpyright vortex-python
uv run ruff check <changed-python-files>
```

If PyO3 Rust files in `vortex-python/src/` change, include `cargo +nightly fmt --check -p
vortex-python`. Always finish Python binding work with `git diff --check`.

For Rust code, public API, feature flag, or generated-file changes, run these before stopping:

```bash
cargo +nightly fmt --all
cargo clippy --all-targets --all-features
```

Notes:

- For `.github/` changes, follow `.github/AGENTS.md` and run
  `yamllint --strict -c .yamllint.yaml` on changed workflow files.
- You can try
  `cargo fix --lib --allow-dirty --allow-staged && cargo clippy --fix --lib --allow-dirty --allow-staged`
  to fix minor Rust diagnostics automatically when working on Rust code.
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

- Prefer `impl AsRef<T>` to `&T` for public interfaces where practical, for example
  `impl AsRef<Path>`.
- Avoid `unsafe` unless it is necessary. Prefer zero-cost safe abstractions, or cheap
  non-zero-cost safe abstractions, over hand-written unsafe code.
- Every new public API definition must have a doc comment. Examples are useful but not required.
- Use `vortex_err!` to create a `VortexError` with a format string.
- Use `vortex_bail!` to create and immediately return a `VortexError` as a `VortexResult<T>`.
- Keep imports at the top of the module. The only exception is a `#[cfg(test)]` test module,
  where imports should be at the top of that module.
- Prefer imports over qualified identifiers when the name is used repeatedly.
- Avoid function-scoped imports unless required or unless fully qualifying both sides would be
  exceptionally verbose.
- Only write comments that explain non-obvious logic or important context. Do not comment
  self-explanatory code.
- Keep public APIs small and consistent with neighboring crates.

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

## Summaries

When summarizing work, write valid Markdown that can be copied into GitHub. Include the checks
you ran and call out any checks you could not run.

## Commits

All commits must be signed off by the committers in this form:

```text
Signed-off-by: "COMMITTER" <COMMITTER_EMAIL>
```
