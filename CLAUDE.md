# Vortex

## Development Guidelines

* project is a monorepo Rust workspace, java bindings in `/java`, python bindings in `/vortex-python`
* run `cargo build -p` to build a specific crate
* use `cargo clippy --all-targets --all-features` to make sure a project is free of lint issues. Please do this every
  time you reach a stopping point or think you've finished work.
* run `cargo +nightly fmt --all` to format Rust source files. Please do this every time you reach a stopping point or
  think you've finished work.
* run `cargo xtask public-api` to re-generate the public API lock files. Please do this every time you reach a stopping
  point or think you've finished work.
* you can try running
  `cargo fix --lib --allow-dirty --allow-staged && cargo clippy --fix --lib --allow-dirty --allow-staged` to
  automatically many fix minor errors.

## Architecture

* `vortex-buffer` defines zero-copy aligned `Buffer<T>` and `BufferMut<T>` that are guaranteed
  to be aligned to `T` (or whatever requested runtime alignment).
* `vortex-array/src/dtype` contains the basic `DType` logical type enum that is the basis of the Vortex
  type system
* `vortex-array` contains the basic `Array` trait, as well as several encodings which impl
  that trait for each encoding. It includes all of most of the Apache Arrow encodings.
* More exotic compressed encodings live in the crates inside of `/encodings/*`
* File IO is defined in `vortex-file`. It uses the concept of a `LayoutReader` defined
  in `vortex-layout` crate.
* `/vortex-python` contains the python bindings. rst flavored docs for the project are in `/docs`

## Code Style

* Prefer `impl AsRef<T>` to `&T` for public interfaces where possible, e.g. `impl AsRef<Path>`
* Avoid usage of unsafe where not necessary, use zero-cost safe abstractions wherever possible,
  or cheap non-zero-cost abstractions.
* Every new public API definition must have a doc comment. Examples are nice to have but not
  strictly required.
* Use `vortex_err!` to create a `VortexError` with a format string and `vortex_bail!` to do the same but immediately
  return it as a `VortexResult<T>` to the surrounding context.
* When writing tests, strongly consider using `rstest` cases to parameterize repetitive test logic.
* If you want to create a large number of tests to an existing file module called `foo.rs`, and if you think doing so
  would
  be too many to inline in a `tests` submodule within `foo.rs`, then first promote `foo` to a directory module. You can
  do
  this by running `mkdir foo && mv foo.rs foo/mod.rs`. Then, you can create a test file `foo/tests.rs` that you include
  in `foo/mod.rs` with the appropriate test config flag.
* If you encounter clippy errors in tests that should only pertain to production code (e.g., prohibiting panic/unwrap,
  possible numerical truncation, etc.), then consider allowing those lints at the test module level.
* Prefer naming test modules `tests`, not `test`.
* Prefer having test return VortexResult<()> and use ? over unwrap.
* All imports must be at the top of the module, never inside functions. The only exception is `#[cfg(test)]` blocks,
  where imports should be at the top of the test module. Function-scoped imports are only acceptable when (a) required,
  or (b) it would be exceptionally verbose otherwise, such as a match statement where left and right sides have similar
  names.
* Imports should be preferred over qualified identifiers.
* Only write comments that explain non-obvious logic or important context. Avoid commenting simple or self-explanatory
  code.
* Use `assert_arrays_eq!` macro for comparing arrays in tests instead of element-by-element comparison.
* Keep tests concise and to the point - avoid unnecessary setup or verbose assertions.
* Run tests for a specific crate with `cargo test -p <crate-name>` (e.g., `cargo test -p vortex-array`).

## Other

* When summarizing your work, please produce summaries in valid Markdown that can be easily copied/pasted to Github.

## Commits

* All commits must be signed of by the committers in the form `Signed-off-by: "COMMITTER" <COMMITTER_EMAIL>`.