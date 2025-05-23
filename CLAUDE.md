# Development Guidelines

* project is a monorepo Rust workspace, java bindings in `/java`, python bindings in `/pyvortex`
* run `cargo build -p` to build a specific crate
* use `cargo clippy --all-targets --all-features` to make sure a project is free of lint issues

# Architecture

* `vortex-buffer` defines zero-copy aligned `Buffer<T>` and `BufferMut<T>` that are guaranteed
  to be aligned to `T` (or whatever requested runtime alignment).
* `vortex-dtype` contains the basic `DType` logical type enum that is the basis of the Vortex
  type system
* `vortex-array` contains the basic `Array` trait, as well as several encodings which impl
  that trait for each encoding. It includes all of most of the Apache Arrow encodings.
* More exotic compressed encodings live in the crates inside of `/encodings/*`
* File IO is defined in `vortex-file`. It uses the concept of a `LayoutReader` defined
  in `vortex-layout` crate.
* `/pyvortex` contains the python bindings. rst flavored docs for the project are in `/docs`

# Code Style

* Prefer `impl AsRef<T>` to `&T` for public interfaces where possible, e.g. `impl AsRef<Path>`
* avoid usage of unsafe where not necessary, use zero-cost safe abstractions wherever possible,
  or cheap non-zero-cost abstractions.
* Every new public API definition must have a doc comment. Examples are nice to have but not
  strictly required.

