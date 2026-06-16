# Vortex multi-language interface (Diplomat)

This crate is an **illustrative** rewrite of the hand-written [`vortex-ffi`](../vortex-ffi) crate
on top of [Diplomat](https://github.com/rust-diplomat/diplomat), following the approach described
in [*Diplomat: Multi-Language FFI for Rust Libraries*][blog].

Where `vortex-ffi` hand-rolls a C ABI (opaque-pointer wrapper macros, `error_out` out-parameters,
manual `_clone`/`_free`, a single cbindgen-generated `vortex.h`), this crate describes the API
**once** in Rust inside `#[diplomat::bridge]` modules and lets `diplomat-tool` generate *idiomatic*
bindings for many languages.

[blog]: https://manishearth.github.io/blog/2026/06/14/diplomat-multi-language-ffi-for-rust-libraries/

## What changed vs. `vortex-ffi`

| Hand-written `vortex-ffi`                        | Diplomat bridge                                       |
|--------------------------------------------------|-------------------------------------------------------|
| `arc_wrapper!` / `box_wrapper!` opaque pointers  | `#[diplomat::opaque] pub struct ...`                  |
| manual `vx_*_free` / `vx_*_clone`                | destructor auto-generated; `clone` only where useful  |
| `error_out: *mut *mut vx_error` + `try_or(..)`   | `Result<T, Box<VortexFfiError>>` (→ exceptions/Result) |
| `*const c_char` + `CStr` UTF-8 validation        | `&str` / `&DiplomatStr`                               |
| returning a borrowed `*const vx_string`          | `out: &mut DiplomatWrite`                             |
| `#[repr(C)]` discriminant enums                  | plain Diplomat `enum`                                 |
| `#[unsafe(no_mangle)] extern "C-unwind" fn`      | a normal method on the opaque inside the bridge       |
| constructors as free `vx_*_new` functions        | `#[diplomat::attr(auto, constructor)]`                |
| getters as free `vx_*_get_*` functions           | `#[diplomat::attr(auto, getter)]`                     |
| iterator via `vx_array_iterator_next`            | `#[diplomat::attr(auto, iterator)]`                   |

## Module layout

Each module holds exactly one `#[diplomat::bridge] mod ffi { .. }`:

- `error`        — `VortexFfiError` opaque; the `Result` error type shared by every bridge.
- `session`      — `VxSession`.
- `log`          — `VxLogLevel` enum + log configuration.
- `file`         — `VxFile` read/write helpers.
- `ptype`        — `VxPType` primitive-type enum.
- `dtype`        — `VxDType` logical type.
- `scalar`       — `VxScalar` typed constructors/accessors.
- `struct_fields`— field-name helpers.
- `array`        — `VxArray` core array + per-primitive accessors + Arrow C Data Interface.
- `array_iterator` — `VxArrayIterator` (Diplomat iterator).
- `struct_array` — struct array helpers.
- `binary` / `string` — byte/UTF-8 element accessors.
- `expression`   — `VxExpr` expression DSL (columns, literals, comparison/logical ops).
- `scan`         — `VxScan` scan builder + execution.
- `data_source`  — `VxDataSource` (see the unidirectional-FFI caveat below).
- `sink`         — `VxArraySink` for writing arrays out.

## Generating bindings

```sh
# Build the bridge crate (produces the static/shared library).
cargo build --release -p vortex-ffi-diplomat

# Install the codegen CLI once.
cargo install diplomat-tool

# Generate idiomatic bindings per language from the bridge crate's src/lib.rs entrypoint.
diplomat-tool c        cinclude            -e src/lib.rs
diplomat-tool cpp      bindings/cpp        -e src/lib.rs
diplomat-tool js       bindings/js         -e src/lib.rs   # JS + TypeScript
diplomat-tool dart     bindings/dart       -e src/lib.rs
diplomat-tool kotlin   bindings/kotlin     -e src/lib.rs
diplomat-tool nanobind bindings/python     -e src/lib.rs   # Python via nanobind
```

The C and C++ output is a drop-in replacement for today's `cinclude/vortex.h`; the other backends
are new capabilities that the hand-written ABI did not provide without bespoke per-language glue.

## Caveats (this is illustrative)

- **Diplomat is unidirectional (Rust → foreign).** The original `data_source` module accepts a C
  *callback* so a foreign caller can feed bytes back into Rust. Diplomat does not generate
  foreign→Rust callbacks, so `VxDataSource` here exposes opaque constructors over concrete sources
  (file / URL / in-memory buffer) instead. A custom-callback source would still need a small
  hand-written shim. See `data_source.rs`.
- **Async.** Vortex scan/IO is async; Diplomat methods are synchronous, so each bridge method drives
  the future to completion on the shared `RUNTIME` (see `lib.rs`).
- The bridges prioritise showing the *shape* of the Diplomat API. They are not guaranteed to
  compile as-is against the current `vortex` crate.
