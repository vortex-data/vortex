// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Multi-language interface to Vortex arrays, types, files and streams, generated with
//! [Diplomat](https://github.com/rust-diplomat/diplomat).
//!
//! # Why Diplomat?
//!
//! The original `vortex-ffi` crate hand-wrote a C ABI and generated a single `vortex.h` with
//! [cbindgen]. Every type was wrapped by one of the `arc_wrapper!` / `box_wrapper!` macros, every
//! fallible call took an `error_out: *mut *mut vx_error` out-parameter, and each wrapper had to
//! hand-roll its own `_clone` / `_free` functions. Supporting a second or third language (C++,
//! JavaScript, Kotlin, Python, …) meant writing and maintaining that idiomatic layer by hand on
//! top of the raw C ABI.
//!
//! Diplomat replaces all of that. We describe the API *once*, in Rust, inside `#[diplomat::bridge]`
//! modules. The proc-macro lowers each bridge to a stable `extern "C"` surface, and the
//! `diplomat-tool` CLI then emits *idiomatic* bindings for every supported backend:
//!
//! ```text
//! diplomat-tool c      cinclude          # C headers
//! diplomat-tool cpp    bindings/cpp
//! diplomat-tool js     bindings/js       # JavaScript + TypeScript
//! diplomat-tool dart   bindings/dart
//! diplomat-tool kotlin bindings/kotlin
//! diplomat-tool nanobind bindings/python # Python via nanobind
//! ```
//!
//! ## Mapping from the old C ABI
//!
//! | Hand-written `vortex-ffi`                         | Diplomat bridge                                   |
//! |---------------------------------------------------|---------------------------------------------------|
//! | `arc_wrapper!` / `box_wrapper!` opaque pointers   | `#[diplomat::opaque] pub struct ...`              |
//! | manual `vx_*_free` / `vx_*_clone`                 | auto-generated destructor (and `clone` where kept)|
//! | `error_out: *mut *mut vx_error` + `try_or(..)`    | `Result<T, Box<VortexFfiError>>`                  |
//! | `*const c_char` + `CStr` validation               | `&str` / `&DiplomatStr`                           |
//! | returning a borrowed `*const vx_string`           | `out: &mut DiplomatWrite`                         |
//! | `#[repr(C)]` discriminant enums (`vx_ptype`, …)   | plain Diplomat `enum`                             |
//! | `#[unsafe(no_mangle)] pub extern "C-unwind" fn`   | a normal method inside `impl` in the bridge       |
//!
//! Each submodule below contains exactly one `#[diplomat::bridge] mod ffi { .. }`. The bridge
//! modules are the entire public surface; the rest of this crate is glue shared between them.
//!
//! [cbindgen]: https://github.com/mozilla/cbindgen

mod array;
mod array_iterator;
mod binary;
mod data_source;
mod dtype;
mod error;
mod expression;
mod file;
mod log;
mod ptype;
mod scalar;
mod scan;
mod session;
mod sink;
mod string;
mod struct_array;
mod struct_fields;

use std::sync::LazyLock;

use vortex::io::runtime::current::CurrentThreadRuntime;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A shared runtime for all FFI operations.
///
/// Diplomat is a *synchronous*, unidirectional (Rust → foreign) framework, so the async Vortex
/// scan/IO APIs are driven to completion on this runtime behind each bridge method.
pub(crate) static RUNTIME: LazyLock<CurrentThreadRuntime> =
    LazyLock::new(CurrentThreadRuntime::new);
