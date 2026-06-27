// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Guest SDK for writing Vortex WebAssembly decoder kernels, in Rust.
//!
//! An encoding author builds a `cdylib` for `wasm32-unknown-unknown` that depends on this crate,
//! implements [`WasmEncoding`], and calls [`export_wasm_encoding!`]. The resulting `.wasm` is
//! embedded in a Vortex file and run by `vortex-wasm`'s `WasmKernel` at read time.
//!
//! The SDK is **dependency-free** (`core`/`alloc` only) to keep kernels small — in particular it
//! does not use `vortex-error`. Decoded arrays cross the host/guest boundary as the
//! [Arrow C Data Interface](crate::arrow), which is plain byte layout, so no Arrow library (or
//! nanoarrow) is needed. Errors use the minimal [`GuestError`].
//!
//! See `docs/design/wasm-encodings.md`.

pub mod abi;
pub mod arrow;
pub mod bitpack;
mod encoding;
mod error;
pub mod host;

#[doc(hidden)]
pub use encoding::__run_decode;
pub use encoding::WasmEncoding;
pub use error::GuestError;
pub use error::GuestResult;
