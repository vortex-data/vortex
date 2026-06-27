// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Guest SDK for writing Vortex WebAssembly decoder kernels.
//!
//! An encoding author builds a `cdylib` for `wasm32-unknown-unknown` that depends on this crate,
//! implements [`WasmEncoding`], and calls [`export_wasm_encoding!`]. The resulting `.wasm` is
//! embedded in a Vortex file by the `vortex-wasm` writer and run by its [`WasmKernel`] at read
//! time.
//!
//! This crate depends only on `vortex-flatbuffers`, `vortex-buffer`, and `vortex-error` — never on
//! the rest of Vortex — so a kernel can parse the array flatbuffer header and produce canonical
//! output without pulling in the full decode stack.
//!
//! See `docs/design/wasm-encodings.md`.

pub mod abi;
mod encoding;
pub mod header;
pub mod host;
pub mod message;

#[doc(hidden)]
pub use encoding::__run_decode;
pub use encoding::WasmEncoding;
