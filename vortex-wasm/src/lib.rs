// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Embedded WebAssembly decoder kernels for the Vortex file format.
//!
//! This crate lets a Vortex file carry the *decoder* for an encoding inside the file as a
//! sandboxed WebAssembly module. A reader that understands the [`WasmLayout`] can decode arrays
//! written with an encoding it was never compiled against by running the embedded kernel against
//! the serialized array and the host's existing decode machinery.
//!
//! - [`abi`] defines the host/guest ABI constants.
//! - [`arrow_ffi`] imports/exports canonical arrays across the boundary as Arrow C Data Interface
//!   structs.
//! - [`WasmKernel`] is the `wasmtime`-backed runtime that drives the ABI.
//! - [`WasmLayout`] / [`WasmReader`] / [`WasmLayoutStrategy`] integrate kernels into the layout
//!   tree, so wasm-decoded arrays read and write like any other layout.
//!
//! Call [`register_wasm_layout`] on a session before reading files that contain a [`WasmLayout`].
//!
//! See `docs/design/wasm-encodings.md` for the full design.

pub mod abi;
pub mod arrow_ffi;
mod kernel;
mod layout;
mod reader;
mod writer;

pub use kernel::HostDecoder;
pub use kernel::WasmKernel;
pub use layout::WASM_LAYOUT_ID;
pub use layout::Wasm;
pub use layout::WasmLayout;
pub use layout::WasmLayoutEncoding;
pub use layout::WasmLayoutMetadata;
pub use reader::WasmReader;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;
pub use writer::IdentityEncoder;
pub use writer::WasmEncoded;
pub use writer::WasmEncoder;
pub use writer::WasmLayoutStrategy;

/// Register the [`WasmLayout`] encoding on a session so files containing it can be read.
pub fn register_wasm_layout(session: &VortexSession) {
    session
        .layouts()
        .registry()
        .register(WasmLayoutEncoding.id(), WasmLayoutEncoding.as_ref());
}
