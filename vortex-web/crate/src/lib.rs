// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WASM bindings for the Vortex web explorer.
//!
//! Built with `wasm-pack build --target web` and consumed by the vortex-web frontend.

#[cfg(any(target_arch = "wasm32", test))]
mod array_tree_json;

#[cfg(target_arch = "wasm32")]
use std::sync::LazyLock;

#[cfg(target_arch = "wasm32")]
use vortex::VortexSessionDefault;
#[cfg(target_arch = "wasm32")]
use vortex::io::runtime::wasm::WasmRuntime;
#[cfg(target_arch = "wasm32")]
use vortex::io::session::RuntimeSessionExt;
#[cfg(target_arch = "wasm32")]
use vortex::session::VortexSession;

#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(target_arch = "wasm32")]
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default().with_handle(WasmRuntime::handle());
    session.allow_unknown();
    session
});
