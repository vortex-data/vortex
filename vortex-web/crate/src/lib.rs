// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(target_arch = "wasm32")]

//! WASM bindings for the Vortex web explorer.
//!
//! Built with `wasm-pack build --target web` and consumed by the vortex-web frontend.

use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::io::runtime::wasm::WasmRuntime;
use vortex::io::session::RuntimeSessionBuilderExt;
use vortex::session::VortexSession;

mod wasm;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::default_builder()
        .with_handle(WasmRuntime::handle())
        .allow_unknown()
        .build()
});
