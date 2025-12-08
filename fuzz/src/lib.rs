// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::use_debug)]

mod array;
pub mod error;

// File module only available for native builds (requires vortex-file which uses tokio)
#[cfg(not(target_arch = "wasm32"))]
pub mod file;
pub use array::Action;
pub use array::CompressorStrategy;
pub use array::ExpectedValue;
pub use array::FuzzArrayAction;
pub use array::run_fuzz_action;
pub use array::sort_canonical_array;
#[cfg(not(target_arch = "wasm32"))]
pub use file::FuzzFileAction;

// Runtime initialization - platform-specific
#[cfg(not(target_arch = "wasm32"))]
mod native_runtime {
    use std::sync::LazyLock;

    use vortex::VortexSessionDefault;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::current::CurrentThreadRuntime;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    pub static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
    pub static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::default().with_handle(RUNTIME.handle()));
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_runtime::RUNTIME;
#[cfg(not(target_arch = "wasm32"))]
pub use native_runtime::SESSION;

#[cfg(target_arch = "wasm32")]
mod wasm_runtime {
    use std::sync::LazyLock;

    use vortex::VortexSessionDefault;
    use vortex_io::runtime::wasm::WasmRuntime;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    pub static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::default().with_handle(WasmRuntime::handle()));
}

#[cfg(target_arch = "wasm32")]
pub use wasm_runtime::SESSION;
