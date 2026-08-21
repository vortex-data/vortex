// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::Handle;

/// Drives a future on the current thread, spinning while no task is runnable.
///
/// A single-threaded WebAssembly thread cannot park, so we spin instead, which is sound because
/// every wake-up must come from this same thread.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    super::spin::block_on(future)
}

/// Browser WebAssembly schedules onto the JavaScript event loop.
#[cfg(all(target_os = "unknown", feature = "wasm-bindgen"))]
pub(crate) fn default_handle() -> Option<Handle> {
    // A `cfg`-conditional import, so it lives in the function rather than at module scope.
    use crate::runtime::wasm::WasmRuntime;

    Some(WasmRuntime::handle())
}

/// Every other WebAssembly target — WASI, or a browser build without `wasm-bindgen` — has no event
/// loop to schedule onto, so callers must drive a
/// [`SingleThreadRuntime`](crate::runtime::single::SingleThreadRuntime) and install its handle
/// themselves.
#[cfg(not(all(target_os = "unknown", feature = "wasm-bindgen")))]
pub(crate) fn default_handle() -> Option<Handle> {
    None
}
