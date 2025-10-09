// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::missing_safety_doc)]
#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod dtype;
mod error;
mod file;
mod log;
mod macros;
mod ptype;
mod session;
mod sink;
mod string;
mod struct_fields;

use std::ffi::{
    CStr,
    c_char,
    c_int,
};
use std::sync::Arc;

pub use log::vx_log_level;
use parking_lot::Mutex;
use tokio::runtime;
use tokio::runtime::Runtime;
use vortex::error::VortexExpect;
use vortex::io::runtime::tokio::TokioRuntime;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Shared runtime for all sessions; may be dropped by calling `try_shutdown_runtime`
// if no more sessions are active
static RUNTIME_STATE: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

pub(crate) fn get_runtime() -> Arc<Runtime> {
    let mut state = RUNTIME_STATE.lock();

    if let Some(runtime) = state.as_ref() {
        runtime.clone()
    } else {
        let runtime = Arc::new(
            runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .vortex_expect("Cannot start runtime"),
        );
        *state = Some(runtime.clone());
        runtime
    }
}

pub(crate) fn get_vx_runtime() -> TokioRuntime {
    TokioRuntime::from(get_runtime().handle())
}

/// Attempt to shutdown the runtime by calling `drop` if no other references exist
/// (e.g., no more VortexSessions are active). May block indefinitely if the runtime
/// is still running tasks.
pub fn try_shutdown_runtime() {
    let mut state = RUNTIME_STATE.lock();

    if let Some(runtime) = state.take() {
        match Arc::try_unwrap(runtime) {
            // We have the only reference, safe to shut down
            Ok(runtime) => drop(runtime),
            // There are other live references, so put it back
            Err(runtime) => *state = Some(runtime),
        }
    }
}

pub(crate) unsafe fn to_string(ptr: *const c_char) -> String {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_string_lossy().into_owned()
}

pub(crate) unsafe fn to_string_vec(ptr: *const *const c_char, len: c_int) -> Vec<String> {
    (0..len)
        .map(|i| unsafe { to_string(*ptr.offset(i as isize)) })
        .collect()
}

/// Attempt to shutdown the shared tokio runtime if no sessions are active.
/// May block indefinitely if the runtime is still running tasks.
#[unsafe(no_mangle)]
pub extern "C" fn vx_try_shutdown_runtime() {
    try_shutdown_runtime();
}
