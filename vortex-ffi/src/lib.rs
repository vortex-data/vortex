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

use std::ffi::{CStr, c_char, c_int};
use std::sync::Arc;

pub use log::vx_log_level;
use parking_lot::Mutex;
use tokio::runtime;
use tokio::runtime::Runtime;
use vortex::error::VortexExpect;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Session-scoped runtime management using Arc reference counting
static RUNTIME_STATE: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

fn get_runtime() -> Arc<Runtime> {
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

/// Get a runtime handle for a new session (increments Arc reference count)
pub(crate) fn get_session_runtime() -> Arc<Runtime> {
    get_runtime()
}

/// Attempt to shutdown the runtime if no more references exist
pub(crate) fn try_shutdown_runtime() {
    let mut state = RUNTIME_STATE.lock();

    if let Some(runtime) = state.take() {
        // Check if we have the only reference (strong_count == 1 means only the one in the Option)
        if Arc::strong_count(&runtime) == 1 {
            // We have the only reference, safe to shut down
            std::thread::spawn(move || {
                if let Ok(runtime) = Arc::try_unwrap(runtime) {
                    runtime.shutdown_background();
                }
            });
        } else {
            // Still have other references, put it back
            *state = Some(runtime);
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
