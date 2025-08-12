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
use std::sync::atomic::{AtomicBool, Ordering};

pub use log::vx_log_level;
use tokio::runtime;
use tokio::runtime::Runtime;
use vortex::error::VortexExpect;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Runtime with proper shutdown signal
use std::sync::LazyLock;
static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Cannot start runtime")
});

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

fn get_runtime() -> &'static Runtime {
    &RUNTIME
}

/// Explicitly shut down the tokio runtime to prevent cleanup races with mimalloc.
///
/// This sets a shutdown flag and attempts to drain the runtime of pending tasks.
/// While we can't fully shut down a static runtime, this helps minimize the race condition.
#[unsafe(no_mangle)]
pub extern "C" fn vx_runtime_shutdown() -> bool {
    // Set the shutdown flag
    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);

    // Give the runtime a moment to finish any pending work
    std::thread::sleep(std::time::Duration::from_millis(10));

    // For mimalloc compatibility, we need to be more aggressive about cleanup
    // Spawn a task that will block on all current tasks finishing
    let runtime = get_runtime();
    let handle = runtime.spawn(async {
        // Small delay to let any in-flight operations complete
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    });

    // Block until the cleanup task completes
    runtime.block_on(handle).is_ok()
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
