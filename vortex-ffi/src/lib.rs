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

/// Global shared Tokio runtime state for all VortexSession instances.
///
/// ## Thread Safety
///
/// This runtime management system is fully thread-safe and designed for concurrent usage:
///
/// - **Reference Counting**: Uses `Arc<Runtime>` for safe sharing across threads
/// - **Mutex Protection**: `parking_lot::Mutex` provides exclusive access to state changes
/// - **Atomic Operations**: Session creation/destruction is race-free
/// - **Clean Shutdown**: `Arc::try_unwrap` ensures coordinated shutdown without races
///
/// ## Lifecycle
///
/// 1. **Lazy Initialization**: Runtime is created on first session creation
/// 2. **Reference Sharing**: Each session holds an `Arc<Runtime>` reference
/// 3. **Coordinated Cleanup**: Runtime shuts down only when all sessions are dropped
/// 4. **Manual Shutdown**: `vx_try_shutdown_runtime()` can force cleanup if safe
///
/// ## Concurrent Access Patterns
///
/// **Safe Concurrent Operations:**
/// - Multiple threads can create sessions simultaneously
/// - Session operations (file I/O, scanning) can run concurrently
/// - Runtime shutdown is safe even during concurrent session creation
///
/// **Thread Coordination:**
/// - Sessions created concurrently will share the same runtime instance
/// - Last session to be dropped will automatically trigger runtime cleanup
/// - Manual shutdown only succeeds when no sessions hold runtime references
static RUNTIME_STATE: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

/// Get or create the shared Tokio runtime for VortexSession instances.
///
/// ## Thread Safety
///
/// This function is fully thread-safe and can be called concurrently from multiple threads:
///
/// - **Lock Duration**: Minimized through clone-before-return pattern
/// - **Race Condition Prevention**: Mutex ensures atomic runtime creation
/// - **Memory Safety**: Arc ensures runtime lives as long as any session references it
///
/// ## Performance
///
/// - **First Call**: Creates runtime (expensive, ~10-50ms)
/// - **Subsequent Calls**: Clone Arc reference (cheap, ~nanoseconds)
/// - **Lock Contention**: Minimal due to short critical sections
///
/// ## Error Handling
///
/// Panics if runtime creation fails - this is intentional as runtime failure
/// indicates system-level issues that should abort the process.
pub(crate) fn get_runtime() -> Arc<Runtime> {
    let mut state = RUNTIME_STATE.lock();

    if let Some(runtime) = state.as_ref() {
        runtime.clone()
    } else {
        let runtime = Arc::new(
            runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .vortex_expect("Cannot start runtime - system may be out of resources"),
        );
        *state = Some(runtime.clone());
        runtime
    }
}

/// Attempt to shutdown the shared Tokio runtime if no active sessions exist.
///
/// ## Behavior
///
/// - **Success**: If no sessions hold runtime references, shuts down immediately  
/// - **Failure**: If sessions still exist, runtime continues running (no-op)
/// - **Blocking**: May block indefinitely if runtime has active tasks
///
/// ## Thread Safety
///
/// This function is thread-safe but coordination is required for predictable behavior:
///
/// - **Race Condition Safe**: Uses atomic Arc operations to detect active references
/// - **Concurrent Calls**: Multiple threads can call this simultaneously (idempotent)
/// - **Session Creation Race**: If called during session creation, may fail to shutdown
///
/// ## When to Use
///
/// **Recommended:**
/// - Application shutdown after all sessions are explicitly dropped
/// - Clean exit in single-threaded applications
/// - Testing scenarios where resource cleanup is required
///
/// **Not Recommended:**
/// - Automatic cleanup during normal operation (sessions handle this)  
/// - Multithreaded environments where session lifecycle is unclear
/// - Performance-critical code paths (this function can block)
///
/// ## Example Usage
///
/// ```c
/// // C usage pattern
/// vx_session* session1 = vx_session_new();
/// vx_session* session2 = vx_session_new();  // Reuses same runtime
///
/// vx_session_free(session1);
/// vx_session_free(session2);  // Runtime auto-cleaned here
///
/// // Or explicit cleanup:
/// vx_try_shutdown_runtime();  // Only succeeds if no sessions active
/// ```
pub fn try_shutdown_runtime() {
    let mut state = RUNTIME_STATE.lock();

    if let Some(runtime) = state.take() {
        match Arc::try_unwrap(runtime) {
            // We have the only reference, safe to shut down
            Ok(runtime) => {
                // Runtime drops here, potentially blocking on active tasks
                drop(runtime);
            }
            // There are other live references, so put it back
            Err(runtime) => {
                *state = Some(runtime);
            }
        }
    }
    // If state was None, runtime was already shut down or never created
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
