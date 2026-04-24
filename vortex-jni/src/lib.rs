// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Java Native Interface bindings for Vortex.
//!
//! The JNI surface mirrors the C FFI in `vortex-ffi` closely. It exposes a small
//! session-oriented scan API (session → data source → scan → partition → Arrow
//! array stream) so that Java callers only see Arrow at the boundary.

use std::sync::LazyLock;

use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::current::CurrentThreadWorkerPool;

macro_rules! throw_runtime {
    ($($tt:tt)*) => {
        return Err(vortex::error::vortex_err!($($tt)*).into())
    };
}

mod data_source;
mod dtype;
mod errors;
mod expression;
mod file;
mod logging;
mod object_store;
mod runtime;
mod scan;
mod session;
mod writer;

/// Shared current-thread runtime backing every JNI call. Using a current-thread
/// runtime (as opposed to multi-thread Tokio) keeps the Java side responsible for
/// parallelism decisions — each partition is consumed on the caller's thread, and
/// writes are bounded by a small in-flight queue on the same thread.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

/// Shared worker pool that can drive [`RUNTIME`]'s executor in the background. Callers
/// configure its size through the `NativeRuntime.setWorkerThreads` JNI entry point. By
/// default the pool has zero workers — nothing is driven unless a Java thread calls
/// the blocking API or workers are added here.
pub(crate) static POOL: LazyLock<CurrentThreadWorkerPool> = LazyLock::new(|| RUNTIME.new_pool());
