// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::clone::Clone;
use std::sync::LazyLock;

use tokio::runtime::Builder;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::error::VortexExpect;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

macro_rules! throw_runtime {
    ($($tt:tt)*) => {
        return Err(vortex::error::vortex_err!($($tt)*).into())
    };
}

mod array;
mod array_iter;
mod dtype;
mod errors;
mod file;
mod logging;
mod object_store;
mod writer;

// Shared Tokio runtime for all the async operations in this package.
static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    // TODO: propagate this error up instead of expecting
    Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Failed to build Tokio runtime")
});
static RUNTIME: LazyLock<TokioRuntime> =
    LazyLock::new(|| TokioRuntime::from(TOKIO_RUNTIME.handle().clone()));
/// Shared Vortex session for the JNI instance.
static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(RUNTIME.handle()));
