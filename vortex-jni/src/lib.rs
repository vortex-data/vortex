// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;
use vortex::error::VortexExpect;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

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
    Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Failed to build Tokio runtime")
});

/// Shared Vortex session for the JNI instance.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| VortexSession::default().with_tokio());

/// Helper to block the JNI thread on asynchronous tasks, with added logging.
pub(crate) fn block_on<F: Future>(task_id: &str, future: F) -> F::Output {
    let start = std::time::Instant::now();
    let result = TOKIO_RUNTIME.block_on(future);
    let elapsed = start.elapsed();

    log::debug!("async task execution id=\"{task_id}\" duration={elapsed:?}");

    result
}

/// Spawn a new asynchronous task onto the global async runtime for the JNI.
pub(crate) fn spawn<F: Future + Send + 'static>(future: F) -> JoinHandle<F::Output>
where
    F::Output: Send + 'static,
{
    TOKIO_RUNTIME.spawn(future)
}
