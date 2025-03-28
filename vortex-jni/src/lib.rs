use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};
use vortex::error::VortexExpect;

macro_rules! throw_runtime {
    ($($tt:tt)*) => {
        return Err(vortex::error::vortex_err!($($tt)*).into());
    };
}

mod array;
mod array_stream;
mod dtype;
mod errors;
mod file;
mod logging;

// Shared Tokio runtime for all of the async operations in this package.
static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Failed to build Tokio runtime")
});

/// Helper to block the JNI thread on asynchronous tasks, with added logging.
pub(crate) fn block_on<F: Future>(task_id: &str, future: F) -> F::Output {
    let start = std::time::Instant::now();
    let result = TOKIO_RUNTIME.block_on(future);
    let elapsed = start.elapsed();

    log::debug!("async task execution id=\"{task_id}\" duration={elapsed:?}");

    result
}
