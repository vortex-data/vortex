#![allow(clippy::expect_used)]
use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};
use vortex::error::VortexExpect;

mod array;
mod array_stream;
mod dtype;
mod errors;
mod file;

// Shared Tokio runtime for all of the async operations in this package.
pub(crate) static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Failed to build Tokio runtime")
});
