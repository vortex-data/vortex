use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};
use vortex::error::{VortexError, VortexExpect};

pub static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(VortexError::IOError)
        .vortex_expect("tokio runtime must not fail to start")
});
