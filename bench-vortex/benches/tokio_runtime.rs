use lazy_static::lazy_static;
use tokio::runtime::Runtime;
use vortex::error::{VortexError, VortexExpect};

lazy_static! {
    pub static ref TOKIO_RUNTIME: Runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(VortexError::IOError)
        .vortex_expect("tokio runtime must not fail to start");
}
