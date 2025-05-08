use tokio::runtime::{Builder, Runtime};

use crate::vortex_panic;

/// Creates a Tokio runtime based on the provided thread count configuration.
///
/// # Arguments
///
/// * `threads` - Optional number of threads to use:
///   * `None` - Uses Tokio's default multi-thread runtime
///   * `Some(0)` - Panics, as 0 threads is invalid
///   * `Some(1)` - Creates a single-threaded runtime
///   * `Some(n)` - Creates a multi-threaded runtime with `n` worker threads
///
/// # Returns
///
/// A configured Tokio runtime
///
/// # Panics
///
/// Panics if `threads` is `Some(0)` or if runtime creation fails
pub fn new_tokio_runtime(threads: Option<usize>) -> Runtime {
    match threads {
        Some(0) => vortex_panic!("Can't use 0 threads for runtime"),
        Some(1) => Builder::new_current_thread().enable_all().build(),
        Some(n) => Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build(),
        None => Builder::new_multi_thread().enable_all().build(),
    }
    .expect("Failed building the Runtime")
}
