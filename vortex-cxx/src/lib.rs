// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{LazyLock, OnceLock};

use futures::executor::ThreadPool;
use tokio::runtime::Runtime;
use vortex::error::{VortexError, VortexExpect};

mod read;
mod write;

/// The tokio runtime for the write-side.
static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new()
        .map_err(VortexError::from)
        .vortex_expect("Failed to create tokio runtime")
});

/// The thread pool for the read-side.
static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Thread pool configuration for the read-side.
#[derive(Clone)]
struct ThreadPoolConfig {
    worker_threads: Option<usize>,
}

impl ThreadPoolConfig {
    const fn new() -> Self {
        Self {
            worker_threads: None,
        }
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Get or initialize the thread pool with the default settings
fn get_thread_pool() -> &'static ThreadPool {
    THREAD_POOL.get_or_init(|| {
        create_thread_pool_with_config(&ThreadPoolConfig::default())
            .vortex_expect("Cannot start thread pool")
    })
}

/// Create a thread pool with the given configuration
fn create_thread_pool_with_config(config: &ThreadPoolConfig) -> Result<ThreadPool, std::io::Error> {
    let mut builder = ThreadPool::builder();

    if let Some(worker_threads) = config.worker_threads {
        builder.pool_size(worker_threads);
    }

    builder.create()
}

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {
    extern "Rust" {
        fn configure_thread_pool(worker_threads: usize) -> Result<()>;
    }
}

/// Configure the read-side thread pool with the specified number of worker threads
///
/// If the thread pool has already been initialized, this function will return an error.
fn configure_thread_pool(
    worker_threads: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Check if thread pool has already been initialized
    if THREAD_POOL.get().is_some() {
        return Err("Thread pool has already been initialized. ".into());
    }

    THREAD_POOL.get_or_init(|| {
        create_thread_pool_with_config(&ThreadPoolConfig {
            worker_threads: Some(worker_threads),
        })
        .vortex_expect("Cannot start thread pool")
    });

    Ok(())
}

// Workaround to conditionally generate bindings of the test function *and* compile the test function: https://github.com/dtolnay/cxx/issues/1325
// This is done with CMakeLists.txt together.
#[cfg(feature = "gen_test_data")]
mod gen_test_data;
