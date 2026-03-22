// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_session::SessionExt;

use crate::runtime::Executor;
use crate::runtime::Handle;

/// Session state for Vortex async runtimes.
pub struct RuntimeSession {
    handle: Option<Handle>,
    /// Strong reference that keeps the executor alive for the lifetime of the session.
    /// The [`Handle`] only holds a [`Weak`] reference, so without this the executor
    /// would be dropped immediately after construction.
    _executor: Option<Arc<dyn Executor>>,
}

impl Default for RuntimeSession {
    fn default() -> Self {
        Self {
            handle: Handle::find(),
            _executor: None,
        }
    }
}

impl Debug for RuntimeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeSession").finish_non_exhaustive()
    }
}

/// Extension trait for accessing runtime session data.
pub trait RuntimeSessionExt: SessionExt {
    /// Returns a handle for this session's runtime.
    fn handle(&self) -> Handle {
        self.get::<RuntimeSession>().handle
                .as_ref()
                .vortex_expect("Runtime handle not configured in Vortex session. Please setup a `CurrentThreadRuntime`, or configure the session for `with_tokio`.")
                .clone()
    }

    /// Configure the runtime session to use the application's Tokio runtime.
    ///
    /// For example, if the application is launched using `#[tokio::main]`.
    #[cfg(feature = "tokio")]
    fn with_tokio(self) -> Self {
        self.get_mut::<RuntimeSession>().handle =
            Some(crate::runtime::tokio::TokioRuntime::current());
        self
    }

    /// Configure the runtime session to use a specific Vortex runtime handle.
    fn with_handle(self, handle: Handle) -> Self {
        self.get_mut::<RuntimeSession>().handle = Some(handle);
        self
    }

    /// Configure the runtime session to use tokio for async I/O but a dedicated
    /// pinned thread pool for CPU-bound decode work (`spawn_cpu`).
    ///
    /// This avoids tokio's work-stealing for CPU-heavy operations (bitunpacking,
    /// FoR, dictionary gather, etc.), keeping data cache-local on the worker that
    /// started the decode.
    ///
    /// The pool is sized to `available_parallelism - 1` by default, leaving one
    /// core for the tokio I/O runtime.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use vortex::session::VortexSession;
    /// use vortex::io::session::RuntimeSessionExt;
    ///
    /// let session = VortexSession::default().with_pinned_cpu_pool();
    /// ```
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    fn with_pinned_cpu_pool(self) -> Self {
        let pool =
            Arc::new(crate::runtime::pinned_pool::PinnedCpuPool::with_available_parallelism());
        self.with_pinned_cpu_pool_sized(pool)
    }

    /// Like [`with_pinned_cpu_pool`][Self::with_pinned_cpu_pool] but accepts a
    /// pre-configured pool, allowing control over the number of worker threads.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use vortex::session::VortexSession;
    /// use vortex::io::session::RuntimeSessionExt;
    /// use vortex::io::runtime::pinned_pool::PinnedCpuPool;
    ///
    /// let pool = Arc::new(PinnedCpuPool::new(4));
    /// let session = VortexSession::default().with_pinned_cpu_pool_sized(pool);
    /// ```
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    fn with_pinned_cpu_pool_sized(
        self,
        pool: Arc<crate::runtime::pinned_pool::PinnedCpuPool>,
    ) -> Self {
        let tokio_handle = tokio::runtime::Handle::current();
        let tokio_executor: Arc<dyn Executor> = Arc::new(tokio_handle);
        let pinned: Arc<dyn Executor> = Arc::new(crate::runtime::pinned_pool::PinnedExecutor::new(
            pool,
            tokio_executor,
        ));
        let handle = Handle::new(Arc::downgrade(&pinned));
        // Store the strong Arc so the executor outlives the Weak reference in Handle.
        self.get_mut::<RuntimeSession>()._executor = Some(pinned);
        self.with_handle(handle)
    }
}
impl<S: SessionExt> RuntimeSessionExt for S {}
