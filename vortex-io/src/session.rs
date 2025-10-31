// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::current::CurrentThreadRuntime;
use crate::runtime::BlockingRuntime;
use crate::runtime::Handle;
use futures::Stream;
use std::fmt::Debug;
use vortex_session::SessionExt;

/// Session state for Vortex async runtimes.
pub struct RuntimeSession {
    runtime: Runtime,
}

/// The choices for the runtime used in a session.
enum Runtime {
    #[cfg(feature = "tokio")]
    Tokio(crate::runtime::tokio::TokioRuntime),
    CurrentThread(CurrentThreadRuntime),
}

impl Default for RuntimeSession {
    fn default() -> Self {
        #[cfg(feature = "tokio")]
        {
            use tokio::runtime::Handle as TokioHandle;
            if let Ok(h) = TokioHandle::try_current() {
                return RuntimeSession {
                    runtime: Runtime::Tokio(crate::runtime::tokio::TokioRuntime::new(h)),
                };
            }
        }

        // Otherwise, by default we use a current-thread runtime.
        Self {
            runtime: Runtime::CurrentThread(CurrentThreadRuntime::default()),
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
        let session = self.get::<RuntimeSession>();
        match &session.runtime {
            #[cfg(feature = "tokio")]
            Runtime::Tokio(rt) => rt.handle(),
            Runtime::CurrentThread(rt) => rt.handle(),
        }
    }

    /// Configure the runtime session to use Tokio.
    #[cfg(feature = "tokio")]
    fn with_tokio(self, handle: tokio::runtime::Handle) -> Self {
        self.get_mut::<RuntimeSession>().runtime =
            Runtime::Tokio(crate::runtime::tokio::TokioRuntime::from(handle));
        self
    }

    /// Configure the runtime session to use a specific current-thread runtime.
    fn with_current_thread_runtime(self, runtime: CurrentThreadRuntime) -> Self {
        self.get_mut::<RuntimeSession>().runtime = Runtime::CurrentThread(runtime);
        self
    }

    /// Use the session's runtime to block on a future.
    ///
    /// Note that care should be used to avoid deadlocks when using this method.
    fn block_on<F, R>(&self, fut: F) -> R
    where
        F: Future<Output = R>,
    {
        let session = self.get::<RuntimeSession>();
        match &session.runtime {
            #[cfg(feature = "tokio")]
            Runtime::Tokio(rt) => BlockingRuntime::block_on(rt, |_| fut),
            Runtime::CurrentThread(rt) => BlockingRuntime::block_on(rt, |_| fut),
        }
    }

    /// Use the session's runtime to block on a stream, returning a blocking iterator.
    ///
    /// Note that care should be used to avoid deadlocks when using this method.
    fn block_on_stream<'a, S, R>(&self, s: S) -> Box<dyn Iterator<Item = R> + Send + 'a>
    where
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        let session = self.get::<RuntimeSession>();
        match &session.runtime {
            #[cfg(feature = "tokio")]
            Runtime::Tokio(rt) => Box::new(BlockingRuntime::block_on_stream(rt, |_| s)),
            Runtime::CurrentThread(rt) => Box::new(BlockingRuntime::block_on_stream(rt, |_| s)),
        }
    }
}
impl<S: SessionExt> RuntimeSessionExt for S {}
