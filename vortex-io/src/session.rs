// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::current::CurrentThreadRuntime;
use crate::runtime::BlockingRuntime;
use crate::runtime::Handle;
use std::fmt::Debug;
use vortex_session::SessionExt;

/// Session state for Vortex async runtimes.
pub struct RuntimeSession {
    runtime: Runtime,
}

/// The choices for the runtime used in a session.
enum Runtime {
    /// A specific Tokio runtime.
    #[cfg(feature = "tokio")]
    Tokio(crate::runtime::tokio::TokioRuntime),
    /// Whatever the current Tokio runtime is.
    #[cfg(feature = "tokio")]
    TokioCurrent,
    /// A current-thread runtime.
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
        // TODO(ngates): is this sensible? How does the caller even execute this runtime?
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
            #[cfg(feature = "tokio")]
            Runtime::TokioCurrent => crate::runtime::tokio::TokioRuntime::current(),
            Runtime::CurrentThread(rt) => rt.handle(),
        }
    }

    /// Configure the runtime session to use the application's Tokio runtime.
    ///
    /// For example, if the application is launched using `#[tokio::main]`.
    #[cfg(feature = "tokio")]
    fn with_tokio(self) -> Self {
        self.get_mut::<RuntimeSession>().runtime = Runtime::TokioCurrent;
        self
    }

    /// Configure the runtime session to use a specific Tokio handle.
    #[cfg(feature = "tokio")]
    fn with_tokio_handle(self, handle: tokio::runtime::Handle) -> Self {
        self.get_mut::<RuntimeSession>().runtime =
            Runtime::Tokio(crate::runtime::tokio::TokioRuntime::from(handle));
        self
    }

    /// Configure the runtime session to use a specific current-thread runtime.
    fn with_current_thread_runtime(self, runtime: CurrentThreadRuntime) -> Self {
        self.get_mut::<RuntimeSession>().runtime = Runtime::CurrentThread(runtime);
        self
    }
}
impl<S: SessionExt> RuntimeSessionExt for S {}
