// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;

use vortex_error::VortexExpect;
use vortex_session::SessionExt;
use vortex_session::SessionVar;

use crate::runtime::Handle;

/// Session state for Vortex async runtimes.
pub struct RuntimeSession {
    handle: Option<Handle>,
}

impl SessionVar for RuntimeSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Default for RuntimeSession {
    fn default() -> Self {
        Self {
            handle: Handle::find(),
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
        use crate::runtime::tokio::TokioRuntime;
        self.get_mut::<RuntimeSession>().handle = Some(TokioRuntime::current());
        self
    }

    /// Configure the runtime session to use a specific Vortex runtime handle.
    fn with_handle(self, handle: Handle) -> Self {
        self.get_mut::<RuntimeSession>().handle = Some(handle);
        self
    }
}
impl<S: SessionExt> RuntimeSessionExt for S {}
