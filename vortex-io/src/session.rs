// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::Handle;
use std::fmt::Debug;
use vortex_error::VortexExpect;
use vortex_session::SessionExt;

/// Session state for Vortex async runtimes.
pub struct RuntimeSession {
    handle: Option<Handle>,
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
    /// Get the runtime handle for this session.
    fn handle(&self) -> Handle {
        self.get::<RuntimeSession>().handle
            .clone()
            .vortex_expect("Runtime session has not been configured with a handle, please call `RuntimeSessionExt::with_tokio` or `RuntimeSessionExt::set_handle` to set one up")
    }

    /// Set the runtime handle for this session.
    ///
    /// Required only when the session was not initialized within a Tokio context.
    fn with_handle(self, handle: Handle) -> Self {
        self.get_mut::<RuntimeSession>().handle = Some(handle);
        self
    }

    /// Configure the runtime session to use Tokio.
    #[cfg(feature = "tokio")]
    fn with_tokio(self) -> Self {
        self.with_handle(crate::runtime::tokio::TokioRuntime::current())
    }
}
impl<S: SessionExt> RuntimeSessionExt for S {}
