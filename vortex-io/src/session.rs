// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_error::VortexExpect;
use vortex_session::SessionExt;

use crate::runtime::Handle;

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
    /// Returns a handle for this session's runtime.
    fn handle(&self) -> Handle {
        self.get::<RuntimeSession>().handle
                .as_ref()
                .vortex_expect("Runtime handle not configured in Vortex session. Please setup a `CurrentThreadRuntime`, or configure the session for `with_tokio`.")
                .clone()
    }

    /// Configure the runtime session to use the application's Tokio runtime.
    ///
    /// On non-wasm32 targets, this uses CPUSegregatedRuntime which separates
    /// CPU-bound work from I/O to prevent I/O starvation. On wasm32, it uses
    /// the standard TokioRuntime.
    ///
    /// For example, if the application is launched using `#[tokio::main]`.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    fn with_tokio(self) -> Self {
        todo!()
        // self.get_mut::<RuntimeSession>().handle =
        //     Some(crate::runtime::CPUSegregatedRuntime::current());
        // self
    }

    /// Configure the runtime session to use the application's Tokio runtime.
    ///
    /// For example, if the application is launched using `#[tokio::main]`.
    #[cfg(all(feature = "tokio", target_arch = "wasm32"))]
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

    /// Configure the runtime session to use a CPUSegregatedRuntime.
    ///
    /// This separates CPU-bound work from I/O to prevent I/O starvation.
    /// The CPU pool will reserve 2 cores for I/O by default.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    fn with_cpu_segregated_runtime(self) -> Self {
        todo!()
        // self.get_mut::<RuntimeSession>().handle =
        //     Some(crate::runtime::CPUSegregatedRuntime::current());
        // self
    }

    /// Configure the runtime session to use a CPUSegregatedRuntime,
    /// reserving the specified number of cores for I/O.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    fn with_cpu_segregated_runtime_reserved(self, reserved_for_io: usize) -> Self {
        todo!()
        // self.get_mut::<RuntimeSession>().handle = Some(
        //     crate::runtime::CPUSegregatedRuntime::current_with_reserved(reserved_for_io),
        // );
        // self
    }
}
impl<S: SessionExt> RuntimeSessionExt for S {}
