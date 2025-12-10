// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kernels represent the CPU physical plan for array execution.

mod closure;
mod ready;
mod validate;

use std::fmt::Debug;

pub use closure::*;
pub use ready::*;
pub use validate::*;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_vector::Vector;

/// A boxed reference to a kernel.
pub type KernelRef = Box<dyn Kernel>;

/// A trait representing the physical CPU execution of an array tree.
pub trait Kernel: 'static + Send + Debug {
    /// Execute the kernel and produce a vector result.
    fn execute(self: Box<Self>) -> VortexResult<Vector>;

    /// Push a selection mask through this kernel.
    ///
    /// Return `Ok(None)` if the filter cannot be pushed down.
    fn push_down_filter(self: Box<Self>, selection: &Mask) -> VortexResult<PushDownResult>;
}

pub enum PushDownResult {
    Pushed(KernelRef),
    NotPushed(KernelRef),
}

/// Bind context for batch array compute.
pub struct BindCtx {
    session: VortexSession,
}

impl BindCtx {
    /// Create a new execution context with the given session.
    pub(crate) fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}
