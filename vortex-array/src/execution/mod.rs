// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod batch;
mod mask;
mod validity;

pub use batch::*;
pub use mask::*;
use vortex_session::VortexSession;

/// Execution context for batch array compute.
// NOTE(ngates): This context will eventually hold cached resources for execution, such as CSE
//  nodes, and may well eventually support a type-map interface for arrays to stash arbitrary
//  execution-related data.
pub struct ExecutionCtx {
    session: VortexSession,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub(crate) fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}
