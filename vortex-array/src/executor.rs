// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;

/// The result of executing an array, which can either be a constant (scalar repeated)
/// or a fully materialized canonical array.
///
/// This allows execution to short-circuit when the array is constant, avoiding
/// unnecessary expansion of scalar values.
#[derive(Debug, Clone)]
pub enum CanonicalOutput {
    /// A constant array representing a scalar value repeated to a given length.
    Constant(ConstantArray),
    /// A fully materialized canonical array.
    Array(Canonical),
}

/// Execution context for batch CPU compute.
pub struct ExecutionCtx {
    session: VortexSession,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

/// Executor for exporting Vortex arrays to canonical form.
pub trait VectorExecutor {
    /// Recursively execute the array to canonical form.
    /// This will replace the recursive usage of `to_canonical()`.
    /// An `ExecutionCtx` is will be used to limit access to buffers.
    fn execute(&self, ctx: &mut ExecutionCtx) -> VortexResult<Canonical>;

    /// Execute the array and return a [`CanonicalOutput`].
    ///
    /// This may short-circuit for constant arrays, returning [`CanonicalOutput::Constant`]
    /// instead of fully materializing the array.
    fn execute_output(&self, ctx: &mut ExecutionCtx) -> VortexResult<CanonicalOutput>;
}

impl VectorExecutor for ArrayRef {
    fn execute(&self, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        // Try and dispatch to a child that can optimize execution.
        for (child_idx, child) in self.children().iter().enumerate() {
            if let Some(result) = child
                .encoding()
                .as_dyn()
                .execute_canonical_parent(child, self, child_idx, ctx)?
            {
                tracing::debug!(
                    "Executed array {} via child {} optimization.",
                    self.encoding_id(),
                    child.encoding_id()
                );
                return Ok(result);
            }
        }

        // Otherwise fall back to the default execution.
        self.encoding().as_dyn().execute_canonical(self, ctx)
    }

    fn execute_output(&self, ctx: &mut ExecutionCtx) -> VortexResult<CanonicalOutput> {
        // Attempt to short-circuit constant arrays.
        if let Some(constant) = self.as_opt::<ConstantVTable>() {
            return Ok(CanonicalOutput::Constant(ConstantArray::new(
                constant.scalar().clone(),
                constant.len(),
            )));
        }

        tracing::debug!("Executing array {}:\n{}", self, self.display_tree());
        Ok(CanonicalOutput::Array(self.execute(ctx)?))
    }
}

/// Extension trait for creating an execution context from a session.
pub trait VortexSessionExecute {
    /// Create a new execution context from this session.
    fn create_execution_ctx(&self) -> ExecutionCtx;
}

impl VortexSessionExecute for VortexSession {
    fn create_execution_ctx(&self) -> ExecutionCtx {
        ExecutionCtx::new(self.clone())
    }
}
