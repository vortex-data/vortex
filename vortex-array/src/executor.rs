// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_vector::Datum;
use vortex_vector::Vector;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ConstantVTable;

/// Execution context for batch CPU compute.
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

/// Executor for exporting a Vortex [`Vector`] or [`Datum`] from an [`ArrayRef`].
pub trait VectorExecutor {
    /// Recursively execute the array.
    fn execute(&self, ctx: &mut ExecutionCtx) -> VortexResult<Vector>;

    /// Execute the array and return the resulting datum.
    fn execute_datum(&self, session: &VortexSession) -> VortexResult<Datum>;
    /// Execute the array and return the resulting vector.
    fn execute_vector(&self, session: &VortexSession) -> VortexResult<Vector>;
}

impl VectorExecutor for ArrayRef {
    fn execute(&self, ctx: &mut ExecutionCtx) -> VortexResult<Vector> {
        // Try and dispatch to a child that can optimize execution.
        for (child_idx, child) in self.children().iter().enumerate() {
            if let Some(result) = child
                .encoding()
                .as_dyn()
                .execute_parent(child, self, child_idx, ctx)?
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
        self.encoding().as_dyn().execute(self, ctx)
    }

    fn execute_datum(&self, session: &VortexSession) -> VortexResult<Datum> {
        // Attempt to short-circuit constant arrays.
        if let Some(constant) = self.as_opt::<ConstantVTable>() {
            return Ok(Datum::Scalar(constant.scalar().to_vector_scalar()));
        }

        let mut ctx = ExecutionCtx::new(session.clone());
        tracing::debug!("Executing array {}:\n{}", self, self.display_tree());
        Ok(Datum::Vector(self.execute(&mut ctx)?))
    }

    fn execute_vector(&self, session: &VortexSession) -> VortexResult<Vector> {
        let len = self.len();
        Ok(self.execute_datum(session)?.unwrap_into_vector(len))
    }
}
