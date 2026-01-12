// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;

/// Marker trait for types that can be executed.
pub trait Executable: Sized {
    /// Options for execution.
    type Options: Default;

    /// Execute the given array to produce [`Self`].
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Self::execute_with_options(array, ctx, Self::Options::default())
    }

    fn execute_with_options(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
        options: Self::Options,
    ) -> VortexResult<Self>;
}

impl dyn Array + '_ {
    /// Execute this array to produce an instance of `E`.
    pub fn execute<E: Executable>(self: Arc<Self>, ctx: &mut ExecutionCtx) -> VortexResult<E> {
        E::execute(self, ctx)
    }

    /// Execute this array to produce an instance of `E` with options.
    pub fn execute_with_options<E: Executable>(
        self: Arc<Self>,
        ctx: &mut ExecutionCtx,
        options: E::Options,
    ) -> VortexResult<E> {
        E::execute_with_options(self, ctx, options)
    }
}

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

/// Recursively execute the array to canonical form.
/// This will replace the recursive usage of `to_canonical()`.
/// An `ExecutionCtx` is will be used to limit access to buffers.
impl Executable for Canonical {
    type Options = ();

    fn execute_with_options(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
        _options: Self::Options,
    ) -> VortexResult<Self> {
        // Try and dispatch to a child that can optimize execution.
        for (child_idx, child) in array.children().iter().enumerate() {
            if let Some(result) = child
                .encoding()
                .as_dyn()
                .execute_canonical_parent(child, &array, child_idx, ctx)?
            {
                tracing::debug!(
                    "Executed array {} via child {} optimization.",
                    array.encoding_id(),
                    child.encoding_id()
                );
                return Ok(result);
            }
        }

        // Otherwise fall back to the default execution.
        array.encoding().as_dyn().execute_canonical(&array, ctx)
    }
}

/// Execute the array and return a [`CanonicalOutput`].
///
/// This may short-circuit for constant arrays, returning [`CanonicalOutput::Constant`]
/// instead of fully materializing the array.
impl Executable for CanonicalOutput {
    type Options = ();

    fn execute_with_options(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
        _options: Self::Options,
    ) -> VortexResult<Self> {
        // Attempt to short-circuit constant arrays.
        if let Some(constant) = array.as_opt::<ConstantVTable>() {
            return Ok(CanonicalOutput::Constant(ConstantArray::new(
                constant.scalar().clone(),
                constant.len(),
            )));
        }

        tracing::debug!("Executing array {}:\n{}", array, array.display_tree());
        Ok(CanonicalOutput::Array(array.execute(ctx)?))
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
