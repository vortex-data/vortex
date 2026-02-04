// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::ConstantVTable;

/// Marker trait for types that an [`ArrayRef`] can be executed into.
///
/// Implementors must provide an implementation of `execute` that takes
/// an [`ArrayRef`] and an [`ExecutionCtx`], and produces an instance of the
/// implementor type.
///
/// Users should use the `Array::execute` or `Array::execute_as` methods
pub trait Executable: Sized {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self>;
}

impl dyn Array + '_ {
    /// Execute this array to produce an instance of `E`.
    ///
    /// See the [`Executable`] implementation for details on how this execution is performed.
    pub fn execute<E: Executable>(self: Arc<Self>, ctx: &mut ExecutionCtx) -> VortexResult<E> {
        ctx.log_entry(
            &self,
            format_args!("execute<{}> {}", short_type_name::<E>(), self),
        );
        E::execute(self, ctx)
    }

    /// Execute this array, labeling the execution step with a name for tracing.
    pub fn execute_as<E: Executable>(
        self: Arc<Self>,
        name: &'static str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<E> {
        ctx.log_entry(
            &self,
            format_args!("{}: execute<{}> {}", name, short_type_name::<E>(), self),
        );
        E::execute(self, ctx)
    }
}

fn short_type_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}

/// Execution context for batch CPU compute.
///
/// Accumulates a trace of execution steps. Individual steps are logged at TRACE level for
/// real-time following, and the full trace is dumped at DEBUG level when the context is dropped.
pub struct ExecutionCtx {
    id: usize,
    session: VortexSession,
    depth: usize,
    ops: Vec<String>,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub fn new(session: VortexSession) -> Self {
        static EXEC_CTX_ID: AtomicUsize = AtomicUsize::new(0);
        let id = EXEC_CTX_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            id,
            session,
            depth: 0,
            ops: Vec::new(),
        }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Log an execution step at the current depth.
    ///
    /// Steps are accumulated and dumped as a single trace on Drop at DEBUG level.
    /// Individual steps are also logged at TRACE level for real-time following.
    ///
    /// Use the [`format_args!`] macro to create the `msg` argument.
    pub fn log(&mut self, msg: fmt::Arguments<'_>) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            let indent = "  ".repeat(self.depth);
            let formatted = format!("{indent}{msg}");
            tracing::trace!("exec[{}]: {formatted}", self.id);
            self.ops.push(formatted);
        }
    }

    /// Log an execution entry point. On the first call into this context, the full
    /// `display_tree` of the array is included so the starting state is visible.
    fn log_entry(&mut self, array: &dyn Array, msg: fmt::Arguments<'_>) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            if self.ops.is_empty() {
                self.log(format_args!("{msg}\n{}", array.display_tree()));
            } else {
                self.log(msg);
            }
        }
    }
}

impl Display for ExecutionCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exec[{}]", self.id)
    }
}

impl Drop for ExecutionCtx {
    fn drop(&mut self) {
        if !self.ops.is_empty()
            && tracing::enabled!(tracing::Level::DEBUG) {
                let trace = self
                    .ops
                    .iter()
                    .map(|op| format!("    - {}", op))
                    .format("\n");
                tracing::debug!("exec[{}] trace:\n{}", self.id, trace);
            }
    }
}

/// Executing an [`ArrayRef`] into an [`ArrayRef`] is the atomic execution loop within Vortex.
///
/// It attempts to take the smallest possible step of execution such that the returned array
/// is incrementally more "executed" than the input array. In other words, it is closer to becoming
/// a constant value or a canonical array.
///
/// The execution steps are as follows:
/// 0. Check for termination conditions: constant or canonical.
/// 1. Attempt to call `reduce_parent` on each child.
/// 2. Attempt to `reduce` the array with metadata-only optimizations.
/// 3. Attempt to call `execute_parent` on each child.
/// 4. Call `execute` on the array itself.
///
/// Most users will not call this method directly, instead preferring to specify an executable
/// target such as [`Columnar`], [`Canonical`], or any of the canonical array types (such as
/// [`crate::arrays::PrimitiveArray`]).
impl Executable for ArrayRef {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // 0. Check for termination conditions
        if let Some(constant) = array.as_opt::<ConstantVTable>() {
            ctx.log(format_args!("-> constant({})", constant.scalar()));
            return Ok(constant.to_array());
        }
        if let Some(canonical) = array.as_opt::<AnyCanonical>() {
            ctx.log(format_args!("-> canonical {}", array));
            return Ok(Canonical::from(canonical).into_array());
        }

        // 1. reduce_parent (child-driven metadata-only rewrites)
        for (child_idx, child) in array.children().iter().enumerate() {
            if let Some(reduced_parent) = child.vtable().reduce_parent(child, &array, child_idx)? {
                ctx.log(format_args!(
                    "reduce_parent: child[{}]({}) rewrote {} -> {}",
                    child_idx,
                    child.encoding_id(),
                    array,
                    reduced_parent
                ));
                return Ok(reduced_parent);
            }
        }

        // 2. reduce (metadata-only rewrites)
        if let Some(reduced) = array.vtable().reduce(&array)? {
            ctx.log(format_args!("reduce: rewrote {} -> {}", array, reduced));
            return Ok(reduced);
        }

        // 3. execute_parent (child-driven optimized execution)
        for (child_idx, child) in array.children().iter().enumerate() {
            if let Some(executed_parent) = child
                .vtable()
                .execute_parent(child, &array, child_idx, ctx)?
            {
                ctx.log(format_args!(
                    "execute_parent: child[{}]({}) rewrote {} -> {}",
                    child_idx,
                    child.encoding_id(),
                    array,
                    executed_parent
                ));
                return Ok(executed_parent);
            }
        }

        // 4. execute (optimized execution)
        // TODO(ngates): move over to calling Array::execute
        ctx.log(format_args!("canonicalize {}", array));
        let array = array
            .vtable()
            .canonicalize(&array, ctx)
            .map(|c| c.into_array())?;
        ctx.log(format_args!("-> {}", array.as_ref()));

        Ok(array)
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
