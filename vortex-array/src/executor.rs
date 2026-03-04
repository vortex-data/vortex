// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::IntoArray;

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

impl dyn DynArray + '_ {
    /// Execute this array to produce an instance of `E`.
    ///
    /// See the [`Executable`] implementation for details on how this execution is performed.
    pub fn execute<E: Executable>(self: Arc<Self>, ctx: &mut ExecutionCtx) -> VortexResult<E> {
        E::execute(self, ctx)
    }

    /// Execute this array, labeling the execution step with a name for tracing.
    pub fn execute_as<E: Executable>(
        self: Arc<Self>,
        _name: &'static str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<E> {
        E::execute(self, ctx)
    }
}

/// Execution context for batch CPU compute.
///
/// Accumulates a trace of execution steps. Individual steps are logged at TRACE level for
/// real-time following, and the full trace is dumped at DEBUG level when the context is dropped.
pub struct ExecutionCtx {
    id: usize,
    session: VortexSession,
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
            let formatted = format!(" - {msg}");
            tracing::trace!("exec[{}]: {formatted}", self.id);
            self.ops.push(formatted);
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
        if !self.ops.is_empty() && tracing::enabled!(tracing::Level::DEBUG) {
            // Unlike itertools `.format()` (panics in 0.14 on second format)
            struct FmtOps<'a>(&'a [String]);
            impl Display for FmtOps<'_> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    for (i, op) in self.0.iter().enumerate() {
                        if i > 0 {
                            f.write_str("\n")?;
                        }
                        f.write_str(op)?;
                    }
                    Ok(())
                }
            }
            tracing::debug!("exec[{}] trace:\n{}", self.id, FmtOps(&self.ops));
        }
    }
}

/// Executing an [`ArrayRef`] into an [`ArrayRef`] is the atomic execution loop within Vortex.
///
/// It attempts to take the smallest possible step of execution such that the returned array
/// is incrementally more "executed" than the input array. In other words, it is closer to becoming
/// a canonical array.
///
/// The execution steps are as follows:
/// 0. Check for canonical.
/// 1. Attempt to call `reduce_parent` on each child.
/// 2. Attempt to `reduce` the array with metadata-only optimizations.
/// 3. Attempt to call `execute_parent` on each child.
/// 4. Call `execute` on the array itself.
///
/// Most users will not call this method directly, instead preferring to specify an executable
/// target such as [`crate::Columnar`], [`Canonical`], or any of the canonical array types (such as
/// [`crate::arrays::PrimitiveArray`]).
impl Executable for ArrayRef {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // 0. Check for canonical
        if let Some(canonical) = array.as_opt::<AnyCanonical>() {
            ctx.log(format_args!("-> canonical {}", array));
            return Ok(Canonical::from(canonical).into_array());
        }

        // 1. reduce (metadata-only rewrites)
        if let Some(reduced) = array.vtable().reduce(&array)? {
            ctx.log(format_args!("reduce: rewrote {} -> {}", array, reduced));
            reduced.statistics().inherit_from(array.statistics());
            return Ok(reduced);
        }

        // 2. reduce_parent (child-driven metadata-only rewrites)
        for child_idx in 0..array.nchildren() {
            let child = array.nth_child(child_idx).vortex_expect("checked length");
            if let Some(reduced_parent) = child.vtable().reduce_parent(&child, &array, child_idx)? {
                ctx.log(format_args!(
                    "reduce_parent: child[{}]({}) rewrote {} -> {}",
                    child_idx,
                    child.encoding_id(),
                    array,
                    reduced_parent
                ));
                reduced_parent.statistics().inherit_from(array.statistics());
                return Ok(reduced_parent);
            }
        }

        // 3. execute_parent (child-driven optimized execution)
        for child_idx in 0..array.nchildren() {
            let child = array.nth_child(child_idx).vortex_expect("checked length");
            if let Some(executed_parent) = child
                .vtable()
                .execute_parent(&child, &array, child_idx, ctx)?
            {
                ctx.log(format_args!(
                    "execute_parent: child[{}]({}) rewrote {} -> {}",
                    child_idx,
                    child.encoding_id(),
                    array,
                    executed_parent
                ));
                executed_parent
                    .statistics()
                    .inherit_from(array.statistics());
                return Ok(executed_parent);
            }
        }

        // 4. execute (optimized execution)
        ctx.log(format_args!("executing {}", array));
        let array = array
            .vtable()
            .execute(&array, ctx)
            .map(|c| c.into_array())?;
        array.statistics().inherit_from(array.statistics());
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
