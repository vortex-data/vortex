// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array execution: the system that evaluates Vortex arrays into canonical form.
//!
//! # Overview
//!
//! In Vortex, operations like filter, slice, take, and scalar functions do not execute eagerly.
//! Instead, calling `array.filter(mask)` returns a [`FilterArray`] that wraps the original array
//! and the mask — a lightweight tree node representing "filter this array by this mask." The same
//! is true for [`SliceArray`], [`DictArray`] (take), [`ScalarFnArray`], and others.
//!
//! These wrapper arrays form an **expression tree**. The execution system walks this tree,
//! repeatedly simplifying it until it converges to a canonical (fully materialized) array like
//! [`PrimitiveArray`], [`BoolArray`], or [`VarBinViewArray`].
//!
//! # Execution loop
//!
//! A single step of execution is defined by the [`Executable`] implementation for [`ArrayRef`].
//! Each step attempts the **smallest possible transformation** to bring the array closer to
//! canonical form. The outer driver — [`Columnar::execute`] — calls this in a loop until
//! the result is either a canonical array or a [`ConstantArray`].
//!
//! Within a single step, the executor tries four phases in order:
//!
//! 1. **Canonical check** — if the array is already canonical, return it immediately.
//!
//! 2. **`reduce`** — ask the array itself to simplify using metadata alone, without reading
//!    any buffers. This is implemented by [`VTable::reduce`] and typically delegates to a
//!    [`ReduceRuleSet`]. For example, a [`ChunkedArray`] with a single chunk reduces to that
//!    chunk. A [`FilterArray`] with an all-true mask reduces to its child.
//!
//! 3. **`reduce_parent`** — for each child of the array, ask the *child's* VTable whether it
//!    can rewrite the parent. This is implemented by [`VTable::reduce_parent`] and delegates
//!    to a [`ParentRuleSet`]. This is the key mechanism for encoding-aware optimization: a
//!    child encoding can recognize its parent operation and produce a cheaper equivalent. For
//!    example, when a [`ConstantArray`] is the child of a [`FilterArray`], filtering a constant
//!    is trivial — just produce a new constant with the output length.
//!
//! 4. **`execute_parent`** — like `reduce_parent`, but the child is allowed to read buffers
//!    and perform real work. This is implemented by [`VTable::execute_parent`] and delegates
//!    to a [`ParentKernelSet`]. For example, a [`ChunkedArray`] inside a [`FilterArray`] can
//!    split the filter mask across its chunks and filter each one independently.
//!
//! 5. **`execute`** — the fallback. The array fully executes itself one step closer to
//!    canonical form. For a [`FilterArray`], this means executing its child to canonical form
//!    and then applying the filter. For a [`ChunkedArray`], this means canonicalizing all
//!    chunks and concatenating.
//!
//! Phases 2–4 return `Option<ArrayRef>` — `None` means "I can't help, try the next phase."
//! The first `Some` result wins and becomes the input for the next iteration of the loop.
//!
//! # Writing rules and kernels for a new encoding
//!
//! When you add a new array encoding, you participate in this system by implementing the
//! relevant VTable methods and registering rules and kernels.
//!
//! ## `reduce` — self-simplification rules
//!
//! Implement [`ArrayReduceRule`] to define metadata-only rewrites for your encoding. These
//! rules examine the array and its structure without touching buffers.
//!
//! Register them as a [`ReduceRuleSet`] and call `RULES.evaluate(array)` from your
//! [`VTable::reduce`] implementation. Examples:
//!
//! - [`ChunkedArray`] with 0 chunks → empty canonical array
//! - [`ChunkedArray`] with 1 chunk → unwrap to the single chunk
//! - [`FilterArray`] with `AllTrue` mask → return the child unchanged
//!
//! ## `reduce_parent` — rewriting the parent without buffers
//!
//! Implement [`ArrayParentReduceRule<V>`] where `V` is your encoding's VTable. Each rule
//! specifies a `type Parent: Matcher` that constrains which parent array type the rule applies
//! to. The `Matcher` trait (via `try_match`) gives you a strongly-typed view of the parent.
//!
//! Register them in a [`ParentRuleSet`] and call `PARENT_RULES.evaluate(...)` from
//! [`VTable::reduce_parent`]. Examples:
//!
//! - A [`ConstantArray`] child of a [`FilterArray`] → just resize the constant to the
//!   filter's output length (no data to actually filter).
//! - A [`ConstantArray`] child of a `SliceArray` → resize to the slice length.
//! - A [`ChunkedArray`] child of a [`ScalarFnArray`] → push the scalar function down into
//!   each chunk, producing a new [`ChunkedArray`] of [`ScalarFnArray`]s.
//!
//! ## `execute_parent` — optimized execution with buffer access
//!
//! Implement [`ExecuteParentKernel<V>`] for kernels that need to read data. Like reduce rules,
//! each kernel specifies a `type Parent: Matcher`. Register them in a [`ParentKernelSet`] and
//! call `PARENT_KERNELS.execute(...)` from [`VTable::execute_parent`]. Examples:
//!
//! - A [`ChunkedArray`] child of a [`FilterArray`] → split the mask by chunk boundaries
//!   and filter each chunk separately.
//! - A [`ChunkedArray`] child of a `SliceArray` → compute which chunks overlap the slice
//!   range and slice them.
//!
//! ## `execute` — the fallback
//!
//! Implement [`VTable::execute`] as the last resort. This should make progress toward canonical
//! form — for example, by decoding the encoding into a simpler representation. The executor
//! will call this only when no reduce rule or parent kernel applied.
//!
//! # Concrete example: `Filter(Chunked([chunk_0, chunk_1]))`
//!
//! Consider filtering a chunked array with two chunks. The expression tree looks like:
//!
//! ```text
//! FilterArray { child: ChunkedArray { chunks: [chunk_0, chunk_1] }, mask }
//! ```
//!
//! **Iteration 1**: The executor sees `FilterArray`. It calls `reduce` — no trivial rules
//! apply (the mask is not all-true or all-false). It then iterates over children. Child 0 is
//! a `ChunkedArray`. It calls `ChunkedVTable::reduce_parent` — no metadata-only rules match
//! a `FilterArray` parent. It then calls `ChunkedVTable::execute_parent` — the
//! `FilterExecuteAdaptor` matches! It splits the filter mask across chunk boundaries and
//! produces:
//!
//! ```text
//! ChunkedArray { chunks: [Filter(chunk_0, mask_0), Filter(chunk_1, mask_1)] }
//! ```
//!
//! **Iteration 2**: The executor sees `ChunkedArray` with 2 chunks. `reduce` finds nothing
//! to simplify. No child-driven rules fire. `execute` canonicalizes the chunks by recursively
//! executing each `FilterArray`, then concatenates the results into a single canonical array.
//!
//! # Concrete example: `Filter(Constant(42, len=1000))`
//!
//! ```text
//! FilterArray { child: ConstantArray(42, len=1000), mask (500 true values) }
//! ```
//!
//! **Iteration 1**: `reduce` finds no trivial simplification. Child 0 is a `ConstantArray`.
//! `ConstantVTable::reduce_parent` fires: `ConstantFilterRule` matches `FilterArray` as the
//! parent and returns `ConstantArray(42, len=500)` — no data was read at all.
//!
//! **Iteration 2**: `ConstantArray` is not canonical, so `execute` materializes it into a
//! `PrimitiveArray` of 500 copies of 42. (Or the outer `Columnar` loop recognizes it as a
//! terminal `Constant` and stops.)
//!
//! [`FilterArray`]: crate::arrays::FilterArray
//! [`SliceArray`]: crate::arrays::SliceArray
//! [`DictArray`]: crate::arrays::DictArray
//! [`ScalarFnArray`]: crate::arrays::ScalarFnArray
//! [`ChunkedArray`]: crate::arrays::ChunkedArray
//! [`ConstantArray`]: crate::arrays::ConstantArray
//! [`PrimitiveArray`]: crate::arrays::PrimitiveArray
//! [`BoolArray`]: crate::arrays::BoolArray
//! [`VarBinViewArray`]: crate::arrays::VarBinViewArray
//! [`ArrayReduceRule`]: crate::optimizer::rules::ArrayReduceRule
//! [`ArrayParentReduceRule<V>`]: crate::optimizer::rules::ArrayParentReduceRule
//! [`ReduceRuleSet`]: crate::optimizer::rules::ReduceRuleSet
//! [`ParentRuleSet`]: crate::optimizer::rules::ParentRuleSet
//! [`ParentKernelSet`]: crate::kernel::ParentKernelSet
//! [`ExecuteParentKernel<V>`]: crate::kernel::ExecuteParentKernel
//! [`VTable::reduce`]: crate::vtable::VTable::reduce
//! [`VTable::reduce_parent`]: crate::vtable::VTable::reduce_parent
//! [`VTable::execute_parent`]: crate::vtable::VTable::execute_parent
//! [`VTable::execute`]: crate::vtable::VTable::execute
//! [`Columnar::execute`]: crate::columnar::Columnar

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
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

impl dyn Array + '_ {
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
