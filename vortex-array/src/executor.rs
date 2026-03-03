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
//! ---
//!
//! # How it works
//!
//! ## Execution loop
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
//!    any buffers. For example, a [`FilterArray`] with an all-true mask reduces to its child.
//!
//! 3. **`reduce_parent`** — for each child of the array, ask the *child's* VTable whether it
//!    can rewrite the parent using metadata alone. For example, when a [`ConstantArray`] is the
//!    child of a [`FilterArray`], filtering a constant is trivial — just produce a new constant
//!    with the output length.
//!
//! 4. **`execute_parent`** — like `reduce_parent`, but the child is allowed to read buffers
//!    and perform real work. For example, a [`ChunkedArray`] inside a [`FilterArray`] can split
//!    the filter mask across its chunks and filter each one independently.
//!
//! 5. **`execute`** — the fallback. The array executes itself one step closer to canonical
//!    form. For a [`FilterArray`], this means executing its child to canonical and applying
//!    the filter directly.
//!
//! Phases 2–4 return `Option<ArrayRef>` — `None` means "I can't help, try the next phase."
//! The first `Some` result wins and becomes the input for the next iteration of the loop.
//!
//! ## The optimizer (reduce-only evaluation)
//!
//! The optimizer ([`ArrayOptimizer::optimize`]) runs `reduce` and `reduce_parent` in a loop
//! but **never calls `execute` or `execute_parent`**. It applies metadata-only rewrites until
//! no more rules fire, without reading any buffers.
//!
//! [`optimize_recursive`] extends this to the entire array tree: optimize the root node, then
//! recurse into children.
//!
//! The optimizer is called during rule rewrites themselves. For example, when a chunked
//! push-down rule builds a new [`ScalarFnArray`] per chunk, it calls `.optimize()` on each
//! one so that further reduce rules (like constant folding) fire immediately. It is also
//! used at expression-building time (e.g., `array.apply(&expr)`) to simplify the tree before
//! execution begins.
//!
//! ## Walkthrough: `Filter(Chunked([chunk_0, chunk_1]))`
//!
//! ```text
//! FilterArray { child: ChunkedArray { chunks: [chunk_0, chunk_1] }, mask }
//! ```
//!
//! **Iteration 1**: The executor sees `FilterArray`. `reduce` — the mask is not trivial, no
//! rules fire. Child 0 is a `ChunkedArray`. `reduce_parent` — no metadata-only rules match
//! `FilterArray`. `execute_parent` — `FilterExecuteAdaptor` matches, splits the mask across
//! chunk boundaries:
//!
//! ```text
//! ChunkedArray { chunks: [Filter(chunk_0, mask_0), Filter(chunk_1, mask_1)] }
//! ```
//!
//! **Iteration 2**: `ChunkedArray` with 2 chunks. No rules simplify it. `execute`
//! canonicalizes each chunk (recursively executing each `FilterArray`) and concatenates.
//!
//! ## Walkthrough: `Filter(Constant(42, len=1000))`
//!
//! ```text
//! FilterArray { child: ConstantArray(42, len=1000), mask (500 true values) }
//! ```
//!
//! **Iteration 1**: `reduce` — no trivial simplification. Child 0 is `ConstantArray`.
//! `reduce_parent` fires: `ConstantFilterRule` matches `FilterArray` as parent and returns
//! `ConstantArray(42, len=500)`. No data read.
//!
//! **Iteration 2**: The outer `Columnar` loop recognizes `ConstantArray` as a terminal form
//! and stops. (If full canonicalization is requested, `execute` materializes it into a
//! `PrimitiveArray`.)
//!
//! ---
//!
//! # Implementing an encoding
//!
//! ## Where to put the code
//!
//! There are two places to put kernel and rule implementations, and they serve complementary
//! roles.
//!
//! **In the operation array** — the array that represents the operation (e.g., `FilterArray`,
//! `SliceArray`, `DictArray`). This is where you define:
//!
//! - **Traits** that encodings implement. For example, `arrays/filter/kernel.rs` defines
//!   [`FilterReduce`] (metadata-only) and [`FilterKernel`] (with buffer access).
//! - **Adaptors** that bridge those traits into `ArrayParentReduceRule` / `ExecuteParentKernel`.
//!   For example, [`FilterReduceAdaptor`] and [`FilterExecuteAdaptor`].
//! - **Self-reduce rules** about the operation itself, in `arrays/filter/rules.rs` (e.g.,
//!   `FilterFilterRule` for collapsing nested filters, `TrivialFilterRule` for all-true masks).
//! - **The fallback `execute`** implementation that handles each canonical type. For
//!   `FilterArray`, this lives in `arrays/filter/execute/` with one file per canonical type
//!   (`filter_primitive`, `filter_bool`, `filter_varbinview`, etc.).
//!
//! **In the encoding** — the encoding that wants to participate in the operation. This is where
//! you assemble `PARENT_RULES` and `PARENT_KERNELS` by registering adaptors. For example:
//!
//! `arrays/constant/compute/rules.rs`:
//! ```text
//! const PARENT_RULES: ParentRuleSet<ConstantVTable> = ParentRuleSet::new(&[
//!     ParentRuleSet::lift(&ConstantFilterRule),       // custom rule
//!     ParentRuleSet::lift(&FilterReduceAdaptor(ConstantVTable)),  // generic adaptor
//!     ParentRuleSet::lift(&SliceReduceAdaptor(ConstantVTable)),
//!     ParentRuleSet::lift(&TakeReduceAdaptor(ConstantVTable)),
//!     ...
//! ]);
//! ```
//!
//! `arrays/chunked/compute/kernel.rs`:
//! ```text
//! static PARENT_KERNELS: ParentKernelSet<ChunkedVTable> = ParentKernelSet::new(&[
//!     ParentKernelSet::lift(&FilterExecuteAdaptor(ChunkedVTable)),
//!     ParentKernelSet::lift(&SliceExecuteAdaptor(ChunkedVTable)),
//!     ParentKernelSet::lift(&TakeExecuteAdaptor(ChunkedVTable)),
//!     ...
//! ]);
//! ```
//!
//! For [`ScalarFnArray`], the scalar function itself provides the execute implementation via
//! `ScalarFnVTable::execute`, while reduce rules live on the `ScalarFnArray` side (e.g.,
//! constant folding when all children are `ConstantArray` in `arrays/scalar_fn/rules.rs`).
//!
//! // TODO: not all operations and encodings follow this pattern consistently yet. The goal is
//! // for every operation array to define Reduce and Kernel traits, and for every encoding to
//! // register adaptors in its own rule/kernel sets. Some older code still inlines logic that
//! // should be factored into this structure.
//!
//! ## `reduce` — self-simplification rules
//!
//! Implement [`ArrayReduceRule`] to define metadata-only rewrites for your encoding. Register
//! them as a [`ReduceRuleSet`] and call `RULES.evaluate(array)` from [`VTable::reduce`].
//!
//! Examples:
//! - [`ChunkedArray`] with 1 chunk → unwrap to the single chunk
//! - [`FilterArray`] with `AllTrue` mask → return the child unchanged
//! - [`ScalarFnArray`] where all children are constants → fold to a single constant
//!
//! ## `reduce_parent` — rewriting the parent without buffers
//!
//! Implement [`ArrayParentReduceRule<V>`] where `V` is your encoding's VTable. Each rule
//! specifies a `type Parent: Matcher` that constrains which parent type the rule applies to.
//! The `Matcher` trait (via `try_match`) gives you a strongly-typed view of the parent.
//!
//! Register them in a [`ParentRuleSet`] and call `PARENT_RULES.evaluate(...)` from
//! [`VTable::reduce_parent`].
//!
//! Examples:
//! - [`ConstantArray`] child of [`FilterArray`] → resize the constant
//! - [`ChunkedArray`] child of [`ScalarFnArray`] → push the scalar function into each chunk
//!
//! ## `execute_parent` — optimized execution with buffer access
//!
//! Implement [`ExecuteParentKernel<V>`] for kernels that need to read data. Like reduce rules,
//! each kernel specifies a `type Parent: Matcher`. Register them in a [`ParentKernelSet`] and
//! call `PARENT_KERNELS.execute(...)` from [`VTable::execute_parent`].
//!
//! Examples:
//! - [`ChunkedArray`] child of [`FilterArray`] → split the mask by chunk boundaries
//! - [`ChunkedArray`] child of `SliceArray` → slice the overlapping chunks
//!
//! ## `execute` — the fallback
//!
//! Implement [`VTable::execute`] as the last resort. This should make progress toward canonical
//! form — for example, by decoding the encoding into a simpler representation. The executor
//! will call this only when no reduce rule or parent kernel applied.
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
//! [`FilterReduce`]: crate::arrays::FilterReduce
//! [`FilterKernel`]: crate::arrays::FilterKernel
//! [`FilterReduceAdaptor`]: crate::arrays::FilterReduceAdaptor
//! [`FilterExecuteAdaptor`]: crate::arrays::FilterExecuteAdaptor
//! [`ArrayOptimizer::optimize`]: crate::optimizer::ArrayOptimizer::optimize
//! [`optimize_recursive`]: crate::optimizer::ArrayOptimizer::optimize_recursive
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
