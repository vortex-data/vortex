// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The optimizer applies metadata-only rewrite rules (`reduce` and `reduce_parent`) in a
//! fixpoint loop until no more transformations are possible.
//!
//! Optimization runs between execution steps, which is what enables cross-step optimizations:
//! after a child is decoded, new `reduce_parent` rules may match that were previously blocked.
//!
//! There are two entry points:
//!
//! * [`ArrayOptimizer::optimize`] — runs the static rules only (the child encoding's
//!   `PARENT_RULES`). It does not require an execution context and is used by helpers like
//!   `ArrayBuiltins::cast` and `ArrayRef::slice` that build wrapped expressions and need them
//!   normalized inline.
//! * [`ArrayOptimizer::optimize_ctx`] — runs the static rules and additionally consults the
//!   session's [`OptimizerSession`] registry keyed by `(parent_encoding_id, child_encoding_id)`
//!   before each `reduce_parent` step. The execute loop calls this entry point so plugin-
//!   registered parent-reduce rules fire during execution.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::SessionExt;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::optimizer::session::OptimizerSession;

pub mod rules;
pub mod session;

/// Pluggable parent-reduce function signature used by [`OptimizerSession`].
///
/// A function of this type rewrites the parent array that holds `child` at `child_idx`, given
/// the child itself and its parent. Returns `Ok(None)` when the function doesn't apply.
///
/// Registered under `(parent_encoding_id, child_encoding_id)`; callers downcast the erased
/// `child`/`parent` to their expected types before applying logic.
pub type ReduceParentFn =
    fn(child: &ArrayRef, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>>;

/// Extension trait for optimizing array trees using reduce/reduce_parent rules.
pub trait ArrayOptimizer {
    /// Optimize the root array node by running reduce and reduce_parent rules to fixpoint.
    ///
    /// Uses only the child encoding's static `PARENT_RULES`. Use [`Self::optimize_ctx`] from
    /// inside the execute loop to also consult the session-scoped [`OptimizerSession`] registry.
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Like [`Self::optimize`], but additionally consults the session's [`OptimizerSession`]
    /// registry for each `(parent_encoding_id, child_encoding_id)` pair before the static
    /// vtable rules.
    fn optimize_ctx(&self, ctx: &ExecutionCtx) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants), static rules only.
    fn optimize_recursive(&self) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        Ok(try_optimize(self, None)?.unwrap_or_else(|| self.clone()))
    }

    fn optimize_ctx(&self, ctx: &ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(try_optimize(self, Some(ctx))?.unwrap_or_else(|| self.clone()))
    }

    fn optimize_recursive(&self) -> VortexResult<ArrayRef> {
        Ok(try_optimize_recursive(self)?.unwrap_or_else(|| self.clone()))
    }
}

/// Resolve a pluggable [`ReduceParentFn`] for `(parent, child)` from the session registry.
///
/// Returns `None` when no [`OptimizerSession`] is registered, or no function is registered under
/// `(parent.encoding_id(), child.encoding_id())`. The returned `Arc` is owned so the caller is
/// free to drop the session borrow before invoking it.
fn plugin_reduce_parent(
    ctx: &ExecutionCtx,
    parent: &ArrayRef,
    child: &ArrayRef,
) -> Option<Arc<ReduceParentFn>> {
    ctx.session().get_opt::<OptimizerSession>().and_then(|s| {
        s.registry()
            .find::<ReduceParentFn>(parent.encoding_id(), child.encoding_id())
    })
}

fn try_optimize(array: &ArrayRef, ctx: Option<&ExecutionCtx>) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimizations = false;

    // Apply reduction rules to the current array until no more rules apply.
    let mut loop_counter = 0;
    'outer: loop {
        if loop_counter > 100 {
            vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
        }
        loop_counter += 1;

        if let Some(new_array) = current_array.reduce()? {
            current_array = new_array;
            any_optimizations = true;
            continue;
        }

        // Apply parent reduction rules to each slot in the context of the current array.
        // Its important to take all slots here, as `current_array` can change inside the loop.
        for (slot_idx, slot) in current_array.slots().iter().enumerate() {
            let Some(child) = slot else { continue };

            // Registry-based override: tried before the child encoding's static PARENT_RULES.
            if let Some(ctx) = ctx
                && let Some(plugin) = plugin_reduce_parent(ctx, &current_array, child)
                && let Some(new_array) = plugin(child, &current_array, slot_idx)?
            {
                current_array = new_array;
                any_optimizations = true;
                continue 'outer;
            }

            if let Some(new_array) = child.reduce_parent(&current_array, slot_idx)? {
                // If the parent was replaced, then we attempt to reduce it again.
                current_array = new_array;
                any_optimizations = true;

                // Continue to the start of the outer loop
                continue 'outer;
            }
        }

        // No more optimizations can be applied
        break;
    }

    if any_optimizations {
        Ok(Some(current_array))
    } else {
        Ok(None)
    }
}

fn try_optimize_recursive(array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimizations = false;

    if let Some(new_array) = try_optimize(&current_array, None)? {
        current_array = new_array;
        any_optimizations = true;
    }

    let mut new_slots = Vec::with_capacity(current_array.slots().len());
    let mut any_slot_optimized = false;
    for slot in current_array.slots() {
        match slot {
            Some(child) => {
                if let Some(new_child) = try_optimize_recursive(child)? {
                    new_slots.push(Some(new_child));
                    any_slot_optimized = true;
                } else {
                    new_slots.push(Some(child.clone()));
                }
            }
            None => new_slots.push(None),
        }
    }

    if any_slot_optimized {
        current_array = current_array.with_slots(new_slots)?;
        any_optimizations = true;
    }

    if any_optimizations {
        Ok(Some(current_array))
    } else {
        Ok(None)
    }
}
