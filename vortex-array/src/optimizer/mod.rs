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
//!   `PARENT_RULES`). It does not require a [`VortexSession`] and is used by helpers like
//!   `ArrayBuiltins::cast` and `ArrayRef::slice` that build wrapped expressions and need them
//!   normalized inline.
//! * [`ArrayOptimizer::optimize_ctx`] — runs the static rules and additionally consults the
//!   session's [`ArrayKernels`] registry keyed by `(parent_encoding_id, child_encoding_id)`
//!   before each `reduce_parent` step. The execute loop calls this entry point so plugin-
//!   registered parent-reduce rules fire during execution.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::optimizer::kernels::ArrayKernels;
use crate::optimizer::kernels::ReduceParentFn;

pub mod kernels;
pub mod rules;

/// Extension trait for optimizing array trees using reduce/reduce_parent rules.
pub trait ArrayOptimizer {
    /// Optimize the root array node by running reduce and reduce_parent rules to fixpoint.
    ///
    /// Uses only the child encoding's static `PARENT_RULES`. Use [`Self::optimize_ctx`] from
    /// inside the execute loop to also consult the session-scoped [`ArrayKernels`] registry.
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Like [`Self::optimize`], but additionally consults the [`ArrayKernels`] registered on
    /// `session` for each `(parent_encoding_id, child_encoding_id)` pair before the static
    /// vtable rules. If `session` does not have an [`ArrayKernels`] registered, falls
    /// through to the static rules.
    fn optimize_ctx(&self, session: &VortexSession) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants).
    ///
    /// Consults the [`ArrayKernels`] registered on `session` for each parent/child pair
    /// encountered during the recursive walk, so plugin-registered rules apply throughout the
    /// tree. Requires a [`VortexSession`] unconditionally so the registry is always honored
    /// when a recursive optimization is requested.
    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        Ok(try_optimize(self, None)?.unwrap_or_else(|| self.clone()))
    }

    fn optimize_ctx(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        Ok(try_optimize(self, Some(session))?.unwrap_or_else(|| self.clone()))
    }

    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        Ok(try_optimize_recursive(self, session)?.unwrap_or_else(|| self.clone()))
    }
}

/// Resolve a pluggable [`ReduceParentFn`] for `(parent, child)` from `session`.
///
/// Returns `None` when no [`ArrayKernels`] is registered, or no function is registered under
/// `(parent.encoding_id(), child.encoding_id())`. The returned `Arc` is owned so the caller can
/// drop the session borrow before invoking it.
fn plugin_reduce_parent(
    session: &VortexSession,
    parent: &ArrayRef,
    child: &ArrayRef,
) -> Option<Arc<ReduceParentFn>> {
    session
        .get_opt::<ArrayKernels>()
        .and_then(|s| s.find_reduce_parent(parent.encoding_id(), child.encoding_id()))
}

fn try_optimize(
    array: &ArrayRef,
    session: Option<&VortexSession>,
) -> VortexResult<Option<ArrayRef>> {
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
            if let Some(session) = session
                && let Some(plugin) = plugin_reduce_parent(session, &current_array, child)
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

fn try_optimize_recursive(
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimizations = false;

    if let Some(new_array) = try_optimize(&current_array, Some(session))? {
        current_array = new_array;
        any_optimizations = true;
    }

    let mut new_slots = Vec::with_capacity(current_array.slots().len());
    let mut any_slot_optimized = false;
    for slot in current_array.slots() {
        match slot {
            Some(child) => {
                if let Some(new_child) = try_optimize_recursive(child, session)? {
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
