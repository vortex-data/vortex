// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The optimizer applies metadata-only rewrite rules (`reduce` and `reduce_parent`) in a
//! fixpoint loop until no more transformations are possible.
//!
//! Optimization runs between execution steps, which is what enables cross-step optimizations:
//! after a child is decoded, new `reduce_parent` rules may match that were previously blocked.
//!
//! There are three public entry points on [`ArrayOptimizer`]:
//!
//! - [`ArrayOptimizer::optimize`] uses only static rules registered on encoding vtables.
//! - [`ArrayOptimizer::optimize_ctx`] also consults session-scoped [`ArrayKernels`] before
//!   static parent-reduce rules, so this is the entry point used by execution.
//! - [`ArrayOptimizer::optimize_recursive`] applies the session-aware optimizer to the root and
//!   every descendant.

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::optimizer::kernels::ArrayKernels;
use crate::trace_op;

pub mod kernels;
pub mod rules;

/// Extension trait for optimizing array trees using reduce/reduce_parent rules.
pub trait ArrayOptimizer {
    /// Optimize the root array node by running reduce and reduce_parent rules to fixpoint.
    ///
    /// This uses only static rules registered on encoding vtables. Use [`Self::optimize_ctx`]
    /// when session-registered [`ArrayKernels`] should participate.
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Optimize the root array node using static rules and any [`ArrayKernels`] on `session`.
    ///
    /// Session kernels are checked for each `(parent_encoding_id, child_encoding_id)` pair before
    /// the child's static `PARENT_RULES`. If `session` does not contain [`ArrayKernels`], this
    /// behaves like [`Self::optimize`].
    fn optimize_ctx(&self, session: &VortexSession) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants).
    ///
    /// This uses the same session-aware rule ordering as [`Self::optimize_ctx`] for every node in
    /// the tree.
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

fn try_optimize(
    array: &ArrayRef,
    session: Option<&VortexSession>,
) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimizations = false;
    let array_ref = session.and_then(|s| s.get_opt::<ArrayKernels>());

    trace_op!(record_optimize_start(array, session.is_some()));

    // Apply reduction rules to the current array until no more rules apply.
    for _ in 0..=100 {
        trace_op!(record_optimize_loop_start(&current_array));

        if let Some(new_array) = current_array.reduce()? {
            current_array = new_array;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_reduce_none(&current_array));

        // Apply parent reduction rules to each slot in the context of the current array.
        // Its important to take all slots here, as `current_array` can change inside the loop.
        let mut parent_reduced = None;
        for (slot_idx, slot) in current_array.slots().iter().enumerate() {
            let Some(child) = slot else { continue };

            // Session kernels take precedence over the child encoding's static PARENT_RULES.
            if let Some(array_ref) = &array_ref
                && let Some(plugins) =
                    array_ref.find_reduce_parent(current_array.encoding_id(), child.encoding_id())
            {
                #[allow(clippy::unused_enumerate_index)]
                for (_plugin_idx, plugin) in plugins.as_ref().iter().enumerate() {
                    if let Some(new_array) = plugin(child, &current_array, slot_idx)? {
                        trace_op!(record_session_parent_reduce_applied(
                            &current_array,
                            child,
                            slot_idx,
                            _plugin_idx,
                            &new_array,
                        ));
                        parent_reduced = Some(new_array);
                        break;
                    }
                    trace_op!(record_session_parent_reduce_declined(
                        &current_array,
                        child,
                        slot_idx,
                        _plugin_idx,
                    ));
                }
                if parent_reduced.is_some() {
                    break;
                }
            }

            if let Some(new_array) = child.reduce_parent(&current_array, slot_idx)? {
                parent_reduced = Some(new_array);
                break;
            }
        }

        if let Some(new_array) = parent_reduced {
            // If the parent was replaced, then we attempt to reduce it again.
            current_array = new_array;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_parent_reduce_none(&current_array));
        trace_op!(record_optimize_loop_end());

        // No more optimizations can be applied
        trace_op!(record_optimize_done(&current_array, any_optimizations));

        if any_optimizations {
            return Ok(Some(current_array));
        } else {
            return Ok(None);
        }
    }

    vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
}

fn try_optimize_recursive(
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimizations = false;

    trace_op!(record_optimize_recursive_start(array));

    if let Some(new_array) = try_optimize(&current_array, Some(session))? {
        current_array = new_array;
        any_optimizations = true;
    }

    let mut new_slots = SmallVec::with_capacity(current_array.slots().len());
    let mut any_slot_optimized = false;
    for slot in current_array.slots() {
        match slot {
            Some(child) => {
                if let Some(new_child) = try_optimize_recursive(child, session)? {
                    trace_op!(record_optimize_recursive_slot(
                        new_slots.len(),
                        child,
                        &new_child,
                    ));
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
