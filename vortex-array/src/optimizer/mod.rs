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
//! - [`ArrayOptimizer::optimize_ctx`] also consults the session's active
//!   [`kernels::ArrayKernels`] registry before static parent-reduce rules, so this is the entry
//!   point used by execution.
//! - [`ArrayOptimizer::optimize_recursive`] applies the session-aware optimizer to the root and
//!   every descendant.

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::optimizer::kernels::ArrayKernels;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::trace_op;

pub mod kernels;
pub mod rules;

/// Last zero-based fixpoint pass attempted before treating continued rewrites as an infinite loop.
/// Increasing this value permits longer rewrite chains but delays detection of cyclic rules.
const MAX_OPTIMIZER_REWRITE_PASS: usize = 100;

/// Extension trait for optimizing array trees using reduce/reduce_parent rules.
pub trait ArrayOptimizer {
    /// Optimize the root array node by running reduce and reduce_parent rules to fixpoint.
    ///
    /// This uses only static rules registered on encoding vtables. Use [`Self::optimize_ctx`]
    /// when a session-scoped [`kernels::ArrayKernels`] registry should participate.
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Optimize the root array node using static rules and the active
    /// [`kernels::ArrayKernels`] registry on `session`, if any.
    ///
    /// Session kernels are checked for each `(parent_encoding_id, child_encoding_id)` pair before
    /// the child's static `PARENT_RULES`. The registry comes from the [`kernels::KernelSession`] on
    /// `session`, if any. If `session` does not contain a [`kernels::KernelSession`], this behaves
    /// like [`Self::optimize`].
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
    let session_kernels = session.map(|session| session.kernels());

    trace_op!(record_optimize_start(array, session.is_some()));

    for _ in 0..=MAX_OPTIMIZER_REWRITE_PASS {
        trace_op!(record_optimize_loop_start(&current_array));

        if let Some(new_array) = current_array.reduce()? {
            current_array = new_array;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_reduce_none(&current_array));

        // Try children in order; the first parent rewrite restarts the fixpoint loop.
        let mut reduced_parent = None;
        for (slot_idx, slot) in current_array.slots().iter().enumerate() {
            let Some(child) = slot else {
                continue;
            };

            // Session kernels take precedence over the child's static parent-reduce rules.
            if let Some(session_kernels) = &session_kernels
                && let Some(new_array) =
                    try_session_parent_reduce(session_kernels, &current_array, child, slot_idx)?
            {
                reduced_parent = Some(new_array);
                break;
            }

            if let Some(new_array) = child.reduce_parent(&current_array, slot_idx)? {
                reduced_parent = Some(new_array);
                break;
            }
        }

        if let Some(new_array) = reduced_parent {
            current_array = new_array;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_parent_reduce_none(&current_array));
        trace_op!(record_optimize_loop_end());

        trace_op!(record_optimize_done(&current_array, any_optimizations));

        return Ok(any_optimizations.then_some(current_array));
    }

    vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
}

fn try_session_parent_reduce(
    kernels: &ArrayKernels,
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    let Some(reduce_parent_fns) =
        kernels.find_reduce_parent(parent.encoding_id(), child.encoding_id())
    else {
        return Ok(None);
    };

    #[allow(clippy::unused_enumerate_index)]
    for (_kernel_idx, reduce_parent) in reduce_parent_fns.iter().enumerate() {
        if let Some(new_array) = reduce_parent(child, parent, slot_idx)? {
            trace_op!(record_session_parent_reduce_applied(
                parent,
                child,
                slot_idx,
                _kernel_idx,
                &new_array,
            ));

            return Ok(Some(new_array));
        }

        trace_op!(record_session_parent_reduce_declined(
            parent,
            child,
            slot_idx,
            _kernel_idx,
        ));
    }

    Ok(None)
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
        // SAFETY: optimizer rules only replace child slots with logically equivalent arrays, so
        // parent logical values and statistics remain valid.
        current_array = unsafe { current_array.with_slots(new_slots) }?;
        any_optimizations = true;
    }

    if any_optimizations {
        Ok(Some(current_array))
    } else {
        Ok(None)
    }
}
