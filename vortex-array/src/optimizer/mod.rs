// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The optimizer applies metadata-only rewrite rules (`reduce` and `reduce_parent`) in a
//! fixpoint loop until no more transformations are possible.
//!
//! Optimization runs between execution steps, which is what enables cross-step optimizations:
//! after a child is decoded, new `reduce_parent` rules may match that were previously blocked.
//!
//! There are two public entry points on [`ArrayOptimizer`], and both take the session whose
//! [`kernels::ArrayKernels`] registry is consulted before the static parent-reduce rules:
//!
//! - [`ArrayOptimizer::optimize`] optimizes the root array node.
//! - [`ArrayOptimizer::optimize_recursive`] optimizes the root and every descendant.
//!
//! Array constructors such as [`ArrayRef::slice`](crate::ArrayRef::slice) do not optimize: they
//! return the wrapped array. Execution optimizes the whole tree with its session before it starts
//! and the array it produced when it finishes. Whenever a rule rewrites a node, the nodes the rule
//! built are optimized too, as constructing them used to do.

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

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
    /// Session kernels are checked for each `(parent_encoding_id, child_encoding_id)` pair before
    /// the child's static `PARENT_RULES`. The registry comes from the [`kernels::KernelSession`] on
    /// `session`.
    fn optimize(&self, session: &VortexSession) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants).
    ///
    /// This uses the same session-aware rule ordering as [`Self::optimize`] for every node in
    /// the tree.
    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        Ok(optimize_owned(self.clone(), session)?.0)
    }

    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        Ok(try_optimize_recursive(self.clone(), session)?.0)
    }
}

fn optimize_owned(
    mut current_array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<(ArrayRef, bool)> {
    let mut any_optimizations = false;
    let session_kernels = session.kernels();

    trace_op!(record_optimize_start(&current_array));

    for _ in 0..=MAX_OPTIMIZER_REWRITE_PASS {
        trace_op!(record_optimize_loop_start(&current_array));

        if let Some(new_array) = current_array.reduce()? {
            current_array = optimize_new_nodes(new_array, &current_array, session)?;
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
            if let Some(new_array) =
                try_session_parent_reduce(&session_kernels, &current_array, child, slot_idx)?
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
            current_array = optimize_new_nodes(new_array, &current_array, session)?;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_parent_reduce_none(&current_array));
        trace_op!(record_optimize_loop_end());

        trace_op!(record_optimize_done(&current_array, any_optimizations));

        return Ok((current_array, any_optimizations));
    }

    vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
}

/// Optimizes the nodes a rule built below its result `array`, which replaces `old`.
///
/// Rules construct their result from lazy array constructors, so the nodes they build are not yet
/// optimized, at any depth. Optimizing them as each rewrite happens keeps the optimizer from
/// undoing one rule with its inverse: `slice(get_item(s))` and `get_item(slice(s))` rewrite into
/// each other until the inner `slice(s)` collapses into `s`'s own encoding. Subtrees the rule
/// carried over from `old` are left as they are.
fn optimize_new_nodes(
    array: ArrayRef,
    old: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let mut existing = HashSet::new();
    let mut stack = vec![old];
    while let Some(node) = stack.pop() {
        if existing.insert(node.addr()) {
            stack.extend(node.slots().iter().flatten());
        }
    }
    optimize_new_slots(array, &existing, session)
}

fn optimize_new_slots(
    array: ArrayRef,
    existing: &HashSet<usize>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let mut new_slots = SmallVec::with_capacity(array.slots().len());
    let mut any_slot_optimized = false;
    for slot in array.slots() {
        match slot {
            Some(child) if !existing.contains(&child.addr()) => {
                let (optimized, _) = optimize_owned(child.clone(), session)?;
                let optimized = optimize_new_slots(optimized, existing, session)?;
                any_slot_optimized |= !ArrayRef::ptr_eq(&optimized, child);
                new_slots.push(Some(optimized));
            }
            slot => new_slots.push(slot.clone()),
        }
    }
    if !any_slot_optimized {
        return Ok(array);
    }
    // SAFETY: optimizer rules only replace child slots with logically equivalent arrays, so the
    // parent's logical values and statistics remain valid.
    unsafe { array.with_slots(new_slots) }
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
    current_array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<(ArrayRef, bool)> {
    trace_op!(record_optimize_recursive_start(&current_array));

    let (mut current_array, mut any_optimizations) = optimize_owned(current_array, session)?;

    let mut new_slots = SmallVec::with_capacity(current_array.slots().len());
    let mut any_slot_optimized = false;
    for slot in current_array.slots() {
        match slot {
            Some(child) => {
                let (new_child, new_slot_optimized) =
                    try_optimize_recursive(child.clone(), session)?;
                if new_slot_optimized {
                    trace_op!(record_optimize_recursive_slot(
                        new_slots.len(),
                        child,
                        &new_child,
                    ));
                }
                new_slots.push(Some(new_child));
                any_slot_optimized |= new_slot_optimized;
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

    Ok((current_array, any_optimizations))
}
