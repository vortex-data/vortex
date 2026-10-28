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
//!
//! Each array records how far the optimizer has taken it in its optimized state. A new array
//! starts unoptimized, and the optimizer marks the arrays it returns, so optimizing an array again
//! returns it without trying any rule. The mark does not record the session: an array optimized
//! with one session is not optimized again with a session that registers more kernels.

use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayRef;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::kernels::ParentReduceKernels;
use crate::optimizer::kernels::reduce_parent_key;
use crate::optimizer::kernels::reduce_parent_marker_key;
use crate::trace_op;

pub mod kernels;
pub mod rules;

/// Last zero-based fixpoint pass attempted before treating continued rewrites as an infinite loop.
/// Increasing this value permits longer rewrite chains but delays detection of cyclic rules.
const MAX_OPTIMIZER_REWRITE_PASS: usize = 100;

/// Records how far the optimizer has already taken an array node.
///
/// A shared array never changes, so a node that reached the optimizer's fixpoint stays there.
/// Every new array starts unoptimized, and changing a uniquely owned array in place clears the
/// state.
#[derive(Debug, Default)]
pub(crate) struct OptimizedState(AtomicU8);

impl OptimizedState {
    /// No rule rewrites the node itself.
    const ROOT: u8 = 1;
    /// No rule rewrites the node or any of its descendants.
    const TREE: u8 = 1 << 1;

    pub(crate) fn is_root_optimized(&self) -> bool {
        self.0.load(Ordering::Relaxed) & Self::ROOT != 0
    }

    pub(crate) fn is_tree_optimized(&self) -> bool {
        self.0.load(Ordering::Relaxed) & Self::TREE != 0
    }

    fn mark_root_optimized(&self) {
        self.0.fetch_or(Self::ROOT, Ordering::Relaxed);
    }

    fn mark_tree_optimized(&self) {
        self.0.fetch_or(Self::ROOT | Self::TREE, Ordering::Relaxed);
    }

    pub(crate) fn clear(&mut self) {
        *self.0.get_mut() = 0;
    }
}

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
        if self.optimized_state().is_root_optimized() {
            return Ok(self.clone());
        }
        let kernels = session.kernels().reduce_parent_snapshot();
        optimize_with_kernels(self.clone(), &kernels)
    }

    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        if self.optimized_state().is_tree_optimized() {
            return Ok(self.clone());
        }
        let kernels = session.kernels().reduce_parent_snapshot();
        optimize_recursive_with_kernels(self.clone(), &kernels)
    }
}

/// [`ArrayOptimizer::optimize`] against a snapshot of the session's parent-reduce kernels, so a
/// caller that optimizes repeatedly, such as the executor, reads the session once.
pub(crate) fn optimize_with_kernels(
    array: ArrayRef,
    kernels: &ParentReduceKernels,
) -> VortexResult<ArrayRef> {
    Ok(optimize_owned(array, kernels)?.0)
}

/// [`ArrayOptimizer::optimize_recursive`] against a snapshot of the session's parent-reduce
/// kernels.
pub(crate) fn optimize_recursive_with_kernels(
    array: ArrayRef,
    kernels: &ParentReduceKernels,
) -> VortexResult<ArrayRef> {
    Ok(try_optimize_recursive(array, kernels)?.0)
}

fn optimize_owned(
    mut current_array: ArrayRef,
    kernels: &ParentReduceKernels,
) -> VortexResult<(ArrayRef, bool)> {
    if current_array.optimized_state().is_root_optimized() {
        return Ok((current_array, false));
    }

    let mut any_optimizations = false;

    trace_op!(record_optimize_start(&current_array));

    for _ in 0..=MAX_OPTIMIZER_REWRITE_PASS {
        trace_op!(record_optimize_loop_start(&current_array));

        if let Some(new_array) = current_array.reduce()? {
            current_array = optimize_new_nodes(new_array, &current_array, kernels)?;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_reduce_none(&current_array));

        // Try children in order; the first parent rewrite restarts the fixpoint loop.
        let mut reduced_parent = None;
        let parent_has_kernels =
            kernels.contains_key(&reduce_parent_marker_key(current_array.encoding_id()));
        for (slot_idx, slot) in current_array.slots().iter().enumerate() {
            let Some(child) = slot else {
                continue;
            };

            // Session kernels take precedence over the child's static parent-reduce rules.
            if parent_has_kernels
                && let Some(new_array) =
                    try_session_parent_reduce(kernels, &current_array, child, slot_idx)?
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
            current_array = optimize_new_nodes(new_array, &current_array, kernels)?;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_parent_reduce_none(&current_array));
        trace_op!(record_optimize_loop_end());

        trace_op!(record_optimize_done(&current_array, any_optimizations));

        current_array.optimized_state().mark_root_optimized();
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
    kernels: &ParentReduceKernels,
) -> VortexResult<ArrayRef> {
    let mut existing = HashSet::new();
    let mut stack = vec![old];
    while let Some(node) = stack.pop() {
        if existing.insert(node.addr()) {
            stack.extend(node.slots().iter().flatten());
        }
    }
    optimize_new_slots(array, &existing, kernels)
}

fn optimize_new_slots(
    array: ArrayRef,
    existing: &HashSet<usize>,
    kernels: &ParentReduceKernels,
) -> VortexResult<ArrayRef> {
    let mut new_slots = SmallVec::with_capacity(array.slots().len());
    let mut any_slot_optimized = false;
    for slot in array.slots() {
        match slot {
            Some(child) if !existing.contains(&child.addr()) => {
                let (optimized, _) = optimize_owned(child.clone(), kernels)?;
                let optimized = optimize_new_slots(optimized, existing, kernels)?;
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
    kernels: &ParentReduceKernels,
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    let Some(reduce_parent_fns) =
        kernels.get(&reduce_parent_key(parent.encoding_id(), child.encoding_id()))
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
    kernels: &ParentReduceKernels,
) -> VortexResult<(ArrayRef, bool)> {
    if current_array.optimized_state().is_tree_optimized() {
        return Ok((current_array, false));
    }

    trace_op!(record_optimize_recursive_start(&current_array));

    let (mut current_array, mut any_optimizations) = optimize_owned(current_array, kernels)?;

    let mut new_slots = SmallVec::with_capacity(current_array.slots().len());
    let mut any_slot_optimized = false;
    for slot in current_array.slots() {
        match slot {
            Some(child) => {
                let (new_child, new_slot_optimized) =
                    try_optimize_recursive(child.clone(), kernels)?;
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

    // New children may let a rule rewrite the parent, so only an unchanged tree is marked; the
    // next pass tries a rebuilt parent again.
    if any_slot_optimized {
        // SAFETY: optimizer rules only replace child slots with logically equivalent arrays, so
        // parent logical values and statistics remain valid.
        current_array = unsafe { current_array.with_slots(new_slots) }?;
        any_optimizations = true;
    } else {
        current_array.optimized_state().mark_tree_optimized();
    }

    Ok((current_array, any_optimizations))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::Primitive;
    use crate::optimizer::ArrayOptimizer;
    use crate::optimizer::MAX_OPTIMIZER_REWRITE_PASS;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn optimize_marks_result_and_skips_it_after() -> VortexResult<()> {
        let sliced = buffer![1i32, 2, 3, 4].into_array().slice(1..3)?;
        assert!(!sliced.is_optimized());

        let optimized = sliced.optimize(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert!(optimized.is_optimized());
        assert!(ArrayRef::ptr_eq(&optimized.optimize(&SESSION)?, &optimized));

        // A new array built over an optimized one starts unoptimized.
        assert!(!optimized.slice(0..1)?.is_optimized());
        Ok(())
    }

    #[test]
    fn optimize_recursive_marks_tree_once_nothing_changes() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3, 4].into_array();
        let filtered = array
            .slice(1..4)?
            .filter(Mask::from_iter([true, false, true]))?;
        assert!(!filtered.is_recursively_optimized());

        // A pass that rewrites a child leaves the parent for the next pass to check.
        let mut optimized = filtered.optimize_recursive(&SESSION)?;
        for _ in 0..MAX_OPTIMIZER_REWRITE_PASS {
            if optimized.is_recursively_optimized() {
                break;
            }
            optimized = optimized.optimize_recursive(&SESSION)?;
        }
        assert!(optimized.is_recursively_optimized());
        assert!(optimized.is_optimized());
        assert!(
            optimized
                .depth_first_traversal()
                .all(|node| node.is_recursively_optimized())
        );
        assert!(ArrayRef::ptr_eq(
            &optimized.optimize_recursive(&SESSION)?,
            &optimized
        ));
        Ok(())
    }
}
