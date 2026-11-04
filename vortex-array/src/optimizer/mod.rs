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
//!
//! Each array records whether [`ArrayOptimizer::optimize_recursive`] has optimized it and all of
//! its descendants. A new array starts unoptimized, and the optimizer marks the arrays that
//! [`ArrayOptimizer::optimize_recursive`] returns, so optimizing a marked array again returns it
//! without trying any rule. The mark does not record the session: an array optimized with one
//! session is not optimized again with a session that registers more kernels.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

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

/// Records whether the optimizer has already optimized an array node and all of its descendants.
///
/// A shared array never changes, so a node that the optimizer took to its fixpoint stays there.
/// Every new array starts unoptimized, and changing a uniquely owned array in place clears the
/// state.
///
/// # What the mark promises
///
/// The mark promises only that no reduce or parent-reduce rule applies to the node or its
/// descendants, given their encodings, dtypes, slots and data. These do not change while the
/// array is shared. A thread that sees the mark skips the optimizer for the whole subtree, so a
/// rule that could apply later is never tried.
///
/// # When an array stays unoptimized
///
/// An array must stay unmarked, or clear its mark, when the decision of a rule depends on
/// something that can change after the optimizer pass:
///
/// - The array changes in place. [`ArrayRef::inner_mut`] clears the mark, because the caller holds
///   the only reference and can replace slots or data.
/// - A rule reads state that can be set on a shared array after the pass, such as statistics.
///   For example, a rule that reduces an array to a constant when [`Stat::IsConstant`] is known
///   does not run if the statistic is computed after the array was marked. A shared array cannot
///   clear its mark, because that needs the only reference. Thus such a rule must not depend on
///   the mark, or the optimizer must not mark the arrays that it can apply to.
/// - The session changes. The mark does not record the kernels it was optimized with, so an
///   array optimized with one session is not optimized again with a session that registers more
///   kernels.
///
/// # Memory ordering
///
/// The flag uses [`Ordering::Relaxed`]. This is correct because the mark only promises things that
/// a thread can already see through its reference to the array. If the mark comes to promise
/// state that another thread writes, such as statistics that a rule reads, store it with
/// [`Ordering::Release`] and load it with [`Ordering::Acquire`]. Then a thread that sees the mark
/// also sees that state.
///
/// [`ArrayRef::inner_mut`]: crate::ArrayRef::inner_mut
/// [`Stat::IsConstant`]: crate::expr::stats::Stat::IsConstant
#[derive(Debug, Default)]
pub(crate) struct OptimizedState(pub(crate) AtomicBool);

impl OptimizedState {
    fn is_optimized(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    fn mark_optimized(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Marks an array decoded from its serialized form as optimized.
///
/// Decoding calls this for each node it decodes, and it decodes the children of a node before the
/// node. Thus every decoded node is marked, and the optimizer does not try reduce or
/// parent-reduce rules on any of them. Only the nodes that an encoding's deserializer builds itself
/// around its decoded children stay unmarked. Arrays are usually written as the compressor built
/// them, so this skips rules on decoded arrays without the guarantee that none applies.
pub(crate) fn mark_decoded_optimized(array: &ArrayRef) {
    array.optimized_state().mark_optimized();
}

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
        if self.optimized_state().is_optimized() {
            return Ok(self.clone());
        }

        Ok(optimize_owned(self.clone(), None)?.0)
    }

    fn optimize_ctx(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        if self.optimized_state().is_optimized() {
            return Ok(self.clone());
        }

        Ok(optimize_owned(self.clone(), Some(session))?.0)
    }

    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        Ok(try_optimize_recursive(self.clone(), session)?.0)
    }
}

fn optimize_owned(
    mut current_array: ArrayRef,
    session: Option<&VortexSession>,
) -> VortexResult<(ArrayRef, bool)> {
    let mut any_optimizations = false;
    let session_kernels = session.map(|session| session.kernels());

    trace_op!(record_optimize_start(&current_array, session.is_some()));

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

        return Ok((current_array, any_optimizations));
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
    current_array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<(ArrayRef, bool)> {
    if current_array.optimized_state().is_optimized() {
        return Ok((current_array, false));
    }

    trace_op!(record_optimize_recursive_start(&current_array));

    let (mut current_array, mut any_optimizations) = optimize_owned(current_array, Some(session))?;

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

    // Every child is marked now, so the mark on this node covers its whole subtree.
    current_array.optimized_state().mark_optimized();
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
    use crate::arrays::FilterArray;
    use crate::arrays::Primitive;
    use crate::arrays::SliceArray;
    use crate::optimizer::ArrayOptimizer;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn optimize_recursive_marks_result_and_skips_it_after() -> VortexResult<()> {
        let sliced = SliceArray::try_new(buffer![1i32, 2, 3, 4].into_array(), 1..3)?.into_array();
        assert!(!sliced.optimized_state().is_optimized());

        let optimized = sliced.optimize_recursive(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert!(optimized.optimized_state().is_optimized());
        assert!(ArrayRef::ptr_eq(
            &optimized.optimize_recursive(&SESSION)?,
            &optimized
        ));
        assert!(ArrayRef::ptr_eq(
            &optimized.optimize_ctx(&SESSION)?,
            &optimized
        ));

        // A new array built over an optimized one starts unoptimized.
        let resliced = SliceArray::try_new(optimized, 0..1)?.into_array();
        assert!(!resliced.optimized_state().is_optimized());
        Ok(())
    }

    #[test]
    fn optimize_recursive_marks_every_descendant() -> VortexResult<()> {
        let sliced = SliceArray::try_new(buffer![1i32, 2, 3, 4].into_array(), 1..4)?.into_array();
        let filtered =
            FilterArray::try_new(sliced, Mask::from_iter([true, false, true]))?.into_array();

        let optimized = filtered.optimize_recursive(&SESSION)?;
        assert!(
            optimized
                .depth_first_traversal()
                .all(|node| node.optimized_state().is_optimized())
        );
        Ok(())
    }
}
