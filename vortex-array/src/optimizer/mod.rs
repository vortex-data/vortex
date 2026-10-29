// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The optimizer applies metadata-only rewrite rules (`reduce` and `reduce_parent`) in a
//! fixpoint loop until no more transformations are possible.
//!
//! Optimization runs between execution steps, which is what enables cross-step optimizations:
//! after a child is decoded, new `reduce_parent` rules may match that were previously blocked.
//!
//! Parent-reduce rules live in the session's [`kernels::ArrayKernels`] registry, keyed by the
//! parent and child encoding ids, as execute-parent kernels are. There are two public entry
//! points on [`ArrayOptimizer`], and both take the session, or an [`ExecutionCtx`] created from it,
//! which reuses the kernels it snapshotted from the session (see [`AsOptimizerKernels`]):
//!
//! - [`ArrayOptimizer::optimize`] optimizes the root array node, after the descendants below it
//!   that were never optimized.
//! - [`ArrayOptimizer::optimize_recursive`] does the same. Optimizing a node first optimizes every
//!   descendant that was never optimized, so the root's mark covers the whole tree.
//!
//! Array constructors such as [`ArrayRef::slice`](crate::ArrayRef::slice) do not optimize: they
//! return the wrapped array. Optimizing a node therefore first optimizes its children that were
//! never optimized, bottom up, as constructing them used to: a parent rule or kernel keyed on a
//! child's encoding sees that encoding rather than a lazy wrapper around it, such as a slice of an
//! FSST array under a `LIKE`. Children already optimized are left without visiting their subtrees.
//! Whenever a rule rewrites a node, the nodes the rule built are optimized the same way.
//!
//! Each array records whether the optimizer has taken it to its fixpoint. A new array starts
//! unoptimized, and the optimizer marks the arrays it returns, so optimizing an array again returns
//! it without trying any rule. The optimizer marks a node only after it optimized the children of
//! the node, so the descendants of a marked node are marked too. The mark does not record the
//! session: an array optimized with one session is not optimized again with a session that
//! registers more kernels.

use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::array::ArrayId;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::executor::ExecutionCtx;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::kernels::ParentReduceKernels;
use crate::trace_op;

pub mod kernels;
pub mod rules;

/// Last zero-based fixpoint pass attempted before treating continued rewrites as an infinite loop.
/// Increasing this value permits longer rewrite chains but delays detection of cyclic rules.
const MAX_OPTIMIZER_REWRITE_PASS: usize = 100;

/// Records whether the optimizer has already taken an array node to its fixpoint.
///
/// A shared array never changes, so a node that reached the optimizer's fixpoint stays there.
/// Every new array starts unoptimized, and changing a uniquely owned array in place clears the
/// state.
#[derive(Debug, Default)]
pub(crate) struct OptimizedState(AtomicBool);

impl OptimizedState {
    pub(crate) fn is_optimized(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    fn mark_optimized(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub(crate) fn clear(&mut self) {
        *self.0.get_mut() = false;
    }
}

/// Marks the root of an array decoded from its serialized form as optimized.
///
/// The executor does not try reduce or parent-reduce rules on the decoded root itself. The
/// children of the root stay unmarked, so a rule that exposes one of them, such as a slice pushed
/// into a dictionary, still optimizes it. Arrays are usually written as the compressor built them,
/// so this skips rules on the root without the guarantee that none applies.
pub(crate) fn mark_decoded_optimized(array: &ArrayRef) {
    array.optimized_state().mark_optimized();
}

/// The parent-reduce kernels [`ArrayOptimizer`] optimizes with, from [`AsOptimizerKernels`].
///
/// A [`VortexSession`] snapshots its registered kernels for each call. An [`ExecutionCtx`] reuses
/// the snapshot it took from its session when it was created, so a kernel that optimizes what it
/// builds, possibly once per element, passes its context.
#[derive(Clone, Copy)]
pub struct OptimizerKernels<'a>(KernelSource<'a>);

#[derive(Clone, Copy)]
enum KernelSource<'a> {
    Session(&'a VortexSession),
    Snapshot(&'a ParentReduceKernels),
}

impl OptimizerKernels<'_> {
    fn with_reduce_parent_kernels<R>(self, f: impl FnOnce(&ParentReduceKernels) -> R) -> R {
        match self.0 {
            KernelSource::Session(session) => f(&session.kernels().reduce_parent_snapshot()),
            KernelSource::Snapshot(kernels) => f(kernels),
        }
    }
}

/// A source of the parent-reduce kernels [`ArrayOptimizer`] optimizes with.
pub trait AsOptimizerKernels {
    /// The kernels to optimize with.
    fn optimizer_kernels(&self) -> OptimizerKernels<'_>;
}

impl AsOptimizerKernels for VortexSession {
    fn optimizer_kernels(&self) -> OptimizerKernels<'_> {
        OptimizerKernels(KernelSource::Session(self))
    }
}

impl AsOptimizerKernels for LazyLock<VortexSession> {
    fn optimizer_kernels(&self) -> OptimizerKernels<'_> {
        OptimizerKernels(KernelSource::Session(self))
    }
}

impl AsOptimizerKernels for ExecutionCtx {
    fn optimizer_kernels(&self) -> OptimizerKernels<'_> {
        OptimizerKernels(KernelSource::Snapshot(self.reduce_parent_kernels()))
    }
}

/// Extension trait for optimizing array trees using reduce/reduce_parent rules.
pub trait ArrayOptimizer {
    /// Optimize the root array node by running reduce and reduce_parent rules to fixpoint.
    ///
    /// Parent-reduce kernels are looked up for each `(parent_encoding_id, child_encoding_id)` pair
    /// in the [`kernels::KernelSession`] of the session, or of the session an execution context
    /// was created from.
    fn optimize(&self, kernels: &(impl AsOptimizerKernels + ?Sized)) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants).
    ///
    /// This is the same as [`Self::optimize`]: optimizing a node first optimizes every descendant
    /// that was never optimized.
    fn optimize_recursive(
        &self,
        kernels: &(impl AsOptimizerKernels + ?Sized),
    ) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self, kernels: &(impl AsOptimizerKernels + ?Sized)) -> VortexResult<ArrayRef> {
        if self.optimized_state().is_optimized() {
            return Ok(self.clone());
        }
        kernels
            .optimizer_kernels()
            .with_reduce_parent_kernels(|kernels| optimize_with_kernels(self.clone(), kernels))
    }

    fn optimize_recursive(
        &self,
        kernels: &(impl AsOptimizerKernels + ?Sized),
    ) -> VortexResult<ArrayRef> {
        self.optimize(kernels)
    }
}

/// [`ArrayOptimizer::optimize`] against a snapshot of the session's parent-reduce kernels, so a
/// caller that optimizes repeatedly, such as the executor, reads the session once.
pub(crate) fn optimize_with_kernels(
    array: ArrayRef,
    kernels: &ParentReduceKernels,
) -> VortexResult<ArrayRef> {
    if array.optimized_state().is_optimized() {
        return Ok(array);
    }

    // Parent rules match on their children's encodings, so optimize the children first.
    let mut current_array = optimize_new_nodes(array.clone(), kernels)?;
    let mut any_optimizations = !ArrayRef::ptr_eq(&current_array, &array);

    trace_op!(record_optimize_start(&current_array));

    for _ in 0..=MAX_OPTIMIZER_REWRITE_PASS {
        trace_op!(record_optimize_loop_start(&current_array));

        if let Some(new_array) = current_array.reduce()? {
            current_array = optimize_new_nodes(new_array, kernels)?;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_reduce_none(&current_array));

        // Try children in order; the first parent rewrite restarts the fixpoint loop. A parent
        // that no kernel is registered for skips its children without probing each pair.
        let mut reduced_parent = None;
        let candidates = ReduceParentCandidates::new(kernels, &current_array);
        if !candidates.is_empty() {
            for (slot_idx, slot) in current_array.slots().iter().enumerate() {
                let Some(child) = slot else {
                    continue;
                };
                if candidates.contains(child)
                    && let Some(new_array) =
                        reduce_parent_with_kernels(kernels, &current_array, child, slot_idx)?
                {
                    reduced_parent = Some(new_array);
                    break;
                }
            }
        }

        if let Some(new_array) = reduced_parent {
            current_array = optimize_new_nodes(new_array, kernels)?;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
            continue;
        }

        trace_op!(record_optimize_parent_reduce_none(&current_array));
        trace_op!(record_optimize_loop_end());

        trace_op!(record_optimize_done(&current_array, any_optimizations));

        // A rewrite keeps the logical values, so every statistic of the input stays valid. Lazy
        // constructors such as `slice` put the statistics they carry over on the lazy node.
        if any_optimizations {
            current_array
                .statistics()
                .inherit_from(array.statistics());
        }

        current_array.optimized_state().mark_optimized();
        return Ok(current_array);
    }

    vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
}

/// Optimizes the nodes below `array` that were never optimized, such as those a rule built below
/// its result.
///
/// Rules and constructors build lazy arrays, so these nodes are not yet optimized, at any depth.
/// Optimizing them as each rewrite happens keeps the optimizer from undoing one rule with its
/// inverse: `slice(get_item(s))` and `get_item(slice(s))` rewrite into each other until the inner
/// `slice(s)` collapses into `s`'s own encoding.
///
/// A child that is already optimized is left as it is without visiting its subtree, which skips
/// the subtrees a rule carries over from the array it rewrote.
fn optimize_new_nodes(array: ArrayRef, kernels: &ParentReduceKernels) -> VortexResult<ArrayRef> {
    let mut new_slots: Option<SmallVec<_>> = None;
    for (slot_idx, slot) in array.slots().iter().enumerate() {
        let optimized = match slot {
            Some(child) if !child.optimized_state().is_optimized() => {
                let optimized = optimize_with_kernels(child.clone(), kernels)?;
                (!ArrayRef::ptr_eq(&optimized, child)).then_some(optimized)
            }
            _ => None,
        };
        match optimized {
            Some(optimized) => new_slots
                .get_or_insert_with(|| {
                    let mut slots = SmallVec::with_capacity(array.slots().len());
                    slots.extend(array.slots()[..slot_idx].iter().cloned());
                    slots
                })
                .push(Some(optimized)),
            None => {
                if let Some(new_slots) = &mut new_slots {
                    new_slots.push(slot.clone());
                }
            }
        }
    }
    let Some(new_slots) = new_slots else {
        return Ok(array);
    };
    // SAFETY: optimizer rules only replace child slots with logically equivalent arrays, so the
    // parent's logical values and statistics remain valid.
    unsafe { array.with_slots(new_slots) }
}

/// The child ids that parent-reduce kernels are registered for under one parent: those for the
/// parent's encoding id and, for a scalar function array, those for [`AnyScalarFn::id`].
struct ReduceParentCandidates<'a> {
    exact: &'a [ArrayId],
    any_scalar_fn: &'a [ArrayId],
    /// Whether either list holds [`AnyScalarFn::id`], so any scalar function child is a candidate.
    any_scalar_fn_child: bool,
}

impl<'a> ReduceParentCandidates<'a> {
    fn new(kernels: &'a ParentReduceKernels, parent: &ArrayRef) -> Self {
        let exact = kernels.children(parent.encoding_id()).unwrap_or_default();
        let any_scalar_fn = if parent.is::<ScalarFn>() {
            kernels.children(AnyScalarFn.id()).unwrap_or_default()
        } else {
            &[]
        };
        let any_scalar_fn_child = (!exact.is_empty() || !any_scalar_fn.is_empty()) && {
            let any_id = AnyScalarFn.id();
            exact.contains(&any_id) || any_scalar_fn.contains(&any_id)
        };
        Self {
            exact,
            any_scalar_fn,
            any_scalar_fn_child,
        }
    }

    fn is_empty(&self) -> bool {
        self.exact.is_empty() && self.any_scalar_fn.is_empty()
    }

    /// Returns whether any kernel may apply to `child` under this parent.
    fn contains(&self, child: &ArrayRef) -> bool {
        let id = child.encoding_id();
        self.exact.contains(&id)
            || self.any_scalar_fn.contains(&id)
            || (self.any_scalar_fn_child && child.is::<ScalarFn>())
    }
}

/// Tries the parent-reduce kernels registered for `child` under `parent`.
///
/// Kernels registered for the exact encoding ids come first. A scalar function array also matches
/// kernels registered under [`AnyScalarFn::id`], on either side, which are tried after them.
pub(crate) fn reduce_parent_with_kernels(
    kernels: &ParentReduceKernels,
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    let any_scalar_fn = AnyScalarFn.id();
    let parent_ids = [
        Some(parent.encoding_id()),
        parent.is::<ScalarFn>().then_some(any_scalar_fn),
    ];
    let child_ids = [
        Some(child.encoding_id()),
        child.is::<ScalarFn>().then_some(any_scalar_fn),
    ];
    for parent_id in parent_ids.into_iter().flatten() {
        for child_id in child_ids.into_iter().flatten() {
            if let Some(reduced) =
                try_parent_reduce_kernels(kernels, parent_id, child_id, parent, child, slot_idx)?
            {
                return Ok(Some(reduced));
            }
        }
    }
    Ok(None)
}

fn try_parent_reduce_kernels(
    kernels: &ParentReduceKernels,
    parent_id: ArrayId,
    child_id: ArrayId,
    parent: &ArrayRef,
    child: &ArrayRef,
    slot_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    let Some(reduce_parent_kernels) = kernels.get(parent_id, child_id) else {
        return Ok(None);
    };

    #[allow(clippy::unused_enumerate_index)]
    for (_kernel_idx, kernel) in reduce_parent_kernels.iter().enumerate() {
        if let Some(new_array) = kernel.reduce_parent(child, parent, slot_idx)? {
            vortex_ensure_eq!(
                new_array.len(),
                parent.len(),
                "parent-reduce kernel for {} under {} changed the length",
                child.encoding_id(),
                parent.encoding_id(),
            );
            vortex_ensure_eq!(
                new_array.dtype(),
                parent.dtype(),
                "parent-reduce kernel for {} under {} changed the dtype",
                child.encoding_id(),
                parent.encoding_id(),
            );
            trace_op!(record_session_parent_reduce_applied(
                parent,
                child,
                slot_idx,
                _kernel_idx,
                kernel,
                &new_array,
            ));

            return Ok(Some(new_array));
        }

        trace_op!(record_session_parent_reduce_declined(
            parent,
            child,
            slot_idx,
            _kernel_idx,
            kernel,
        ));
    }

    Ok(None)
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
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::ScalarValue;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn optimize_marks_result_and_skips_it_after() -> VortexResult<()> {
        let sliced = buffer![1i32, 2, 3, 4].into_array().slice(1..3)?;
        assert!(!sliced.optimized_state().is_optimized());

        let optimized = sliced.optimize(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert!(optimized.optimized_state().is_optimized());
        assert!(ArrayRef::ptr_eq(&optimized.optimize(&SESSION)?, &optimized));

        // A new array built over an optimized one starts unoptimized.
        assert!(!optimized.slice(0..1)?.optimized_state().is_optimized());
        Ok(())
    }

    #[test]
    fn optimize_marks_every_descendant() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3, 4].into_array();
        let filtered = array
            .slice(1..4)?
            .filter(Mask::from_iter([true, false, true]))?;

        let optimized = filtered.optimize(&SESSION)?;
        assert!(
            optimized
                .depth_first_traversal()
                .all(|node| node.optimized_state().is_optimized())
        );
        assert!(ArrayRef::ptr_eq(
            &optimized.optimize_recursive(&SESSION)?,
            &optimized
        ));
        Ok(())
    }

    #[test]
    fn optimize_keeps_stats_of_rewritten_array() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3, 4].into_array();
        array
            .statistics()
            .set(Stat::IsSorted, Precision::exact(ScalarValue::Bool(true)));

        let optimized = array.slice(1..3)?.optimize(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert_eq!(
            optimized.statistics().get_as::<bool>(Stat::IsSorted),
            Precision::exact(true)
        );
        Ok(())
    }
}
