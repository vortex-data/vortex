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
//! - [`ArrayOptimizer::optimize`] uses only static rules registered on encoding vtables, and only
//!   rewrites the root node.
//! - [`ArrayOptimizer::optimize_ctx`] also consults the session's active
//!   [`kernels::ArrayKernels`] registry before static parent-reduce rules, so this is the entry
//!   point used by execution. It optimizes the root, then the descendants below it that were
//!   never optimized, then the root again if a child changed.
//! - [`ArrayOptimizer::optimize_recursive`] is the same as [`ArrayOptimizer::optimize_ctx`].
//!
//! With a session, the optimizer works top down. It first tries the rules of a node on its children
//! as they are, so a rule that drops or replaces a child does not optimize that child first. Then
//! it optimizes the children that were never optimized, and if any child changed, it tries the
//! rules of the node again. Thus a parent rule or kernel keyed on a child's encoding still sees
//! that encoding rather than a lazy wrapper around it, such as a slice of an FSST array under a
//! `LIKE`. Children already optimized are left without visiting their subtrees.
//!
//! Whenever a rule rewrites a node, the nodes the rule built are optimized bottom up before the
//! next pass on the node. Otherwise two rules that are inverses of each other, such as
//! `slice(get_item(s))` and `get_item(slice(s))`, could rewrite the node back and forth.
//!
//! Each array records whether the optimizer has taken it to its fixpoint. A new array starts
//! unoptimized, and optimizing with a session marks the arrays it returns, so optimizing an array
//! again returns it without trying any rule. The optimizer marks a node only after it optimized
//! the children of the node, so the descendants of a marked node are marked too. The mark does not
//! record the session: an array optimized with one session is not optimized again with a session
//! that registers more kernels.

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

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

    /// Optimize the array tree using static rules and the active [`kernels::ArrayKernels`]
    /// registry on `session`, if any.
    ///
    /// The descendants that were never optimized are optimized before the root. Session kernels
    /// are checked for each `(parent_encoding_id, child_encoding_id)` pair before the child's
    /// static `PARENT_RULES`. The registry comes from the [`kernels::KernelSession`] on `session`,
    /// if any. If `session` does not contain a [`kernels::KernelSession`], this behaves like
    /// [`Self::optimize`] on the root and its never optimized descendants.
    fn optimize_ctx(&self, session: &VortexSession) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants).
    ///
    /// This is the same as [`Self::optimize_ctx`]: optimizing a node first optimizes every
    /// descendant that was never optimized.
    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        if self.optimized_state().is_optimized() {
            return Ok(self.clone());
        }

        Optimizer::new(None).optimize_root(self.clone())
    }

    fn optimize_ctx(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        if self.optimized_state().is_optimized() {
            return Ok(self.clone());
        }

        let session_kernels = session.kernels();
        let session_kernels: &ArrayKernels = &session_kernels;
        Optimizer::new(Some(session_kernels)).optimize_array(self)
    }

    fn optimize_recursive(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        self.optimize_ctx(session)
    }
}

/// An array keyed by identity: two keys are equal only when they hold the same array, as
/// [`ArrayRef::ptr_eq`] decides.
///
/// The key holds the array, so its address cannot be freed and reused by another array while the
/// key exists.
struct ArrayByPtr(ArrayRef);

impl PartialEq for ArrayByPtr {
    fn eq(&self, other: &Self) -> bool {
        ArrayRef::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ArrayByPtr {}

impl Hash for ArrayByPtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.addr().hash(state);
    }
}

/// The optimizer state, which can be kept for many optimize calls.
///
/// Execution keeps one optimizer for all of [`ArrayRef::execute_until`], so a shared node that one
/// step rewrote is not rewritten again when a later step reaches it through another parent.
pub(crate) struct Optimizer<'a> {
    /// The session kernels, tried before the static parent-reduce rules.
    ///
    /// Without them, the optimizer rewrites only the root node and does not mark the result, so
    /// optimizing it later with a session still tries the session kernels.
    session_kernels: Option<&'a ArrayKernels>,
    /// The shared nodes this optimizer rewrote, each with the array it was rewritten into.
    ///
    /// Many parents can hold the same unoptimized node, such as the struct below every field of a
    /// projection. A rewrite returns a new array and does not mark the node, so without this each
    /// parent would rewrite the node and its whole subtree again. The key keeps the node alive
    /// for the life of the optimizer.
    rewritten_shared: HashMap<ArrayByPtr, ArrayRef>,
}

impl<'a> Optimizer<'a> {
    /// Creates an optimizer that tries `session_kernels` before the static parent-reduce rules.
    pub(crate) fn new(session_kernels: Option<&'a ArrayKernels>) -> Self {
        Self {
            session_kernels,
            rewritten_shared: HashMap::new(),
        }
    }

    /// Optimizes `array` and the descendants below it that were never optimized, and marks the
    /// result.
    ///
    /// If another owner holds `array` too, the rewrite is remembered, and a later call with the
    /// same array returns it.
    pub(crate) fn optimize_array(&mut self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        if !array.is_shared() {
            return self.optimize(array.clone());
        }

        let key = ArrayByPtr(array.clone());
        if let Some(rewritten) = self.rewritten_shared.get(&key) {
            return Ok(rewritten.clone());
        }

        let optimized = self.optimize(array.clone())?;
        if !ArrayRef::ptr_eq(&optimized, array) {
            self.rewritten_shared.insert(key, optimized.clone());
        }

        Ok(optimized)
    }

    /// Optimizes the root of `array` top down: first the root, then the descendants below it that
    /// were never optimized, then the root again while its children change. Marks the result.
    fn optimize(&mut self, array: ArrayRef) -> VortexResult<ArrayRef> {
        if array.optimized_state().is_optimized() {
            return Ok(array);
        }

        // The rules of the root get the first try, on children that can still be lazy, so a rule
        // that drops or replaces a child does not pay to optimize that child's subtree first.
        let mut optimized = self.optimize_root(array.clone())?;

        for _ in 0..=MAX_OPTIMIZER_REWRITE_PASS {
            let with_children = self.optimize_new_nodes(optimized.clone())?;
            if ArrayRef::ptr_eq(&with_children, &optimized) {
                break;
            }

            // A child changed its encoding, so a parent rule keyed on that encoding can apply now.
            optimized = self.optimize_root(with_children)?;
        }

        // A rewrite keeps the logical values, so every statistic of the input stays valid. Lazy
        // constructors such as `slice` put the statistics they carry over on the lazy node.
        if !ArrayRef::ptr_eq(&optimized, &array) {
            optimized.statistics().inherit_from(array.statistics());
        }

        optimized.optimized_state().mark_optimized();
        Ok(optimized)
    }

    /// Runs the reduce and parent-reduce rules on the root of `current_array` to fixpoint.
    ///
    /// With session kernels, the nodes each rewrite built are optimized before the next pass.
    fn optimize_root(&mut self, mut current_array: ArrayRef) -> VortexResult<ArrayRef> {
        let mut any_optimizations = false;

        trace_op!(record_optimize_start(
            &current_array,
            self.session_kernels.is_some()
        ));

        for _ in 0..=MAX_OPTIMIZER_REWRITE_PASS {
            trace_op!(record_optimize_loop_start(&current_array));

            if let Some(new_array) = current_array.reduce()? {
                current_array = self.optimize_rewritten(new_array)?;
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
                if let Some(session_kernels) = self.session_kernels
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
                current_array = self.optimize_rewritten(new_array)?;
                any_optimizations = true;
                trace_op!(record_optimize_loop_end());
                continue;
            }

            trace_op!(record_optimize_parent_reduce_none(&current_array));
            trace_op!(record_optimize_loop_end());

            trace_op!(record_optimize_done(&current_array, any_optimizations));

            return Ok(current_array);
        }

        vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
    }

    /// Optimizes the nodes that a rule built below the array it returned, if the optimizer has
    /// session kernels.
    fn optimize_rewritten(&mut self, array: ArrayRef) -> VortexResult<ArrayRef> {
        if self.session_kernels.is_none() {
            return Ok(array);
        }

        self.optimize_new_nodes(array)
    }

    /// Optimizes the nodes below `array` that were never optimized, such as those a rule built
    /// below its result.
    ///
    /// Rules and constructors build lazy arrays, so these nodes are not yet optimized, at any
    /// depth. Optimizing them as each rewrite happens keeps the optimizer from undoing one rule
    /// with its inverse: `slice(get_item(s))` and `get_item(slice(s))` rewrite into each other
    /// until the inner `slice(s)` collapses into `s`'s own encoding.
    ///
    /// A child that is already optimized is left as it is without visiting its subtree, which
    /// skips the subtrees a rule carries over from the array it rewrote.
    fn optimize_new_nodes(&mut self, array: ArrayRef) -> VortexResult<ArrayRef> {
        let mut new_slots: Option<SmallVec<_>> = None;
        for (slot_idx, slot) in array.slots().iter().enumerate() {
            let optimized = match slot {
                Some(child) if !child.optimized_state().is_optimized() => {
                    let optimized = self.optimize_array(child)?;
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

        // SAFETY: optimizer rules only replace child slots with logically equivalent arrays, so
        // the parent's logical values and statistics remain valid.
        unsafe { array.with_slots(new_slots) }
    }
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
    use crate::arrays::StructArray;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::ScalarValue;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn lazy_slice(array: ArrayRef, range: std::ops::Range<usize>) -> VortexResult<ArrayRef> {
        Ok(SliceArray::try_new(array, range)?.into_array())
    }

    #[test]
    fn optimize_marks_result_and_skips_it_after() -> VortexResult<()> {
        let sliced = lazy_slice(buffer![1i32, 2, 3, 4].into_array(), 1..3)?;
        assert!(!sliced.optimized_state().is_optimized());

        let optimized = sliced.optimize_ctx(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert!(optimized.optimized_state().is_optimized());
        assert!(ArrayRef::ptr_eq(
            &optimized.optimize_ctx(&SESSION)?,
            &optimized
        ));

        // A new array built over an optimized one starts unoptimized.
        assert!(
            !lazy_slice(optimized, 0..1)?
                .optimized_state()
                .is_optimized()
        );
        Ok(())
    }

    #[test]
    fn optimize_without_session_does_not_mark() -> VortexResult<()> {
        let sliced = lazy_slice(buffer![1i32, 2, 3, 4].into_array(), 1..3)?;

        let optimized = sliced.optimize()?;
        assert!(optimized.is::<Primitive>());
        assert!(!optimized.optimized_state().is_optimized());
        Ok(())
    }

    #[test]
    fn optimize_rewrites_a_shared_node_once() -> VortexResult<()> {
        // Both fields hold the same unoptimized slice. The second field reuses the rewrite of the
        // first, so the optimizer does not rewrite the slice and its subtree again.
        let shared = lazy_slice(buffer![1i32, 2, 3, 4].into_array(), 1..3)?;
        let parent =
            StructArray::from_fields(&[("a", shared.clone()), ("b", shared)])?.into_array();

        let optimized = parent.optimize_ctx(&SESSION)?;
        let fields = optimized.slots().iter().flatten().collect::<Vec<_>>();
        assert_eq!(fields.len(), 2);
        assert!(fields[0].is::<Primitive>());
        assert!(ArrayRef::ptr_eq(fields[0], fields[1]));
        Ok(())
    }

    #[test]
    fn optimize_marks_every_descendant() -> VortexResult<()> {
        let sliced = lazy_slice(buffer![1i32, 2, 3, 4].into_array(), 1..4)?;
        let filtered =
            FilterArray::try_new(sliced, Mask::from_iter([true, false, true]))?.into_array();

        let optimized = filtered.optimize_ctx(&SESSION)?;
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

        let sliced = lazy_slice(array, 1..3)?;
        sliced
            .statistics()
            .set(Stat::IsSorted, Precision::exact(ScalarValue::Bool(true)));

        let optimized = sliced.optimize_ctx(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert_eq!(
            optimized.statistics().get_as::<bool>(Stat::IsSorted),
            Precision::exact(true)
        );
        Ok(())
    }
}
