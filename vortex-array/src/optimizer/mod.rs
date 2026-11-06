// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The optimizer applies metadata-only rewrite rules (`reduce` and `reduce_parent`) in a
//! fixpoint loop until no more rules apply.
//!
//! Optimization runs between execution steps. After a child is decoded, `reduce_parent` rules
//! that did not match before can match.
//!
//! Parent-reduce rules come from the session's [`kernels::ArrayKernels`]. To optimize, use
//! [`ArrayOptimizer::optimize_ctx`] with the session or with an [`ExecutionCtx`] created from it.
//!
//! Array constructors do not optimize. The optimizer works top down: it tries the rules of a node,
//! then optimizes the children that were never optimized, and tries the rules of the node again if
//! a child changed. Thus a rule for a child encoding sees that encoding, not a lazy wrapper around
//! it. The nodes that a rule builds are optimized before the next pass, so two rules that are
//! inverses of each other cannot rewrite a node back and forth.
//!
//! The optimizer marks the arrays that it returns. Optimizing a marked array again tries no rule.

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

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
/// them, so this skips rules on decoded arrays without the guarantee that none applies. A rule
/// that rewrites a parent of a decoded node, such as a slice pushed into a dictionary, optimizes
/// the new nodes it builds, but not the decoded nodes that it keeps.
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
            KernelSource::Session(session) => session.kernels().with_reduce_parent_kernels(f),
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
    /// Optimize the array with only the parent-reduce rules that `vortex-array` registers by
    /// default.
    ///
    /// Encodings outside `vortex-array` register their rules with a session, so this misses them.
    /// The result is not marked as optimized, so optimizing it later with a session still tries
    /// the session's rules.
    #[deprecated(
        note = "use `optimize_ctx` with the session, or an `ExecutionCtx` created from it, instead"
    )]
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Optimize the root array node by running reduce and reduce_parent rules to fixpoint.
    ///
    /// The root is optimized first, then the descendants that were never optimized, then the root
    /// again if a child changed. Parent-reduce kernels are looked up for each
    /// `(parent_encoding_id, child_encoding_id)` pair in the [`kernels::KernelSession`] of the
    /// session, or of the session an execution context was created from.
    fn optimize_ctx(&self, kernels: &(impl AsOptimizerKernels + ?Sized)) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants).
    ///
    /// This is the same as [`Self::optimize_ctx`]: optimizing a node also optimizes every
    /// descendant that was never optimized.
    fn optimize_recursive(
        &self,
        kernels: &(impl AsOptimizerKernels + ?Sized),
    ) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        /// A session that holds only the parent-reduce rules `vortex-array` registers by default.
        static BUILTIN_SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

        if self.optimized_state().is_optimized() {
            return Ok(self.clone());
        }

        BUILTIN_SESSION
            .kernels()
            .with_reduce_parent_kernels(|kernels| {
                Optimizer {
                    kernels,
                    mark: false,
                    rewritten_shared: HashMap::new(),
                }
                .optimize_array(self)
            })
    }

    fn optimize_ctx(&self, kernels: &(impl AsOptimizerKernels + ?Sized)) -> VortexResult<ArrayRef> {
        if self.optimized_state().is_optimized() {
            return Ok(self.clone());
        }

        kernels
            .optimizer_kernels()
            .with_reduce_parent_kernels(|kernels| optimize_with_kernels(self, kernels))
    }

    fn optimize_recursive(
        &self,
        kernels: &(impl AsOptimizerKernels + ?Sized),
    ) -> VortexResult<ArrayRef> {
        self.optimize_ctx(kernels)
    }
}

/// [`ArrayOptimizer::optimize_ctx`] against a snapshot of the session's parent-reduce kernels.
pub(crate) fn optimize_with_kernels(
    array: &ArrayRef,
    kernels: &ParentReduceKernels,
) -> VortexResult<ArrayRef> {
    Optimizer::new(kernels).optimize_array(array)
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
    kernels: &'a ParentReduceKernels,
    /// Whether to mark the arrays this optimizer returns as optimized. Only the deprecated
    /// [`ArrayOptimizer::optimize`] leaves them unmarked, since it misses the session's rules.
    mark: bool,
    /// The shared nodes this optimizer rewrote, each with the array it was rewritten into.
    ///
    /// Many parents can hold the same unoptimized node, such as the struct below every field of a
    /// projection. A rewrite returns a new array and does not mark the node, so without this each
    /// parent would rewrite the node and its whole subtree again. The key keeps the node alive
    /// for the life of the optimizer.
    rewritten_shared: HashMap<ArrayByPtr, ArrayRef>,
}

impl<'a> Optimizer<'a> {
    /// Creates an optimizer that tries `kernels` and marks the arrays it returns.
    pub(crate) fn new(kernels: &'a ParentReduceKernels) -> Self {
        Self {
            kernels,
            mark: true,
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

    /// Optimizes `array` top down: first the root, then the descendants below it that were never
    /// optimized, then the root again while its children change. Marks the result.
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

        if self.mark {
            optimized.optimized_state().mark_optimized();
        }

        Ok(optimized)
    }

    /// Runs the reduce and parent-reduce rules on the root of `current_array` to fixpoint.
    ///
    /// The nodes each rewrite built are optimized before the next pass.
    fn optimize_root(&mut self, mut current_array: ArrayRef) -> VortexResult<ArrayRef> {
        let mut any_optimizations = false;

        trace_op!(record_optimize_start(&current_array));

        for _ in 0..=MAX_OPTIMIZER_REWRITE_PASS {
            trace_op!(record_optimize_loop_start(&current_array));

            if let Some(new_array) = current_array.reduce()? {
                current_array = self.optimize_new_nodes(new_array)?;
                any_optimizations = true;
                trace_op!(record_optimize_loop_end());
                continue;
            }

            trace_op!(record_optimize_reduce_none(&current_array));

            // Try children in order; the first parent rewrite restarts the fixpoint loop. A parent
            // that no kernel is registered for skips its children without probing each pair.
            let mut reduced_parent = None;
            let candidates = ReduceParentCandidates::new(self.kernels, &current_array);
            if !candidates.is_empty() {
                for (slot_idx, slot) in current_array.slots().iter().enumerate() {
                    let Some(child) = slot else {
                        continue;
                    };
                    if candidates.contains(child)
                        && let Some(new_array) = reduce_parent_with_kernels(
                            self.kernels,
                            &current_array,
                            child,
                            slot_idx,
                        )?
                    {
                        reduced_parent = Some(new_array);
                        break;
                    }
                }
            }

            if let Some(new_array) = reduced_parent {
                current_array = self.optimize_new_nodes(new_array)?;
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
    use crate::arrays::StructArray;
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

        let optimized = sliced.optimize_ctx(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert!(optimized.optimized_state().is_optimized());
        assert!(ArrayRef::ptr_eq(
            &optimized.optimize_ctx(&SESSION)?,
            &optimized
        ));

        // A new array built over an optimized one starts unoptimized.
        assert!(!optimized.slice(0..1)?.optimized_state().is_optimized());
        Ok(())
    }

    #[test]
    fn optimize_rewrites_a_shared_node_once() -> VortexResult<()> {
        // Both fields hold the same unoptimized slice. The second field reuses the rewrite of the
        // first, so the optimizer does not rewrite the slice and its subtree again.
        let shared = buffer![1i32, 2, 3, 4].into_array().slice(1..3)?;
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
        let array = buffer![1i32, 2, 3, 4].into_array();
        let filtered = array
            .slice(1..4)?
            .filter(Mask::from_iter([true, false, true]))?;

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

        let optimized = array.slice(1..3)?.optimize_ctx(&SESSION)?;
        assert!(optimized.is::<Primitive>());
        assert_eq!(
            optimized.statistics().get_as::<bool>(Stat::IsSorted),
            Precision::exact(true)
        );
        Ok(())
    }
}
