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
use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::array::ArraySlots;
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
        if array.is_shared()
            && let Some(rewritten) = self.rewritten_shared.get(&ArrayByPtr(array.clone()))
        {
            return Ok(rewritten.clone());
        }

        self.optimize(array.clone())
    }

    /// Optimizes the root of `array` top down: first the root, then the descendants below it that
    /// were never optimized, then the root again while its children change. Marks the result.
    fn optimize(&mut self, array: ArrayRef) -> VortexResult<ArrayRef> {
        if array.optimized_state().is_optimized() {
            return Ok(array);
        }

        if self.session_kernels.is_some() {
            // The root pass walks the nodes each rewrite built, so it needs frames anyway.
            return self.run(OptimizeFrame::new(array, Scope::Tree));
        }

        // The rules of the root get the first try, on children that can still be lazy, so a rule
        // that drops or replaces a child does not pay to optimize that child's subtree first.
        let optimized = self.optimize_root(array.clone())?;

        // Most lazy nodes sit over children that are already optimized, so the root rules alone
        // took them to their fixpoint and no frame is needed.
        if children_optimized(&optimized) {
            return Ok(self.mark_optimized(array, optimized));
        }

        let mut frame = OptimizeFrame::new(array, Scope::Tree);
        frame.current = optimized;
        frame.start_children(After::Finish);
        self.run(frame)
    }

    /// Runs the reduce and parent-reduce rules on the root of `array` to fixpoint.
    ///
    /// With session kernels, the nodes each rewrite built are optimized before the next pass.
    fn optimize_root(&mut self, array: ArrayRef) -> VortexResult<ArrayRef> {
        if self.session_kernels.is_some() {
            return self.run(OptimizeFrame::new(array, Scope::Root));
        }

        // Without session kernels the nodes a rewrite built are left alone, so the fixpoint loop
        // needs no frames. This is the path of every lazy constructor, so it stays allocation free.
        let mut current = array;
        let mut passes = 0;
        let mut any_optimizations = false;
        while let Some(new_array) = self.reduce_root(&current, &mut passes)? {
            current = new_array;
            any_optimizations = true;
            trace_op!(record_optimize_loop_end());
        }
        trace_op!(record_optimize_parent_reduce_none(&current));
        trace_op!(record_optimize_loop_end());
        trace_op!(record_optimize_done(&current, any_optimizations));
        Ok(current)
    }

    /// Drives the frames of the optimization from `root`: one frame per node whose subtree is
    /// being optimized.
    ///
    /// The walk keeps its own stack, so a lazy tree deeper than the thread's stack allows, such as
    /// a chain of thousands of nested scalar functions, still optimizes.
    fn run(&mut self, root: OptimizeFrame) -> VortexResult<ArrayRef> {
        let mut stack: SmallVec<[OptimizeFrame; 8]> = smallvec![root];

        loop {
            let frame = stack
                .last_mut()
                .vortex_expect("a frame is on the stack until its subtree is optimized");

            match frame.phase {
                Phase::Root => {
                    if let Some(new_array) =
                        self.reduce_root(&frame.current, &mut frame.root_passes)?
                    {
                        frame.current = new_array;
                        frame.any_optimizations = true;
                        trace_op!(record_optimize_loop_end());
                        // The rules of the root get the first try, on children that can still
                        // be lazy, so a rule that drops or replaces a child does not pay to
                        // optimize that child's subtree first. The nodes a rewrite built are
                        // optimized before the next pass, with session kernels.
                        if self.session_kernels.is_some() {
                            frame.start_children(After::ContinueRoot);
                        }
                        continue;
                    }

                    trace_op!(record_optimize_parent_reduce_none(&frame.current));
                    trace_op!(record_optimize_loop_end());
                    trace_op!(record_optimize_done(
                        &frame.current,
                        frame.any_optimizations
                    ));

                    match frame.scope {
                        Scope::Root => {
                            let frame = stack.pop().vortex_expect("the frame was just read");
                            match stack.last_mut() {
                                Some(parent) => parent.absorb(frame.current),
                                None => return Ok(frame.current),
                            }
                        }
                        // Most lazy nodes sit over children that are already optimized, so the
                        // root rules alone took them to their fixpoint.
                        Scope::Tree
                            if frame.outer_passes > MAX_OPTIMIZER_REWRITE_PASS
                                || children_optimized(&frame.current) =>
                        {
                            if let Some(result) = self.finish(&mut stack) {
                                return Ok(result);
                            }
                        }
                        Scope::Tree => frame.start_children(After::Finish),
                    }
                }
                Phase::Children(after) => {
                    if let Some(child) = frame.next_unoptimized_child(&self.rewritten_shared) {
                        stack.push(OptimizeFrame::new(child, Scope::Tree));
                        continue;
                    }

                    let changed = match frame.take_children()? {
                        Some(with_children) => {
                            frame.current = with_children;
                            true
                        }
                        None => false,
                    };
                    match after {
                        // Continue the root fixpoint over the rewrite's optimized children.
                        After::ContinueRoot => frame.phase = Phase::Root,
                        After::Finish if changed => {
                            // A child changed its encoding, so a parent rule keyed on that
                            // encoding can apply now.
                            frame.outer_passes += 1;
                            frame.rerun_root();
                        }
                        After::Finish => {
                            if let Some(result) = self.finish(&mut stack) {
                                return Ok(result);
                            }
                        }
                    }
                }
            }
        }
    }

    /// One pass of the root fixpoint loop over `current`, the `passes`-th so far: the rewrite a
    /// reduce or parent-reduce rule made, if any applied.
    #[inline]
    fn reduce_root(
        &self,
        current: &ArrayRef,
        passes: &mut usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if *passes == 0 {
            trace_op!(record_optimize_start(
                current,
                self.session_kernels.is_some()
            ));
        }
        if *passes > MAX_OPTIMIZER_REWRITE_PASS {
            vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
        }
        *passes += 1;

        trace_op!(record_optimize_loop_start(current));

        if let Some(new_array) = current.reduce()? {
            return Ok(Some(new_array));
        }

        trace_op!(record_optimize_reduce_none(current));

        // Try children in order; the first parent rewrite restarts the fixpoint loop.
        for (slot_idx, slot) in current.slots().iter().enumerate() {
            let Some(child) = slot else {
                continue;
            };

            // Session kernels take precedence over the child's static parent-reduce rules.
            if let Some(session_kernels) = self.session_kernels
                && let Some(new_array) =
                    try_session_parent_reduce(session_kernels, current, child, slot_idx)?
            {
                return Ok(Some(new_array));
            }

            if let Some(new_array) = child.reduce_parent(current, slot_idx)? {
                return Ok(Some(new_array));
            }
        }

        Ok(None)
    }

    /// Pops the top frame as optimized: marks its result and hands it to the parent frame, or
    /// returns it when the frame was the root.
    fn finish(&mut self, stack: &mut SmallVec<[OptimizeFrame; 8]>) -> Option<ArrayRef> {
        let frame = stack.pop().vortex_expect("the frame was just read");
        let OptimizeFrame { input, current, .. } = frame;
        let current = self.mark_optimized(input, current);

        match stack.last_mut() {
            Some(parent) => {
                parent.absorb(current);
                None
            }
            None => Some(current),
        }
    }
}

/// Whether every child of `array` is already optimized.
#[inline]
fn children_optimized(array: &ArrayRef) -> bool {
    array
        .slots()
        .iter()
        .flatten()
        .all(|child| child.optimized_state().is_optimized())
}

impl Optimizer<'_> {
    /// Marks `optimized`, the fixpoint of `input` and its subtree, and remembers the rewrite when
    /// `input` is shared.
    fn mark_optimized(&mut self, input: ArrayRef, optimized: ArrayRef) -> ArrayRef {
        // A rewrite keeps the logical values, so every statistic of the input stays valid. Lazy
        // constructors such as `slice` put the statistics they carry over on the lazy node.
        if !ArrayRef::ptr_eq(&optimized, &input) {
            optimized.statistics().inherit_from(input.statistics());
            if input.is_shared() {
                self.rewritten_shared
                    .insert(ArrayByPtr(input), optimized.clone());
            }
        }

        optimized.optimized_state().mark_optimized();
        optimized
    }
}

/// How much of a node's tree an [`OptimizeFrame`] optimizes.
#[derive(Clone, Copy)]
enum Scope {
    /// The root rules to fixpoint, without marking the result.
    Root,
    /// The root, the descendants below it that were never optimized, then the root again while
    /// its children change, marking the result.
    Tree,
}

/// What an [`OptimizeFrame`] does next.
#[derive(Clone, Copy)]
enum Phase {
    /// One pass of the reduce and parent-reduce rules on the root.
    Root,
    /// Walk the children, optimizing those never optimized.
    Children(After),
}

/// What an [`OptimizeFrame`] does after a walk over its children.
#[derive(Clone, Copy)]
enum After {
    /// Continue the root fixpoint loop: the children were those of a rewrite.
    ContinueRoot,
    /// Rerun the root fixpoint loop if a child changed, else finish the frame.
    Finish,
}

/// A node whose subtree is being optimized.
struct OptimizeFrame {
    scope: Scope,
    phase: Phase,
    /// The node as it was handed to the optimizer.
    input: ArrayRef,
    /// The node as rewritten so far.
    current: ArrayRef,
    /// Passes of the root fixpoint loop so far, reset when the loop reruns over changed children.
    root_passes: usize,
    any_optimizations: bool,
    /// Reruns of the root fixpoint loop over children that changed.
    outer_passes: usize,
    /// The next slot the children walk looks at.
    next_slot: usize,
    /// The children walked so far. Stays `None` until a child actually changes, so a node whose
    /// children are left as they are is not rebuilt, and the frame stays small to move.
    new_slots: Option<Box<ArraySlots>>,
}

impl OptimizeFrame {
    fn new(array: ArrayRef, scope: Scope) -> Self {
        Self {
            scope,
            phase: Phase::Root,
            input: array.clone(),
            current: array,
            root_passes: 0,
            any_optimizations: false,
            outer_passes: 0,
            next_slot: 0,
            new_slots: None,
        }
    }

    /// Start walking the children of `current`, to continue with `after` afterwards.
    fn start_children(&mut self, after: After) {
        self.phase = Phase::Children(after);
        self.next_slot = 0;
        self.new_slots = None;
    }

    /// Start the root fixpoint loop again, over the current children.
    fn rerun_root(&mut self) {
        self.phase = Phase::Root;
        self.root_passes = 0;
        self.any_optimizations = false;
    }

    /// The next child of `current` that was never optimized, with the walk advanced past it. A
    /// shared child this optimizer already rewrote takes its rewrite instead of a walk.
    fn next_unoptimized_child(
        &mut self,
        rewritten_shared: &HashMap<ArrayByPtr, ArrayRef>,
    ) -> Option<ArrayRef> {
        while let Some(slot) = self.current.slots().get(self.next_slot) {
            self.next_slot += 1;
            match slot {
                Some(child) if !child.optimized_state().is_optimized() => {
                    if child.is_shared()
                        && let Some(rewritten) = rewritten_shared.get(&ArrayByPtr(child.clone()))
                    {
                        let rewritten = rewritten.clone();
                        self.absorb(rewritten);
                        continue;
                    }
                    return Some(child.clone());
                }
                _ => self.keep_slot(),
            }
        }
        None
    }

    /// Take the optimized child of the slot the walk last advanced past.
    fn absorb(&mut self, optimized: ArrayRef) {
        let slot_idx = self.next_slot - 1;
        let slots = self.current.slots();
        let unchanged = slots[slot_idx]
            .as_ref()
            .is_some_and(|child| ArrayRef::ptr_eq(child, &optimized));
        if unchanged {
            self.keep_slot();
            return;
        }
        self.new_slots
            .get_or_insert_with(|| {
                let mut new_slots = ArraySlots::with_capacity(slots.len());
                new_slots.extend(slots[..slot_idx].iter().cloned());
                Box::new(new_slots)
            })
            .push(Some(optimized));
    }

    /// Keep the slot the walk last advanced past as it is.
    fn keep_slot(&mut self) {
        if let Some(new_slots) = &mut self.new_slots {
            new_slots.push(self.current.slots()[self.next_slot - 1].clone());
        }
    }

    /// The node rebuilt over its walked children, or `None` when no child changed.
    fn take_children(&mut self) -> VortexResult<Option<ArrayRef>> {
        let Some(new_slots) = self.new_slots.take() else {
            return Ok(None);
        };

        // SAFETY: optimizer rules only replace child slots with logically equivalent arrays, so
        // the parent's logical values and statistics remain valid.
        unsafe { self.current.clone().with_slots(*new_slots) }.map(Some)
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
