// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The execution engine: iteratively transforms arrays toward canonical form.
//!
//! Execution proceeds through four layers tried in order on each iteration:
//!
//! 1. **`reduce`** -- metadata-only self-rewrite (cheapest).
//! 2. **`reduce_parent`** -- metadata-only child-driven parent rewrite.
//! 3. **`execute_parent`** -- child-driven fused execution (may read buffers).
//! 4. **`execute`** -- the encoding's own decode step (most expensive).
//!
//! The main entry point is [`DynArray::execute_until`], which uses an explicit work stack
//! to drive execution iteratively without recursion. Between steps, the optimizer runs
//! reduce/reduce_parent rules to fixpoint.
//!
//! See <https://docs.vortex.dev/developer-guide/internals/execution> for a full description
//! of the model.

use std::env::VarError;
use std::fmt;
use std::fmt::Display;
use std::ops::Range;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::dtype::DType;
use crate::matcher::Matcher;
use crate::memory::HostAllocatorRef;
use crate::memory::MemorySessionExt;
use crate::optimizer::ArrayOptimizer;

/// Returns the maximum number of iterations to attempt when executing an array before giving up and returning
/// an error, can be by the `VORTEX_MAX_ITERATIONS` env variables, otherwise defaults to 128.
pub(crate) fn max_iterations() -> usize {
    static MAX_ITERATIONS: LazyLock<usize> =
        LazyLock::new(|| match std::env::var("VORTEX_MAX_ITERATIONS") {
            Ok(val) => val.parse::<usize>().unwrap_or_else(|e| {
                vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid usize: {e}")
            }),
            Err(VarError::NotPresent) => 1025,
            Err(VarError::NotUnicode(_)) => {
                vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid unicode string")
            }
        });
    *MAX_ITERATIONS
}

/// Marker trait for types that an [`ArrayRef`] can be executed into.
///
/// Implementors must provide an implementation of `execute` that takes
/// an [`ArrayRef`] and an [`ExecutionCtx`], and produces an instance of the
/// implementor type.
///
/// Users should use the `Array::execute` or `Array::execute_as` methods
pub trait Executable: Sized {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self>;
}

#[expect(clippy::same_name_method)]
impl ArrayRef {
    /// Execute this array to produce an instance of `E`.
    ///
    /// See the [`Executable`] implementation for details on how this execution is performed.
    pub fn execute<E: Executable>(self, ctx: &mut ExecutionCtx) -> VortexResult<E> {
        E::execute(self, ctx)
    }

    /// Execute this array, labeling the execution step with a name for tracing.
    pub fn execute_as<E: Executable>(
        self,
        _name: &'static str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<E> {
        E::execute(self, ctx)
    }

    /// Iteratively execute this array until the [`Matcher`] matches, using an explicit work
    /// stack.
    ///
    /// The stack is a flat `Vec<Entry>` processed back-to-front. Each entry holds an array
    /// and optionally links back to a parent entry (by index) where the result should be
    /// placed when done. Entries without a parent link are the root.
    ///
    /// When an encoding returns `ExecuteSlot` or `ExecuteSlots`, the current array is pushed
    /// back as a parent entry, its children are extracted and pushed in reverse order (so the
    /// first child is on top and processed first). Parent indices are stable because entries
    /// are only pushed/popped at the top.
    ///
    /// For safety, we will error when the number of execution iterations reaches a configurable
    /// maximum (default 128, override with `VORTEX_MAX_ITERATIONS`).
    pub fn execute_until<M: Matcher>(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let root = self.optimize()?;
        let mut stack: Vec<Entry> = vec![Entry::root(root, M::matches)];

        for _ in 0..max_iterations() {
            let mut entry = match stack.pop() {
                Some(e) => e,
                None => vortex_bail!("Empty execution stack"),
            };
            let current = entry.array.take().expect("entry must have array");

            // Step 1: done / canonical — put back into parent or return.
            if (entry.done)(&current) || AnyCanonical::matches(&current) {
                match entry.parent {
                    None => {
                        ctx.log(format_args!("-> {}", current));
                        return Ok(current);
                    }
                    Some(ref link) => {
                        put_back_child(
                            &mut stack,
                            link.parent_idx,
                            link.slot_idx,
                            current,
                            &entry,
                        )?;
                        continue;
                    }
                }
            }

            // Step 2: execute_parent, then execute encoding's own step.
            let result = execute_one_step(current, ctx)?;
            match result {
                OneStepResult::Rewritten(rewritten) => {
                    let rewritten = rewritten.optimize()?;
                    match entry.parent {
                        None => {
                            entry.array = Some(rewritten);
                            stack.push(entry);
                        }
                        Some(link) => {
                            put_back_rewritten(&mut stack, link, rewritten)?;
                        }
                    }
                }
                OneStepResult::Execute(result) => {
                    let (array, step) = result.into_parts();
                    match step {
                        ExecutionStep::ExecuteSlot(i, done) => {
                            push_children(
                                array,
                                vec![(i, done)],
                                entry.parent,
                                entry.done,
                                &mut stack,
                                ctx,
                            )?;
                        }
                        ExecutionStep::ExecuteSlots(slots) => {
                            push_children(array, slots, entry.parent, entry.done, &mut stack, ctx)?;
                        }
                        ExecutionStep::Done => {
                            ctx.log(format_args!("Done: {}", array));
                            entry.array = Some(array);
                            stack.push(entry);
                        }
                    }
                }
            }
        }

        vortex_bail!(
            "Exceeded maximum execution iterations ({}) while executing array",
            max_iterations(),
        )
    }
}

/// A work item on the execution stack.
///
/// Each entry holds an array and optionally links back to a parent entry (by index) where
/// the result should be placed when done. Entries without a parent link are the root.
struct Entry {
    array: Option<ArrayRef>,
    parent: Option<ParentLink>,
    done: DonePredicate,
    original_dtype: DType,
    original_len: usize,
}

/// Points from a child [`Entry`] back to its parent in the stack.
#[derive(Clone)]
struct ParentLink {
    /// Index of the parent [`Entry`] in the stack vec.
    parent_idx: usize,
    /// Which slot in the parent array to put this child's result into.
    slot_idx: usize,
}

impl Entry {
    fn root(array: ArrayRef, done: DonePredicate) -> Self {
        let dtype = array.dtype().clone();
        let len = array.len();
        Self {
            array: Some(array),
            parent: None,
            done,
            original_dtype: dtype,
            original_len: len,
        }
    }

    fn child(array: ArrayRef, parent_idx: usize, slot_idx: usize, done: DonePredicate) -> Self {
        let dtype = array.dtype().clone();
        let len = array.len();
        Self {
            array: Some(array),
            parent: Some(ParentLink {
                parent_idx,
                slot_idx,
            }),
            done,
            original_dtype: dtype,
            original_len: len,
        }
    }
}

/// Put a completed child back into its parent entry in the stack.
fn put_back_child(
    stack: &mut [Entry],
    parent_idx: usize,
    slot_idx: usize,
    child: ArrayRef,
    child_entry: &Entry,
) -> VortexResult<()> {
    debug_assert_eq!(
        child.dtype(),
        &child_entry.original_dtype,
        "slot {} dtype changed from {} to {} during execution",
        slot_idx,
        child_entry.original_dtype,
        child.dtype()
    );
    debug_assert_eq!(
        child.len(),
        child_entry.original_len,
        "slot {} len changed from {} to {} during execution",
        slot_idx,
        child_entry.original_len,
        child.len()
    );
    let parent_array = stack[parent_idx]
        .array
        .take()
        .expect("parent must have array");
    // SAFETY: we assert above that dtype and len are preserved.
    let updated = unsafe { parent_array.put_slot_unchecked(slot_idx, child)? };
    stack[parent_idx].array = Some(updated);
    Ok(())
}

/// Put a rewritten child back into its parent. Siblings remain on the stack
/// and will put themselves back when they complete.
fn put_back_rewritten(stack: &mut [Entry], link: ParentLink, child: ArrayRef) -> VortexResult<()> {
    let parent = stack[link.parent_idx]
        .array
        .take()
        .expect("parent must have array");
    let updated = unsafe { parent.put_slot_unchecked(link.slot_idx, child)? };
    stack[link.parent_idx].array = Some(updated);
    Ok(())
}

/// Extract all undone children from the parent and push them onto the stack in
/// reverse order (first-to-process on top). Each child puts itself back into the
/// parent via [`put_back_child`] when done.
fn push_children(
    mut parent: ArrayRef,
    mut slots: Vec<(usize, DonePredicate)>,
    parent_parent: Option<ParentLink>,
    parent_done: DonePredicate,
    stack: &mut Vec<Entry>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let parent_idx = stack.len();

    // Push parent entry up front; children will reference it by index.
    let mut parent_entry = Entry::root(parent, parent_done);
    parent_entry.parent = parent_parent;
    stack.push(parent_entry);

    // Iterate in reverse so first-to-process ends up on top of the stack.
    // Extract each undone child from the parent as we go.
    let mut parent = stack[parent_idx].array.take().unwrap();
    slots.reverse();
    for (slot_idx, done) in slots {
        let Some(child) = parent.slots().get(slot_idx).and_then(Option::as_ref) else {
            vortex_bail!(
                "Execution requested slot {} but array {} has no occupied slot there",
                slot_idx,
                parent
            );
        };
        if done(child) || AnyCanonical::matches(child) {
            continue;
        }
        let (new_parent, child) = unsafe { parent.take_slot_unchecked(slot_idx) }?;
        parent = new_parent;
        ctx.log(format_args!(
            "ExecuteSlot({slot_idx}): pushing, focusing on {child}"
        ));
        stack.push(Entry::child(child, parent_idx, slot_idx, done));
    }

    // Store parent back (with extracted slots emptied).
    stack[parent_idx].array = Some(parent);

    Ok(())
}

/// Execution context for batch CPU compute.
#[derive(Debug, Clone)]
pub struct ExecutionCtx {
    id: usize,
    session: VortexSession,
    ops: Vec<String>,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub fn new(session: VortexSession) -> Self {
        static EXEC_CTX_ID: AtomicUsize = AtomicUsize::new(0);
        let id = EXEC_CTX_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            session,
            id,
            ops: Vec::new(),
        }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Get the session-scoped host allocator for this execution context.
    pub fn allocator(&self) -> HostAllocatorRef {
        self.session.allocator()
    }

    /// Log an execution step at the current depth.
    ///
    /// Steps are accumulated and dumped as a single trace on Drop at DEBUG level.
    /// Individual steps are also logged at TRACE level for real-time following.
    ///
    /// Use the [`format_args!`] macro to create the `msg` argument.
    pub fn log(&mut self, msg: fmt::Arguments<'_>) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            let formatted = format!(" - {msg}");
            tracing::trace!("exec[{}]: {formatted}", self.id);
            self.ops.push(formatted);
        }
    }
}

impl Display for ExecutionCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exec[{}]", self.id)
    }
}

impl Drop for ExecutionCtx {
    fn drop(&mut self) {
        if !self.ops.is_empty() && tracing::enabled!(tracing::Level::DEBUG) {
            // Unlike itertools `.format()` (panics in 0.14 on second format)
            struct FmtOps<'a>(&'a [String]);
            impl Display for FmtOps<'_> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    for (i, op) in self.0.iter().enumerate() {
                        if i > 0 {
                            f.write_str("\n")?;
                        }
                        f.write_str(op)?;
                    }
                    Ok(())
                }
            }
            tracing::debug!("exec[{}] trace:\n{}", self.id, FmtOps(&self.ops));
        }
    }
}

/// Executing an [`ArrayRef`] into an [`ArrayRef`] is the atomic execution loop within Vortex.
///
/// It attempts to take the smallest possible step of execution such that the returned array
/// is incrementally more "executed" than the input array. In other words, it is closer to becoming
/// a canonical array.
///
/// Most users will not call this method directly, instead preferring to specify an executable
/// target such as [`crate::Columnar`], [`Canonical`], or any of the canonical array types (such as
/// [`crate::arrays::PrimitiveArray`]).
impl Executable for ArrayRef {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let result = execute_one_step(array, ctx)?;
        match result {
            OneStepResult::Rewritten(array) => Ok(array),
            OneStepResult::Execute(result) => {
                let (array, step) = result.into_parts();
                match step {
                    ExecutionStep::Done => {
                        ctx.log(format_args!("-> {}", array));
                        Ok(array)
                    }
                    ExecutionStep::ExecuteSlot(i, _) => {
                        let (array, child) = unsafe { array.take_slot_unchecked(i)? };
                        let executed = child.execute::<ArrayRef>(ctx)?;
                        unsafe { array.put_slot_unchecked(i, executed) }
                    }
                    ExecutionStep::ExecuteSlots(slots) => {
                        // Single-step: execute only the first undone child.
                        for (slot_idx, done) in slots {
                            let already_done = array
                                .slots()
                                .get(slot_idx)
                                .and_then(Option::as_ref)
                                .is_some_and(|c| done(c) || AnyCanonical::matches(c));
                            if already_done {
                                continue;
                            }
                            let (parent, child) = unsafe { array.take_slot_unchecked(slot_idx)? };
                            let executed = child.execute::<ArrayRef>(ctx)?;
                            return unsafe { parent.put_slot_unchecked(slot_idx, executed) };
                        }
                        Ok(array)
                    }
                }
            }
        }
    }
}

/// The result of a single execution step, before the caller decides how to handle it.
enum OneStepResult {
    /// The array was rewritten by a canonical check, reduce, reduce_parent, or execute_parent step.
    Rewritten(ArrayRef),
    /// The encoding's own `execute` step ran and returned an [`ExecutionResult`].
    Execute(ExecutionResult),
}

/// Perform one step of execution on an array, trying each layer in priority order:
///
/// 0. Check for canonical.
/// 1. `reduce` — metadata-only self-rewrite.
/// 2. `reduce_parent` — metadata-only child-driven parent rewrite.
/// 3. `execute_parent` — child-driven fused execution (may read buffers).
/// 4. `execute` — the encoding's own decode step.
///
/// Returns a [`OneStepResult`] so the caller can decide how to handle the outcome
/// (single-step vs iterative loop with stack).
fn execute_one_step(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<OneStepResult> {
    // 0. Check for canonical
    if let Some(canonical) = array.as_opt::<AnyCanonical>() {
        ctx.log(format_args!("-> canonical {}", array));
        return Ok(OneStepResult::Rewritten(
            Canonical::from(canonical).into_array(),
        ));
    }

    // 1. reduce (metadata-only rewrites)
    if let Some(reduced) = array.reduce()? {
        ctx.log(format_args!("reduce: rewrote {} -> {}", array, reduced));
        reduced.statistics().inherit_from(array.statistics());
        return Ok(OneStepResult::Rewritten(reduced));
    }

    // 2. reduce_parent (child-driven metadata-only rewrites)
    for (slot_idx, slot) in array.slots().iter().enumerate() {
        let Some(child) = slot else {
            continue;
        };
        if let Some(reduced_parent) = child.reduce_parent(&array, slot_idx)? {
            ctx.log(format_args!(
                "reduce_parent: slot[{}]({}) rewrote {} -> {}",
                slot_idx,
                child.encoding_id(),
                array,
                reduced_parent
            ));
            reduced_parent.statistics().inherit_from(array.statistics());
            return Ok(OneStepResult::Rewritten(reduced_parent));
        }
    }

    // 3. execute_parent (child-driven optimized execution)
    for (slot_idx, slot) in array.slots().iter().enumerate() {
        let Some(child) = slot else {
            continue;
        };
        if let Some(executed_parent) = child.execute_parent(&array, slot_idx, ctx)? {
            ctx.log(format_args!(
                "execute_parent: slot[{}]({}) rewrote {} -> {}",
                slot_idx,
                child.encoding_id(),
                array,
                executed_parent
            ));
            executed_parent
                .statistics()
                .inherit_from(array.statistics());
            return Ok(OneStepResult::Rewritten(executed_parent));
        }
    }

    // 4. execute (returns an ExecutionResult)
    ctx.log(format_args!("executing {}", array));
    Ok(OneStepResult::Execute(array.execute_encoding(ctx)?))
}

/// A predicate that determines when an array has reached a desired form during execution.
pub type DonePredicate = fn(&ArrayRef) -> bool;

/// Metadata-only step indicator returned alongside an array in [`ExecutionResult`].
///
/// Instead of recursively executing children, encodings return an `ExecutionStep` that tells the
/// scheduler what to do next. This enables the scheduler to manage execution iteratively using
/// an explicit work stack, run cross-step optimizations, and cache shared sub-expressions.
pub enum ExecutionStep {
    /// Request that the scheduler execute the slot at the given index, using the provided
    /// [`DonePredicate`] to determine when the slot is "done", then replace the slot in this
    /// array and re-enter execution.
    ///
    /// Between steps, the scheduler runs reduce/reduce_parent rules to fixpoint, enabling
    /// cross-step optimization (e.g., pushing scalar functions through newly-decoded children).
    ///
    /// Use [`ExecutionResult::execute_slot`] instead of constructing this variant directly.
    ExecuteSlot(usize, DonePredicate),

    /// Request that the scheduler execute multiple slots, each using its paired
    /// [`DonePredicate`], then replace them in this array and re-enter execution.
    ///
    /// Slots are executed in the order they appear in the vector. The scheduler keeps the parent
    /// shape stable until the requested slots are exhausted, because queued slot indices are only
    /// meaningful for the parent that produced them.
    ///
    /// Use [`ExecutionResult::execute_slots`] or [`ExecutionResult::execute_range`] instead of
    /// constructing this variant directly.
    ExecuteSlots(Vec<(usize, DonePredicate)>),

    /// Execution is complete. The array in the accompanying [`ExecutionResult`] is the result.
    /// The scheduler will continue executing if it has not yet reached the target form.
    Done,
}

impl fmt::Debug for ExecutionStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionStep::ExecuteSlot(idx, _) => f.debug_tuple("ExecuteSlot").field(idx).finish(),
            ExecutionStep::ExecuteSlots(slots) => f
                .debug_tuple("ExecuteSlots")
                .field(&slots.iter().map(|(idx, _)| *idx).collect::<Vec<_>>())
                .finish(),
            ExecutionStep::Done => write!(f, "Done"),
        }
    }
}

/// The result of a single execution step on an array encoding.
///
/// Combines an [`ArrayRef`] with an [`ExecutionStep`] to tell the scheduler both what to do next
/// and what array to work with.
pub struct ExecutionResult {
    array: ArrayRef,
    step: ExecutionStep,
}

impl ExecutionResult {
    /// Signal that execution is complete with the given result array.
    pub fn done(result: impl IntoArray) -> Self {
        Self {
            array: result.into_array(),
            step: ExecutionStep::Done,
        }
    }

    /// Request execution of slot at `slot_idx` until it matches the given [`Matcher`].
    ///
    /// The provided array is the (possibly modified) parent that still needs its slot executed.
    pub fn execute_slot<M: Matcher>(array: impl IntoArray, slot_idx: usize) -> Self {
        Self {
            array: array.into_array(),
            step: ExecutionStep::ExecuteSlot(slot_idx, M::matches),
        }
    }

    /// Request execution of multiple slots until each matches the given [`Matcher`].
    ///
    /// The slots are executed in iterator order. This is useful for encodings whose next
    /// execution step requires a batch of homogeneous children, such as all chunks in a chunked
    /// array. The iterator must yield at least one slot.
    pub fn execute_slots<M: Matcher>(
        array: impl IntoArray,
        slots: impl IntoIterator<Item = usize>,
    ) -> Self {
        Self {
            array: array.into_array(),
            step: ExecutionStep::ExecuteSlots(
                slots
                    .into_iter()
                    .map(|slot_idx| (slot_idx, M::matches as DonePredicate))
                    .collect(),
            ),
        }
    }

    /// Request execution of a non-empty contiguous range of slots until each matches the given
    /// [`Matcher`].
    pub fn execute_range<M: Matcher>(array: impl IntoArray, slots: Range<usize>) -> Self {
        Self::execute_slots::<M>(array, slots)
    }

    /// Returns a reference to the array.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Returns a reference to the step.
    pub fn step(&self) -> &ExecutionStep {
        &self.step
    }

    /// Decompose into parts.
    pub fn into_parts(self) -> (ArrayRef, ExecutionStep) {
        (self.array, self.step)
    }
}

impl fmt::Debug for ExecutionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionResult")
            .field("array", &self.array)
            .field("step", &self.step)
            .finish()
    }
}

/// Require that a child array matches `$M`. If the child already matches, returns the same
/// array unchanged. Otherwise, early-returns an [`ExecutionResult`] requesting execution of
/// child `$idx` until it matches `$M`.
///
/// ```ignore
/// let array = require_child!(array, array.codes(), 0 => Primitive);
/// let array = require_child!(array, array.values(), 1 => AnyCanonical);
/// ```
#[macro_export]
macro_rules! require_child {
    ($parent:expr, $child:expr, $idx:expr => $M:ty) => {{
        if !$child.is::<$M>() {
            return Ok($crate::ExecutionResult::execute_slot::<$M>(
                $parent.clone(),
                $idx,
            ));
        }
        $parent
    }};
}

/// Like [`require_child!`], but for optional children. If the child is `None`, this is a no-op.
/// If the child is `Some` but does not match `$M`, early-returns an [`ExecutionResult`] requesting
/// execution of child `$idx`.
///
/// Unlike `require_child!`, this is a statement macro (no value produced) and does not clone
/// `$parent` — it is moved into the early-return path.
///
/// ```ignore
/// require_opt_child!(array, array.patches().map(|p| p.indices()), 1 => Primitive);
/// ```
#[macro_export]
macro_rules! require_opt_child {
    ($parent:expr, $child_opt:expr, $idx:expr => $M:ty) => {
        if $child_opt.is_some_and(|child| !child.is::<$M>()) {
            return Ok($crate::ExecutionResult::execute_slot::<$M>($parent, $idx));
        }
    };
}

/// Require that patch slots (indices, values, and optionally chunk_offsets) are `Primitive`.
/// If no patches are present (slots are `None`), this is a no-op.
///
/// Like [`require_opt_child!`], `$parent` is moved (not cloned) into the early-return path.
///
/// ```ignore
/// require_patches!(array, PATCH_INDICES_SLOT, PATCH_VALUES_SLOT, PATCH_CHUNK_OFFSETS_SLOT);
/// ```
#[macro_export]
macro_rules! require_patches {
    ($parent:expr, $indices_slot:expr, $values_slot:expr, $chunk_offsets_slot:expr) => {
        $crate::require_opt_child!(
            $parent,
            $parent.slots()[$indices_slot].as_ref(),
            $indices_slot => $crate::arrays::Primitive
        );
        $crate::require_opt_child!(
            $parent,
            $parent.slots()[$values_slot].as_ref(),
            $values_slot => $crate::arrays::Primitive
        );
        $crate::require_opt_child!(
            $parent,
            $parent.slots()[$chunk_offsets_slot].as_ref(),
            $chunk_offsets_slot => $crate::arrays::Primitive
        );
    };
}

/// Require that the validity slot is a [`Bool`](crate::arrays::Bool) array. If validity is not
/// array-backed (e.g. `NonNullable` or `AllValid`), this is a no-op. If it is array-backed but
/// not `Bool`, early-returns an [`ExecutionResult`] requesting execution of the validity slot.
///
/// Like [`require_opt_child!`], `$parent` is moved (not cloned) into the early-return path.
///
/// ```ignore
/// require_validity!(array, VALIDITY_SLOT);
/// ```
#[macro_export]
macro_rules! require_validity {
    ($parent:expr, $idx:expr) => {
        $crate::require_opt_child!(
            $parent,
            $parent.slots()[$idx].as_ref(),
            $idx => $crate::arrays::Bool
        );
    };
}

/// Extension trait for creating an execution context from a session.
pub trait VortexSessionExecute {
    /// Create a new execution context from this session.
    fn create_execution_ctx(&self) -> ExecutionCtx;
}

impl VortexSessionExecute for VortexSession {
    fn create_execution_ctx(&self) -> ExecutionCtx {
        ExecutionCtx::new(self.clone())
    }
}
