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
//! The main entry point is [`ArrayRef::execute_until`], which uses an explicit work stack
//! to drive execution iteratively without recursion. Between steps, the optimizer runs
//! reduce/reduce_parent rules to fixpoint using the active [`ExecutionCtx`] session, so
//! session-registered optimizer kernels participate during execution.
//!
//! See <https://docs.vortex.dev/developer-guide/internals/execution> for a full description
//! of the model.

use std::env::VarError;
use std::fmt;
use std::fmt::Display;
use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::array::ArrayId;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::matcher::Matcher;
use crate::memory::HostAllocatorRef;
use crate::memory::MemorySessionExt;
use crate::optimizer::ArrayOptimizer;
use crate::stats::ArrayStats;
use crate::stats::StatsSet;

/// Returns the maximum number of iterations to attempt when executing an array before giving up and returning
/// an error, can be by the `VORTEX_MAX_ITERATIONS` env variables, otherwise defaults to 2^22.
pub(crate) fn max_iterations() -> usize {
    static MAX_ITERATIONS: LazyLock<usize> =
        LazyLock::new(|| match std::env::var("VORTEX_MAX_ITERATIONS") {
            Ok(val) => val.parse::<usize>().unwrap_or_else(|e| {
                vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid usize: {e}")
            }),
            Err(VarError::NotPresent) => 2 << 21, // 2 ^ 22
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
    /// Note: the returned array may not match `M`. If execution converges to a canonical form
    /// that does not match `M`, the canonical array is returned since no further execution
    /// progress is possible.
    ///
    /// For safety, we will error when the number of execution iterations reaches a configurable
    /// maximum (default 128, override with `VORTEX_MAX_ITERATIONS`).
    pub fn execute_until<M: Matcher>(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        execute_loop(self, M::matches, ctx)
    }
}

struct StackFrame {
    parent_array: ArrayRef,
    parent_builder: Option<Box<dyn ArrayBuilder>>,
    slot_idx: usize,
    done: DonePredicate,
    original_dtype: DType,
    original_len: usize,
}

/// Execution context for batch CPU compute.
#[derive(Debug, Clone)]
pub struct ExecutionCtx {
    session: VortexSession,
    #[cfg(debug_assertions)]
    id: usize,
    #[cfg(debug_assertions)]
    ops: Vec<String>,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            #[cfg(debug_assertions)]
            id: {
                static EXEC_CTX_ID: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                EXEC_CTX_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            #[cfg(debug_assertions)]
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
        #[cfg(debug_assertions)]
        if tracing::enabled!(tracing::Level::DEBUG) {
            let formatted = format!(" - {msg}");
            tracing::trace!("exec[{}]: {formatted}", self.id);
            self.ops.push(formatted);
        }
        let _ = msg;
    }
}

impl Display for ExecutionCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(debug_assertions)]
        return write!(f, "exec[{}]", self.id);
        #[cfg(not(debug_assertions))]
        write!(f, "exec")
    }
}

#[cfg(debug_assertions)]
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

/// Single-step execution: takes one step toward canonical form.
///
/// Steps through reduce, reduce_parent, execute_parent, then execute. For `ExecuteSlot`,
/// only a single child execution step is performed — the child is executed once and put back,
/// making this a lightweight, bounded operation.
///
/// **However**, if `execute_step` returns [`ExecutionStep::AppendChild`], this implementation
/// drives the *entire* array to completion via [`execute_into_builder`] in a single call.
/// This can do substantially more work than a normal step because it creates a builder and
/// fully decodes the array into that builder before returning. Callers should be aware that a
/// single `.execute::<ArrayRef>(ctx)` call may perform O(n_children * decode_cost) work when
/// `AppendChild` is returned.
impl Executable for ArrayRef {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(canonical) = array.as_opt::<AnyCanonical>() {
            ctx.log(format_args!("-> canonical {}", array));
            return Ok(Canonical::from(canonical).into_array());
        }

        if let Some(reduced) = array.reduce()? {
            ctx.log(format_args!("reduce: rewrote {} -> {}", array, reduced));
            reduced.statistics().inherit_from(array.statistics());
            return Ok(reduced);
        }

        for (slot_idx, slot) in array.slots().iter().enumerate() {
            let Some(child) = slot else { continue };
            if let Some(reduced_parent) = child.reduce_parent(&array, slot_idx)? {
                ctx.log(format_args!(
                    "reduce_parent: slot[{}]({}) rewrote {} -> {}",
                    slot_idx,
                    child.encoding_id(),
                    array,
                    reduced_parent
                ));
                reduced_parent.statistics().inherit_from(array.statistics());
                return Ok(reduced_parent);
            }
        }

        for (slot_idx, slot) in array.slots().iter().enumerate() {
            let Some(child) = slot else { continue };
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
                return Ok(executed_parent);
            }
        }

        ctx.log(format_args!("executing {}", array));
        let result = execute_step_checked(array, ctx)?;
        let (array, step) = result.into_parts();
        match step {
            ExecutionStep::Done => {
                ctx.log(format_args!("-> {}", array));
                Ok(array)
            }
            ExecutionStep::ExecuteSlot(i, _) => {
                let child = array.slots()[i].clone().vortex_expect("valid slot index");
                let executed_child = child.execute::<ArrayRef>(ctx)?;
                array.with_slot(i, executed_child)
            }
            ExecutionStep::AppendChild(_) => {
                // Single-step: build the entire parent via the builder path.
                let builder = builder_with_capacity_in(ctx.allocator(), array.dtype(), array.len());
                let mut builder = execute_into_builder(array, builder, ctx)?;
                Ok(builder.finish())
            }
        }
    }
}

/// Execute `array` into the given `builder`.
///
/// This uses the encoding's [`crate::array::VTable::append_to_builder`] implementation. Most
/// encodings use the default path of `execute::<Canonical>` followed by `builder.extend_from_array`,
/// while encodings like `Chunked` can override that to append child-by-child without materializing
/// the entire parent.
///
/// The builder must have a [`DType`] that is a nullability-superset of `array.dtype()`.
pub fn execute_into_builder(
    array: ArrayRef,
    mut builder: Box<dyn ArrayBuilder>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ArrayBuilder>> {
    array.append_to_builder(builder.as_mut(), ctx)?;
    Ok(builder)
}

/// Iterative execution loop for array-to-array execution.
fn execute_loop(
    array: ArrayRef,
    root_done: DonePredicate,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut current_array = array;
    let mut current_builder: Option<Box<dyn ArrayBuilder>> = None;
    let mut stack: Vec<StackFrame> = Vec::new();
    let max_iterations = max_iterations();

    for _ in 0..max_iterations {
        let is_done = stack.last().map_or(root_done, |frame| frame.done);

        if is_done(&current_array) || AnyCanonical::matches(&current_array) {
            match stack.pop() {
                None => {
                    debug_assert!(
                        current_builder.is_none(),
                        "root activation should not retain a builder"
                    );
                    ctx.log(format_args!("-> {}", current_array));
                    return Ok(current_array);
                }
                Some(frame) => {
                    (current_array, current_builder) = pop_frame(frame, current_array)?;
                    continue;
                }
            }
        }

        // ── Step 2a: execute_parent against stack parent ───────────────────
        //
        // When executing a child for ExecuteSlot, try execute_parent against
        // the suspended parent on the stack. This lets kernels like RunEnd's
        // FilterKernel fire before the child is forced to canonical.
        if let Some(frame) = stack.last() {
            if let Some(result) =
                current_array.execute_parent(&frame.parent_array, frame.slot_idx, ctx)?
            {
                ctx.log(format_args!(
                    "execute_parent (stack) rewrote {} -> {}",
                    current_array, result
                ));
                let frame = stack.pop().vortex_expect("just peeked");
                current_array = result.optimize_ctx(ctx.session())?;
                current_builder = frame.parent_builder;
                continue;
            }
        }

        // ── Step 2b: execute_parent ─────────────────────────────────────────
        //
        // Skip execute_parent when we have a builder attached — the parent array is
        // executor-private suspended state with child slots already taken out.
        if current_builder.is_none()
            && let Some(rewritten) = try_execute_parent(&current_array, ctx)?
        {
            ctx.log(format_args!(
                "execute_parent rewrote {} -> {}",
                current_array, rewritten
            ));
            current_array = rewritten.optimize_ctx(ctx.session())?;
            continue;
        }

        // ── Step 3: execute step ───────────────────────────────────────────
        let expected_len = current_array.len();
        let expected_dtype = current_array.dtype().clone();
        let stats = current_array.statistics().to_array_stats();
        let encoding_id = current_array.encoding_id();
        let result = execute_step_unchecked(current_array, ctx)?;
        let (array, step) = result.into_parts();
        match step {
            ExecutionStep::ExecuteSlot(i, done) => {
                let (parent, child) = unsafe { array.take_slot_unchecked(i) }?;
                ctx.log(format_args!(
                    "ExecuteSlot({i}): pushing {}, focusing on {}",
                    parent, child
                ));
                stack.push(StackFrame {
                    parent_array: parent,
                    parent_builder: current_builder.take(),
                    slot_idx: i,
                    done,
                    original_dtype: child.dtype().clone(),
                    original_len: child.len(),
                });
                current_array = child;
                current_builder = None;
            }
            ExecutionStep::AppendChild(i) => {
                if current_builder.is_none() {
                    current_builder = Some(builder_with_capacity_in(
                        ctx.allocator(),
                        array.dtype(),
                        array.len(),
                    ));
                }
                let (parent, child) = unsafe { array.take_slot_unchecked(i) }?;
                ctx.log(format_args!(
                    "AppendChild({i}): appending {} into builder",
                    child
                ));
                // TODO(perf): replace with a builder kernel registry so we don't
                // need to go through the VTable append_to_builder indirection.
                child.append_to_builder(
                    current_builder
                        .as_deref_mut()
                        .vortex_expect("builder must exist"),
                    ctx,
                )?;
                current_array = parent;
            }
            ExecutionStep::Done => {
                ctx.log(format_args!("Done: {}", array));
                (current_array, current_builder) = finalize_done(
                    array,
                    current_builder,
                    expected_len,
                    expected_dtype,
                    stats,
                    encoding_id,
                )?;
            }
        }
    }

    vortex_bail!(
        "Exceeded maximum execution iterations ({}) while executing array",
        max_iterations,
    )
}

/// Pop a stack frame, restoring the parent with the finished child in its slot.
fn pop_frame(
    frame: StackFrame,
    child: ArrayRef,
) -> VortexResult<(ArrayRef, Option<Box<dyn ArrayBuilder>>)> {
    debug_assert_eq!(
        child.dtype(),
        &frame.original_dtype,
        "child dtype changed during execution"
    );
    debug_assert_eq!(
        child.len(),
        frame.original_len,
        "child len changed during execution"
    );
    let parent_array = unsafe { frame.parent_array.put_slot_unchecked(frame.slot_idx, child) }?;
    Ok((parent_array, frame.parent_builder))
}

/// Execute a single step on an array, consuming it.
///
/// Extracts the vtable before consuming the array to avoid borrow conflicts.
fn execute_step_checked(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
    array.execute_encoding(ctx)
}

fn execute_step_unchecked(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    array.execute_encoding_unchecked(ctx)
}

fn finalize_done(
    result: ArrayRef,
    mut builder: Option<Box<dyn ArrayBuilder>>,
    expected_len: usize,
    expected_dtype: DType,
    stats: ArrayStats,
    encoding_id: ArrayId,
) -> VortexResult<(ArrayRef, Option<Box<dyn ArrayBuilder>>)> {
    let output = if let Some(mut builder) = builder.take() {
        builder.finish()
    } else {
        result
    };

    if cfg!(debug_assertions) {
        vortex_ensure!(
            output.len() == expected_len,
            "Result length mismatch for {:?}",
            encoding_id
        );
        vortex_ensure!(
            output.dtype() == &expected_dtype,
            "Executed canonical dtype mismatch for {:?}",
            encoding_id
        );
    }

    output
        .statistics()
        .set_iter(StatsSet::from(stats).into_iter());
    Ok((output, None))
}

/// Try execute_parent on each occupied slot of the array.
fn try_execute_parent(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<ArrayRef>> {
    for (slot_idx, slot) in array.slots().iter().enumerate() {
        let Some(child) = slot else {
            continue;
        };
        if let Some(result) = child.execute_parent(array, slot_idx, ctx)? {
            result.statistics().inherit_from(array.statistics());
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// A predicate that determines when an array has reached a desired form during execution.
pub type DonePredicate = fn(&ArrayRef) -> bool;

/// Metadata-only step indicator returned alongside an array in [`ExecutionResult`].
///
/// Instead of recursively executing children, encodings return an `ExecutionStep` that tells the
/// scheduler what to do next. This enables the scheduler to manage execution iteratively using
/// an explicit work stack, run cross-step optimizations, and cache shared sub-expressions.
///
/// # Semantics
///
/// Each variant describes a different execution strategy with distinct cost profiles:
///
/// - [`Done`](ExecutionStep::Done): The encoding has finished its work in this step. The
///   returned array is the result. The scheduler may continue executing if the target form
///   (e.g. canonical) has not yet been reached.
///
/// - [`ExecuteSlot`](ExecutionStep::ExecuteSlot): The encoding needs one of its children
///   decoded before it can make further progress. The scheduler takes ownership of the child,
///   executes it until the [`DonePredicate`] matches, puts it back, and re-enters the parent.
///   Between steps the optimizer runs reduce/reduce_parent rules to fixpoint, enabling
///   cross-step optimization (e.g. pushing scalar functions through newly-decoded children).
///   This is a cooperative yield — the encoding does a bounded amount of work per step.
///
/// - [`AppendChild`](ExecutionStep::AppendChild): The encoding needs one child executed to
///   canonical form and then appended into a builder owned by the current activation. The
///   scheduler suspends the parent, executes the child, appends the finished child into the
///   parent's builder, and then resumes the same parent so it can continue with more
///   `AppendChild` or `ExecuteSlot` steps. **Important:** in the single-step executor
///   ([`Executable`] for [`ArrayRef`]), returning `AppendChild` still causes the executor to
///   drive the *entire* array to completion via [`execute_into_builder`] in one call — this can
///   do significantly more work than a single `ExecuteSlot` step.
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

    /// Execute the slot at the given index to canonical form, then append it into a canonical
    /// builder owned by the current activation.
    ///
    /// The parent activation remains suspended with its builder while the child executes. Once
    /// the child reaches canonical form, the scheduler appends it into the parent builder and
    /// resumes the same parent activation.
    ///
    /// **Note:** In the single-step executor ([`Executable`] for [`ArrayRef`]), this variant
    /// drives the entire parent to completion in one call via [`execute_into_builder`], which
    /// may perform substantially more work than a single `ExecuteSlot` step.
    AppendChild(usize),

    /// Execution is complete. The array in the accompanying [`ExecutionResult`] is the result.
    /// The scheduler will continue executing if it has not yet reached the target form.
    Done,
}

impl fmt::Debug for ExecutionStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionStep::ExecuteSlot(idx, _) => f.debug_tuple("ExecuteSlot").field(idx).finish(),
            ExecutionStep::AppendChild(idx) => f.debug_tuple("AppendChild").field(idx).finish(),
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

    /// Request that the child slot at `slot_idx` be executed and appended into the current
    /// activation's canonical builder.
    pub fn append_child(array: impl IntoArray, slot_idx: usize) -> Self {
        Self {
            array: array.into_array(),
            step: ExecutionStep::AppendChild(slot_idx),
        }
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
/// `$parent` - it is moved into the early-return path.
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
