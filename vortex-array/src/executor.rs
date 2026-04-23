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
use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::BuilderKernelSession;
use crate::BuilderStep;
use crate::Canonical;
use crate::IntoArray;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity_in;
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
            Err(VarError::NotPresent) => 8_192,
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
    /// The unified loop handles both normal execution (`ExecuteSlot`) and builder-driven
    /// execution (`AppendChild`). Builder state is threaded via `current_builder`; when an
    /// `ExecuteSlot` is pushed from builder mode, the builder is stashed on the frame and
    /// restored on pop.
    ///
    /// Note: the returned array may not match `M`. If execution converges to a canonical form
    /// that does not match `M`, the canonical array is returned since no further execution
    /// progress is possible.
    ///
    /// For safety, we will error when the number of execution iterations reaches a configurable
    /// maximum (default 128, override with `VORTEX_MAX_ITERATIONS`).
    pub fn execute_until<M: Matcher>(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        match execute_loop(self, None, M::matches, ctx)? {
            ExecuteLoopResult::Array(a) => Ok(a),
            ExecuteLoopResult::Builder(mut b) => Ok(b.finish()),
        }
    }
}

/// A stack frame for the unified iterative executor.
///
/// Each frame records a parent whose slot was taken. The `builder` field distinguishes the
/// frame's role:
///
/// - **`builder = None`**: the child was taken for either `ExecuteSlot` (normal mode) or
///   `AppendChild` (builder mode). On pop, the distinction comes from `current_builder`:
///   if `Some`, the child was appended into the builder and `current` becomes the parent
///   (slot stays `None`). If `None`, the child is put back via `put_slot_unchecked`.
///
/// - **`builder = Some(stashed)`**: an `ExecuteSlot` was pushed while in builder mode. The
///   builder was stashed here so the child executes in normal mode. On pop, the child is
///   put back and the stashed builder is restored to `current_builder`.
struct StackFrame {
    parent: ArrayRef,
    slot_idx: usize,
    done: DonePredicate,
    /// Stashed builder from an `ExecuteSlot` pushed while in builder mode.
    builder: Option<Box<dyn ArrayBuilder>>,
    original_dtype: DType,
    original_len: usize,
}

/// The result of the unified execution loop.
enum ExecuteLoopResult {
    Array(ArrayRef),
    Builder(Box<dyn ArrayBuilder>),
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
/// Steps through reduce, reduce_parent, execute_parent, then execute. For `AppendChild`,
/// drives the builder path to completion in one call (no useful partial progress point).
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
        let result = execute_step(array, ctx)?;
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
                // Single-step: build the entire parent via the builder-kernel path.
                let builder = builder_with_capacity_in(ctx.allocator(), array.dtype(), array.len());
                let mut builder = execute_into_builder(array, builder, ctx)?;
                Ok(builder.finish())
            }
        }
    }
}

/// Execute `array` into the given `builder` via the builder-kernel path.
///
/// Drives `array` toward canonical form, using builder kernels when available and falling back
/// to normal execution otherwise. The builder is threaded through the execution: children
/// appended via `AppendChild` share the same builder, while `ExecuteSlot` children are executed
/// normally with the builder stashed on the stack frame.
///
/// The builder must have a [`DType`] that is a nullability-superset of `array.dtype()`.
pub fn execute_into_builder(
    array: ArrayRef,
    builder: Box<dyn ArrayBuilder>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ArrayBuilder>> {
    match execute_loop(array, Some(builder), AnyCanonical::matches, ctx)? {
        ExecuteLoopResult::Builder(b) => Ok(b),
        ExecuteLoopResult::Array(_) => {
            vortex_panic!("execute_into_builder started with a builder but got Array result")
        }
    }
}

/// Unified execution loop. Handles both normal execution and builder-driven execution in a
/// single stack.
///
/// `initial_builder`:
/// - `None`: normal execution mode, returns `Array` when done.
/// - `Some(builder)`: builder mode, returns `Builder` when the array's data has been fully
///   appended.
///
/// The loop dispatches to builder kernels when `current_builder` is `Some`, falling back to
/// `execute_step` when no kernel is registered. `ExecuteSlot` frames stash the builder so
/// children execute in normal mode; on pop the builder is restored.
fn execute_loop(
    array: ArrayRef,
    initial_builder: Option<Box<dyn ArrayBuilder>>,
    root_done: DonePredicate,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecuteLoopResult> {
    let mut current = array.optimize()?;
    let mut current_builder = initial_builder;
    let mut stack: Vec<StackFrame> = Vec::new();

    for _ in 0..max_iterations() {
        // ── Step 1: done / canonical check ─────────────────────────────────
        let is_done = if current_builder.is_some() {
            AnyCanonical::matches as DonePredicate
        } else {
            stack.last().map_or(root_done, |frame| frame.done)
        };

        if is_done(&current) || AnyCanonical::matches(&current) {
            // In builder mode, extend the builder from the canonical array.
            if AnyCanonical::matches(&current)
                && let Some(ref mut b) = current_builder
            {
                ctx.log(format_args!("extend builder from canonical {}", current));
                b.extend_from_array(&current);
            }

            match stack.pop() {
                None => {
                    if let Some(b) = current_builder.take() {
                        return Ok(ExecuteLoopResult::Builder(b));
                    }
                    ctx.log(format_args!("-> {}", current));
                    return Ok(ExecuteLoopResult::Array(current));
                }
                Some(frame) => {
                    current = pop_frame(frame, current, &mut current_builder, ctx)?;
                    continue;
                }
            }
        }

        // ── Step 2: builder kernel dispatch (builder mode only) ────────────
        if current_builder.is_some() {
            let kernel = {
                let session = ctx.session().clone();
                let builder_kernels = session.get::<BuilderKernelSession>();
                builder_kernels.find(&current.encoding_id())
            };

            if let Some(kernel) = kernel {
                let builder = current_builder
                    .take()
                    .vortex_expect("current_builder must be Some in builder mode");
                let result = kernel.append_to_builder(current, builder, ctx)?;
                let (array_after, builder_after, step) = result.into_parts();
                current_builder = Some(builder_after);

                match step {
                    BuilderStep::Done => {
                        ctx.log(format_args!("builder kernel Done for {}", array_after));
                        match stack.pop() {
                            None => {
                                return Ok(ExecuteLoopResult::Builder(
                                    current_builder
                                        .take()
                                        .vortex_expect("builder must be Some after kernel Done"),
                                ));
                            }
                            Some(frame) => {
                                current = pop_frame(frame, array_after, &mut current_builder, ctx)?;
                                continue;
                            }
                        }
                    }
                    BuilderStep::ExecuteSlot(i, done) => {
                        ctx.log(format_args!("builder ExecuteSlot({i}) for {}", array_after));
                        let (parent, child) = unsafe { array_after.take_slot_unchecked(i) }?;
                        stack.push(StackFrame {
                            parent,
                            slot_idx: i,
                            done,
                            builder: None,
                            original_dtype: child.dtype().clone(),
                            original_len: child.len(),
                        });
                        // Don't optimize — parent already has None slots, and the
                        // child will be checked by the canonical/done check next iteration.
                        current = child;
                    }
                }
                continue;
            }

            // No builder kernel — fall through to execute_step below.
        }

        // ── Step 3: execute_parent (normal mode only) ──────────────────────
        if current_builder.is_none()
            && let Some(rewritten) = try_execute_parent(&current, ctx)?
        {
            ctx.log(format_args!(
                "execute_parent rewrote {} -> {}",
                current, rewritten
            ));
            current = rewritten.optimize()?;
            if let Some(frame) = stack.pop() {
                current = pop_frame(frame, current, &mut current_builder, ctx)?;
            }
            continue;
        }

        // ── Step 4: execute step ───────────────────────────────────────────
        let result = execute_step(current, ctx)?;
        let (array, step) = result.into_parts();
        match step {
            ExecutionStep::ExecuteSlot(i, done) => {
                let (parent, child) = unsafe { array.take_slot_unchecked(i) }?;
                ctx.log(format_args!(
                    "ExecuteSlot({i}): pushing {}, focusing on {}",
                    parent, child
                ));
                stack.push(StackFrame {
                    parent,
                    slot_idx: i,
                    done,
                    builder: current_builder.take(), // stash builder if in builder mode
                    original_dtype: child.dtype().clone(),
                    original_len: child.len(),
                });
                current = child.optimize()?;
            }
            ExecutionStep::AppendChild(i) => {
                let (parent, child) = unsafe { array.take_slot_unchecked(i) }?;
                ctx.log(format_args!("AppendChild({i}): focusing on {}", child));
                // Builder is for the parent — all children append into it.
                if current_builder.is_none() {
                    current_builder = Some(builder_with_capacity_in(
                        ctx.allocator(),
                        parent.dtype(),
                        parent.len(),
                    ));
                }
                stack.push(StackFrame {
                    parent,
                    slot_idx: i,
                    done: AnyCanonical::matches,
                    builder: None, // builder stays on current_builder
                    original_dtype: child.dtype().clone(),
                    original_len: child.len(),
                });
                current = child;
            }
            ExecutionStep::Done => {
                ctx.log(format_args!("Done: {}", array));
                current = array;
            }
        }
    }

    vortex_bail!(
        "Exceeded maximum execution iterations ({}) while executing array",
        max_iterations(),
    )
}

/// Pop a stack frame, updating `current` and `current_builder` accordingly.
fn pop_frame(
    frame: StackFrame,
    current: ArrayRef,
    current_builder: &mut Option<Box<dyn ArrayBuilder>>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(stashed) = frame.builder {
        // ExecuteSlot with stashed builder: put child back, restore builder.
        debug_assert_eq!(
            current.dtype(),
            &frame.original_dtype,
            "slot {} dtype changed during execution",
            frame.slot_idx
        );
        debug_assert_eq!(
            current.len(),
            frame.original_len,
            "slot {} len changed during execution",
            frame.slot_idx
        );
        *current_builder = Some(stashed);
        unsafe { frame.parent.put_slot_unchecked(frame.slot_idx, current) }?.optimize()
    } else if current_builder.is_some() {
        // AppendChild: builder already extended from canonical. Return parent (slot stays None).
        Ok(frame.parent)
    } else {
        // Normal ExecuteSlot: put child back.
        debug_assert_eq!(
            current.dtype(),
            &frame.original_dtype,
            "slot {} dtype changed during execution",
            frame.slot_idx
        );
        debug_assert_eq!(
            current.len(),
            frame.original_len,
            "slot {} len changed during execution",
            frame.slot_idx
        );
        unsafe { frame.parent.put_slot_unchecked(frame.slot_idx, current) }?.optimize()
    }
}

/// Execute a single step on an array, consuming it.
///
/// Extracts the vtable before consuming the array to avoid borrow conflicts.
fn execute_step(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
    array.execute_encoding(ctx)
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

    /// Enter builder mode for the child at this slot index.
    ///
    /// The scheduler creates a builder (if one does not already exist), takes the child,
    /// and drives it toward canonical via builder kernels. When the child is canonical the
    /// scheduler extends the builder and nulls the slot. The encoding must have a builder
    /// kernel registered so the scheduler can re-enter it after each child is consumed.
    ///
    /// When the builder kernel returns [`crate::BuilderStep::Done`], the scheduler finishes
    /// the builder and uses the resulting canonical array as the execution result.
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

    /// Enter builder mode for the child at `slot_idx`.
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
