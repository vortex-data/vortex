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
use std::sync::atomic::AtomicUsize;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::matcher::Matcher;
use crate::memory::HostAllocatorRef;
use crate::memory::MemorySessionExt;
use crate::optimizer::ArrayOptimizer;

/// Maximum number of iterations to attempt when executing an array before giving up and returning
/// an error.
pub(crate) static MAX_ITERATIONS: LazyLock<usize> =
    LazyLock::new(|| match std::env::var("VORTEX_MAX_ITERATIONS") {
        Ok(val) => val
            .parse::<usize>()
            .unwrap_or_else(|e| vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid usize: {e}")),
        Err(VarError::NotPresent) => 128,
        Err(VarError::NotUnicode(_)) => {
            vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid unicode string")
        }
    });

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
    /// The scheduler repeatedly:
    /// 1. Checks if the current array matches `M` — if so, pops the stack or returns.
    /// 2. Runs `execute_parent` on each child for child-driven optimizations.
    /// 3. Calls `execute` which returns an [`ExecutionStep`].
    ///
    /// Note: the returned array may not match `M`. If execution converges to a canonical form
    /// that does not match `M`, the canonical array is returned since no further execution
    /// progress is possible.
    ///
    /// For safety, we will error when the number of execution iterations reaches a configurable
    /// maximum (default 128, override with `VORTEX_MAX_ITERATIONS`).
    pub fn execute_until<M: Matcher>(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        static MAX_ITERATIONS: LazyLock<usize> =
            LazyLock::new(|| match std::env::var("VORTEX_MAX_ITERATIONS") {
                Ok(val) => val.parse::<usize>().unwrap_or_else(|e| {
                    vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid usize: {e}")
                }),
                Err(VarError::NotPresent) => 128,
                Err(VarError::NotUnicode(_)) => {
                    vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid unicode string")
                }
            });

        let mut current = self.optimize()?;
        // Stack frames: (parent, slot_idx, done_predicate_for_slot)
        let mut stack: Vec<(ArrayRef, usize, DonePredicate)> = Vec::new();

        for _ in 0..*MAX_ITERATIONS {
            // Check for termination: use the stack frame's done predicate, or the root matcher.
            let is_done = stack
                .last()
                .map_or(M::matches as DonePredicate, |frame| frame.2);
            if is_done(&current) {
                match stack.pop() {
                    None => {
                        ctx.log(format_args!("-> {}", current));
                        return Ok(current);
                    }
                    Some((parent, slot_idx, _)) => {
                        current = parent.with_slot(slot_idx, current)?;
                        current = current.optimize()?;
                        continue;
                    }
                }
            }

            // If we've reached canonical form, we can't execute any further regardless
            // of whether the matcher matched.
            if AnyCanonical::matches(&current) {
                match stack.pop() {
                    None => {
                        ctx.log(format_args!("-> canonical (unmatched) {}", current));
                        return Ok(current);
                    }
                    Some((parent, slot_idx, _)) => {
                        current = parent.with_slot(slot_idx, current)?;
                        current = current.optimize()?;
                        continue;
                    }
                }
            }

            // Try execute_parent (child-driven optimized execution)
            if let Some(rewritten) = try_execute_parent(&current, ctx)? {
                ctx.log(format_args!(
                    "execute_parent rewrote {} -> {}",
                    current, rewritten
                ));
                current = rewritten.optimize()?;
                continue;
            }

            // Execute the array itself.
            let result = execute_step(current, ctx)?;
            let (array, step) = result.into_parts();
            match step {
                ExecutionStep::ExecuteSlot(i, done) => {
                    let child = array.slots()[i]
                        .clone()
                        .vortex_expect("ExecuteSlot index in bounds");
                    ctx.log(format_args!(
                        "ExecuteSlot({i}): pushing {}, focusing on {}",
                        array, child
                    ));
                    stack.push((array, i, done));
                    current = child.optimize()?;
                }
                ExecutionStep::Done => {
                    ctx.log(format_args!("Done: {}", array));
                    current = array;
                }
            }
        }

        vortex_bail!(
            "Exceeded maximum execution iterations ({}) while executing array",
            *MAX_ITERATIONS,
        )
    }
}

/// Execution context for batch CPU compute.
///
/// Accumulates a trace of execution steps. Individual steps are logged at TRACE level for
/// real-time following, and the full trace is dumped at DEBUG level when the context is dropped.
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
            id,
            session,
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
/// The execution steps are as follows:
/// 0. Check for canonical.
/// 1. Attempt to `reduce` the array with metadata-only optimizations.
/// 2. Attempt to call `reduce_parent` on each child.
/// 3. Attempt to call `execute_parent` on each child.
/// 4. Call `execute` on the array itself (which returns an [`ExecutionStep`]).
///
/// Most users will not call this method directly, instead preferring to specify an executable
/// target such as [`crate::Columnar`], [`Canonical`], or any of the canonical array types (such as
/// [`crate::arrays::PrimitiveArray`]).
impl Executable for ArrayRef {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // 0. Check for canonical
        if let Some(canonical) = array.as_opt::<AnyCanonical>() {
            ctx.log(format_args!("-> canonical {}", array));
            return Ok(Canonical::from(canonical).into_array());
        }

        // 1. reduce (metadata-only rewrites)
        if let Some(reduced) = array.reduce()? {
            ctx.log(format_args!("reduce: rewrote {} -> {}", array, reduced));
            reduced.statistics().inherit_from(array.statistics());
            return Ok(reduced);
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
                return Ok(reduced_parent);
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
                return Ok(executed_parent);
            }
        }

        // 4. execute (returns an ExecutionResult)
        ctx.log(format_args!("executing {}", array));
        let result = execute_step(array, ctx)?;
        let (array, step) = result.into_parts();
        match step {
            ExecutionStep::Done => {
                ctx.log(format_args!("-> {}", array));
                Ok(array)
            }
            ExecutionStep::ExecuteSlot(i, _) => {
                // For single-step execution, handle ExecuteSlot by executing the slot,
                // replacing it, and returning the updated array.
                let child = array.slots()[i].clone().vortex_expect("valid slot index");
                let executed_child = child.execute::<ArrayRef>(ctx)?;
                array.with_slot(i, executed_child)
            }
        }
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

    /// Execution is complete. The array in the accompanying [`ExecutionResult`] is the result.
    /// The scheduler will continue executing if it has not yet reached the target form.
    Done,
}

impl fmt::Debug for ExecutionStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionStep::ExecuteSlot(idx, _) => f.debug_tuple("ExecuteSlot").field(idx).finish(),
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
