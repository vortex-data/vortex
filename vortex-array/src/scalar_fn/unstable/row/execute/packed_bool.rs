// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct packed Boolean collection for infallible and deferred row computations.
//!
//! Deferred owned execution buffers a Boolean value per row before packing the finished column.
//! These executors combine predicate evaluation, failure reduction, and bit packing in one loop,
//! avoiding that temporary buffer and the later packing pass. Infallible owned execution already
//! supports direct packing through its output type, but these entry points also support runtime
//! CPU selection and constructing the array with batch validity attached.
//!
//! The infallible collectors build [`BoolArray`] themselves, so they can accept input-derived
//! validity without an all-valid intermediate array or a separate masking step. Generic output
//! sinks and element buffers must return all-valid storage because batch execution cannot assume
//! that an arbitrary output encoding attached the input validity correctly.
//!
//! [`execute_owned_bool`] reports rejected failure evidence as a terminal error.
//! [`execute_bool_dense_attempt`] hands it back as a [`DenseAttempt::DeferredError`] so that batch
//! execution can retry over valid rows. These executors keep separate loops because the retry
//! path's capture layout affects vectorization, as [`DeferredBoolState`] describes. Extracting
//! shared collector state can change that layout even when the helper is inlined.

use std::mem::size_of;

use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::DenseAttempt;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::batch::RowInputs;
use crate::scalar_fn::unstable::row::types::decode_row_inputs;
use crate::scalar_fn::unstable::row::types::decoded_source;
use crate::validity::Validity;

/// Decode every input column, then pack infallible Boolean outputs with `validity` attached.
///
/// The caller supplies the final row validity. Attaching it during construction preserves lazy
/// validity and avoids rebuilding the array for masking. `apply` still evaluates every payload,
/// so the caller must establish that null-row evaluation is permitted.
pub(crate) fn execute_owned_infallible_bool<Args, const MULTIVERSIONED: bool>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    validity: Validity,
    apply: impl Fn(Args::Elems<'_>) -> bool,
) -> VortexResult<ArrayRef>
where
    Args: IndexedElementTuple,
{
    let columns = Args::decode(args, ctx)?;

    pack_infallible_bool::<Args, MULTIVERSIONED>(&columns, args.row_count(), ctx, validity, apply)
}

/// Decode one borrowed array per argument, then pack infallible Boolean outputs with `validity`
/// attached.
///
/// `const_mask` marks the batch-constant inputs, as [`decode_row_inputs`] describes.
/// Reusing that classification over borrowed inputs avoids collecting owned array references and
/// repeating constant detection before the packing loop.
// Keep decoding and packing out of the dispatch body. Inlining this executor duplicates that work
// for each primitive type and comparison operator, increasing the dispatch's generated code size.
// The helpers still inline within this executor so decoding and traversal can optimize together.
#[inline(never)]
pub(crate) fn execute_row_inputs_infallible_bool<Args, I, const MULTIVERSIONED: bool>(
    inputs: &I,
    const_mask: u32,
    ctx: &mut ExecutionCtx,
    validity: Validity,
    apply: impl Fn(Args::Elems<'_>) -> bool,
) -> VortexResult<ArrayRef>
where
    Args: IndexedElementTuple,
    I: RowInputs + ?Sized,
{
    let columns = decode_row_inputs::<Args, I>(inputs, const_mask, ctx)?;

    pack_infallible_bool::<Args, MULTIVERSIONED>(&columns, inputs.row_count(), ctx, validity, apply)
}

/// Pack `apply` over the `row_count` rows of `columns`, then attach `validity`.
///
/// This fails when a column or `validity` does not have exactly `row_count` rows.
#[expect(clippy::inline_always)]
#[inline(always)]
fn pack_infallible_bool<Args, const MULTIVERSIONED: bool>(
    columns: &Args::Columns,
    row_count: usize,
    ctx: &mut ExecutionCtx,
    validity: Validity,
    apply: impl Fn(Args::Elems<'_>) -> bool,
) -> VortexResult<ArrayRef>
where
    Args: IndexedElementTuple,
{
    // The indexed source keeps each argument's constant choice visible to LLVM, allowing it to
    // move those checks outside the row loop. A separate no-constant path would duplicate decoding
    // and packing. Recheck generated loops for array pairs and both constant orientations before
    // introducing that split.
    let Some(source) = decoded_source::<Args>(columns, row_count) else {
        vortex_bail!("a decoded row input does not address exactly {row_count} rows");
    };

    // SAFETY: `decoded_source` validated that this source addresses exactly `row_count` rows.
    let values =
        unsafe { collect_source_bits::<MULTIVERSIONED, _>(&source, row_count, ctx, &apply) };

    Ok(BoolArray::try_new(values, validity)?.into_array())
}

/// Pack `apply` over the `row_count` rows of `source`.
///
/// # Safety
///
/// `source` must address at least `row_count` rows throughout collection. Violating this requirement
/// can cause undefined behavior through an out-of-bounds read.
#[expect(clippy::inline_always)]
#[inline(always)]
unsafe fn collect_source_bits<const MULTIVERSIONED: bool, S: IndexedSource>(
    source: &S,
    row_count: usize,
    ctx: &ExecutionCtx,
    apply: &impl Fn(S::Item) -> bool,
) -> BitBuffer {
    let collect = |index| {
        // SAFETY: Both collectors only invoke this closure with `index < row_count`, and the
        // caller guarantees that the source addresses at least that many rows.
        apply(unsafe { source.get_unchecked(index) })
    };

    if MULTIVERSIONED {
        BitBuffer::collect_bool_multiversioned_in(row_count, collect, ctx.allocator().clone())
    } else {
        BitBuffer::collect_bool_in(row_count, collect, ctx.allocator().clone())
    }
}

/// Decode every input column, then pack Boolean outputs while combining failure evidence.
pub(crate) fn execute_owned_bool<Args, Prepared, Fail, const MULTIVERSIONED: bool>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> (bool, Fail),
    finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
) -> VortexResult<ArrayRef>
where
    Args: IndexedElementTuple,
    Fail: FailureEvidence,
{
    const {
        assert!(
            size_of::<Fail>() <= size_of::<bool>(),
            "failure evidence must be no wider than the value, or it bounds the vector width"
        )
    };

    let columns = Args::decode(args, ctx)?;
    let prepared = prepare(Args::const_values(&columns));
    let row_count = args.row_count();

    let Some(source) = decoded_source::<Args>(&columns, row_count) else {
        vortex_bail!("a decoded row input does not address exactly {row_count} rows");
    };

    let mut failure = Fail::default();
    let collect = |index| {
        // SAFETY: Both collectors only invoke this closure with `index < row_count`, and the
        // decoded source was constructed with exactly `row_count` rows.
        let elements = unsafe { source.get_unchecked(index) };
        let (value, row_failure) = apply(&prepared, elements);
        failure |= row_failure;

        value
    };

    let values = if MULTIVERSIONED {
        BitBuffer::collect_bool_multiversioned_in(row_count, collect, ctx.allocator().clone())
    } else {
        BitBuffer::collect_bool_in(row_count, collect, ctx.allocator().clone())
    };

    finish_failure(failure)?;

    Ok(BoolArray::new(values, Validity::NonNullable).into_array())
}

/// Pack a dense Boolean attempt and report rejected failure evidence to the batch executor.
///
/// Returns the same terminal-versus-deferred split as
/// [`execute_owned_dense_attempt`](super::execute_owned_dense_attempt), which owns that contract.
pub(crate) fn execute_bool_dense_attempt<Args, Prepared, Fail, const MULTIVERSIONED: bool>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> (bool, Fail),
    finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
) -> VortexResult<DenseAttempt>
where
    Args: IndexedElementTuple,
    Fail: FailureEvidence,
{
    const {
        assert!(
            size_of::<Fail>() <= size_of::<bool>(),
            "failure evidence must be no wider than the value, or it bounds the vector width"
        )
    };

    // Keep this dense row loop separate from `execute_owned_bool`. Factoring their shared state
    // into a helper changes LLVM's optimized dense kernel even when the helper is inlined.
    let columns = Args::decode(args, ctx)?;
    let prepared = prepare(Args::const_values(&columns));
    let row_count = args.row_count();

    let Some(source) = decoded_source::<Args>(&columns, row_count) else {
        vortex_bail!("a decoded row input does not address exactly {row_count} rows");
    };

    // NB: The collector must capture one borrow of the whole state. That shape is load-bearing for
    // code generation, not a borrow checker workaround. See the `DeferredBoolState` documentation.
    let mut state = DeferredBoolState {
        source,
        prepared,
        apply,
        failure: Fail::default(),
    };
    let state_ref = &mut state;

    let collect = move |index| {
        let state = &mut *state_ref;

        // SAFETY: Both collectors only invoke this closure with `index < row_count`, and the
        // decoded source was constructed with exactly `row_count` rows.
        let elements = unsafe { state.source.get_unchecked(index) };
        let (value, row_failure) = (state.apply)(&state.prepared, elements);
        state.failure |= row_failure;

        value
    };

    let values = if MULTIVERSIONED {
        BitBuffer::collect_bool_multiversioned_in(row_count, collect, ctx.allocator().clone())
    } else {
        BitBuffer::collect_bool_in(row_count, collect, ctx.allocator().clone())
    };

    match finish_failure(state.failure) {
        Ok(()) => Ok(DenseAttempt::Values(
            BoolArray::new(values, Validity::NonNullable).into_array(),
        )),
        Err(error) => Ok(DenseAttempt::DeferredError(error)),
    }
}

/// Row-loop state for a deferred Boolean computation: the decoded input, the prepared batch
/// constants, the per-row callback, and the accumulated failure evidence.
///
/// The collector closure captures this struct through one mutable borrow, giving the out-of-line
/// collector one exclusive pointer to its state. With multiple codegen units and no LTO, separate
/// captures can lose that aliasing information and leave the multiversioned word loop scalar.
///
/// Preserve the whole-state capture when changing this loop. Recheck optimized IR and the
/// `row_fn_bool_retry` benchmark under the bench profile before splitting the captured state.
struct DeferredBoolState<Source, Prepared, Apply, Fail> {
    source: Source,
    prepared: Prepared,
    apply: Apply,
    failure: Fail,
}
