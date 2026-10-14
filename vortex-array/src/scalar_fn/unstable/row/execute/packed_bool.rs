// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct packed Boolean collection for infallible and deferred row computations.
//!
//! These executors share the indexed input decoding used by owned outputs, but construct a
//! canonical [`BoolArray`] without first collecting one byte per row.
//!
//! [`execute_owned_bool`] reports rejected failure evidence as a terminal error.
//! [`execute_bool_dense_attempt`] hands it back as a [`DenseAttempt::DeferredError`] so that batch
//! execution can retry over valid rows. The two keep separate copies of the row loop on purpose.

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
use crate::scalar_fn::unstable::row::types::decode_rows;
use crate::scalar_fn::unstable::row::types::decoded_source;
use crate::validity::Validity;

/// Decode every input column, then pack infallible Boolean outputs with `validity` attached.
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
/// Bit `i` of `const_mask` is set exactly when input `i` is batch-constant.
///
/// Decoding and packing inline into this one frame, and this frame stays out of the dispatch
/// that calls it. A short batch then pays for one call boundary, not one per step, and each arm
/// of the dispatch stays small.
#[inline(never)]
pub(crate) fn execute_rows_infallible_bool<Args, I, const MULTIVERSIONED: bool>(
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
    let columns = decode_rows::<Args, I>(inputs, const_mask, ctx)?;

    pack_infallible_bool::<Args, MULTIVERSIONED>(&columns, inputs.row_count(), ctx, validity, apply)
}

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
    // Without batch constants, read plain views. Selecting each argument's constant or row
    // source per row stays in short scalar loops, which LLVM does not unswitch.
    let values = if let Some(views) = Args::views_if_no_consts(columns) {
        if !Args::view_lens_match(&views, row_count) {
            vortex_bail!("a decoded row input does not address exactly {row_count} rows");
        }

        // SAFETY: `view_lens_match` checked that these exact views address `row_count` rows.
        let source = unsafe { Args::indexed_source(views, row_count) };
        collect_source_bits::<MULTIVERSIONED, _>(&source, row_count, ctx, &apply)
    } else {
        let Some(source) = decoded_source::<Args>(columns, row_count) else {
            vortex_bail!("a decoded row input does not address exactly {row_count} rows");
        };
        collect_source_bits::<MULTIVERSIONED, _>(&source, row_count, ctx, &apply)
    };

    debug_assert!(validity.maybe_len().is_none_or(|len| len == row_count));

    // SAFETY: the collector packs exactly `row_count` bits from offset zero, and every batch
    // validity is the conjunction of input validities that each address `row_count` rows.
    Ok(unsafe { BoolArray::new_unchecked(values, validity) }.into_array())
}

/// Pack `apply` over the `row_count` rows of `source`.
#[expect(clippy::inline_always)]
#[inline(always)]
fn collect_source_bits<const MULTIVERSIONED: bool, S: IndexedSource>(
    source: &S,
    row_count: usize,
    ctx: &ExecutionCtx,
    apply: &impl Fn(S::Item) -> bool,
) -> BitBuffer {
    let collect = |index| {
        // SAFETY: Both collectors only invoke this closure with `index < row_count`, and the
        // source addresses exactly `row_count` rows.
        apply(unsafe { source.get_unchecked(index) })
    };

    if MULTIVERSIONED {
        BitBuffer::collect_bool_multiversioned_in(row_count, collect, ctx.allocator().clone())
    } else {
        BitBuffer::collect_bool_in(row_count, collect, ctx.allocator().clone())
    }
}

/// Decode every input column, then pack Boolean outputs with `validity` attached while combining
/// failure evidence.
pub(crate) fn execute_owned_bool<Args, Prepared, Fail, const MULTIVERSIONED: bool>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    validity: Validity,
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

    Ok(BoolArray::try_new(values, validity)?.into_array())
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
/// The collector closure captures this struct through one mutable borrow. It does not capture the
/// four parts separately. The single borrow gives the out-of-line collector one exclusive pointer
/// to its state, and the vectorizer needs that to keep the multiversioned word loop packed.
/// Separate captures lose the same aliasing information, and the loop stays scalar.
///
/// The capture shape is therefore load-bearing for code generation. The measurements come from
/// the `row_fn_bool_retry` benchmark, which builds in the `bench` profile with 16 codegen units
/// and no LTO. Recheck that benchmark and its optimized IR before you change the captures.
struct DeferredBoolState<Source, Prepared, Apply, Fail> {
    source: Source,
    prepared: Prepared,
    apply: Apply,
    failure: Fail,
}
