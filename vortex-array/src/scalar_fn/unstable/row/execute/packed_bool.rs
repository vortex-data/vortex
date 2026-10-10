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
use crate::scalar_fn::unstable::row::types::decoded_source;
use crate::validity::Validity;

/// Decode every input column, then pack infallible Boolean outputs.
pub(crate) fn execute_owned_infallible_bool<Args, const MULTIVERSIONED: bool>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    apply: impl Fn(Args::Elems<'_>) -> bool,
) -> VortexResult<ArrayRef>
where
    Args: IndexedElementTuple,
{
    let columns = Args::decode(args, ctx)?;
    let row_count = args.row_count();

    let Some(source) = decoded_source::<Args>(&columns, row_count) else {
        vortex_bail!(AssertionFailed: "a decoded row input does not address exactly {row_count} rows");
    };

    let collect = |index| {
        // SAFETY: Both collectors only invoke this closure with `index < row_count`, and the decoded
        // source was constructed with exactly `row_count` rows.
        apply(unsafe { source.get_unchecked(index) })
    };
    let values = if MULTIVERSIONED {
        BitBuffer::collect_bool_multiversioned_in(row_count, collect, ctx.allocator().clone())
    } else {
        BitBuffer::collect_bool_in(row_count, collect, ctx.allocator().clone())
    };

    Ok(BoolArray::new(values, Validity::NonNullable).into_array())
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
        vortex_bail!(AssertionFailed: "a decoded row input does not address exactly {row_count} rows");
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
        vortex_bail!(AssertionFailed: "a decoded row input does not address exactly {row_count} rows");
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
