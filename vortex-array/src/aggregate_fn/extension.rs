// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Storage dispatch and logical partials for aggregates over extension values.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::dtype::Claim;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// The storage dtype of an extension input that holds `claim`.
pub(crate) fn claimed_storage_dtype(input: &DType, claim: Claim) -> Option<&DType> {
    let ext = input.as_extension_opt()?;
    input.holds(claim).then(|| ext.storage_dtype())
}

/// The innermost storage dtype, which the aggregate uses for its in-memory state.
pub(super) fn storage_dtype(mut dtype: &DType) -> &DType {
    while let Some(ext) = dtype.as_extension_opt() {
        dtype = ext.storage_dtype();
    }
    dtype
}

/// Remove extension wrappers when a value enters an aggregate's partial state.
pub(super) fn into_storage_scalar(mut scalar: Scalar) -> Scalar {
    while let Some(ext) = scalar.as_extension_opt() {
        scalar = ext.to_storage_scalar();
    }
    scalar
}

/// Restore the logical dtype of a value selected by the aggregate.
pub(super) fn wrap_storage_scalar(value: &Scalar, dtype: &DType) -> VortexResult<Scalar> {
    if !dtype.is_extension() {
        return value.cast(dtype);
    }

    let storage = storage_dtype(dtype);
    vortex_ensure_eq!(
        storage,
        &value.dtype().with_nullability(storage.nullability())
    );
    Scalar::try_new(dtype.clone(), value.value().cloned())
}

/// Run an aggregate on the storage encoding and fold its in-memory state into the logical call.
pub(super) fn accumulate_storage<A: AggregateFnVTable>(
    aggregate: &A,
    args: AggregateArgs<'_, A::Options>,
    partial: &mut A::Partial,
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mut accumulator = Accumulator::try_new(
        aggregate.clone(),
        args.options.clone(),
        storage.dtype().clone(),
    )?;
    accumulator.accumulate(storage, ctx)?;
    if let Some(storage_partial) = accumulator.take_partial() {
        let running = std::mem::replace(partial, aggregate.empty_partial(args)?);
        *partial = aggregate.merge_partials(args, running, storage_partial)?;
    }
    Ok(())
}
