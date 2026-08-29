// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct packed Boolean collection for infallible and deferred row computations.
//!
//! These executors share the indexed input decoding used by owned outputs, but construct a
//! canonical [`BoolArray`] without first collecting one byte per row.

use std::mem::size_of;

use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

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
        vortex_bail!("a decoded row input does not address exactly {row_count} rows");
    };

    let collect = |index| {
        // SAFETY: Both collectors only invoke this closure with `index < row_count`, and the decoded
        // source was constructed with exactly `row_count` rows.
        apply(unsafe { source.get_unchecked(index) })
    };
    let values = if MULTIVERSIONED {
        BitBuffer::collect_bool_multiversioned(row_count, collect)
    } else {
        BitBuffer::collect_bool(row_count, collect)
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
        vortex_bail!("a decoded row input does not address exactly {row_count} rows");
    };

    let mut failure = Fail::default();
    let collect = |index| {
        // SAFETY: Both collectors only invoke this closure with `index < row_count`, and the decoded
        // source was constructed with exactly `row_count` rows.
        let elements = unsafe { source.get_unchecked(index) };
        let (value, row_failure) = apply(&prepared, elements);
        failure |= row_failure;

        value
    };
    let values = if MULTIVERSIONED {
        BitBuffer::collect_bool_multiversioned(row_count, collect)
    } else {
        BitBuffer::collect_bool(row_count, collect)
    };

    finish_failure(failure)?;

    Ok(BoolArray::new(values, Validity::NonNullable).into_array())
}
