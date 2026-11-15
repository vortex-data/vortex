// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate pushdown into the integer most significant part of [`DecimalByteParts`].
//!
//! With no lower parts, the MSP holds the unscaled decimal values. Integer aggregates can reuse
//! child kernels, but their results must preserve decimal scale, precision, and state nullability.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;

/// Computes decimal partials from the MSP without decoding the DBP batch.
#[derive(Debug)]
pub(crate) struct DecimalBytePartsAggregateKernel;

impl DynAggregateKernel for DecimalBytePartsAggregateKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(array) = batch.as_opt::<DecimalByteParts>() else {
            return Ok(None);
        };

        if !array.lower_parts().is_empty() {
            return Ok(None);
        }

        if let Some(options) = aggregate_fn.as_opt::<IsSorted>() {
            return Ok(Some(is_sorted_partial(
                batch,
                array.msp(),
                options.strict,
                ctx,
            )?));
        }

        if aggregate_fn.is::<Sum>() {
            return sum_partial(aggregate_fn, batch, array.msp(), ctx);
        }

        extrema_partial(aggregate_fn, batch, array.msp(), ctx)
    }
}

fn is_sorted_partial(
    batch: &ArrayRef,
    msp: &ArrayRef,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let sorted = if strict {
        is_strict_sorted(msp, ctx)?
    } else {
        is_sorted(msp, ctx)?
    };

    IsSorted::make_partial(batch, sorted, strict, ctx)
}

/// Returns `None` to decline, so the accumulator falls back to decoding the batch.
fn sum_partial(
    aggregate_fn: &AggregateFnRef,
    batch: &ArrayRef,
    msp: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    // Only the Stat::Sum path sums each batch from zero. Other variants add values to the
    // running sum and can overflow at different points than a batch total.
    if Stat::from_aggregate_fn(aggregate_fn) != Some(Stat::Sum) {
        return Ok(None);
    }

    // TODO(connor): Predict integer overflow from dtypes and batch length using Widen's
    // is_exact_sum bound in a later refinement. For now, retry decimal execution on overflow.
    let total = sum(msp, ctx)?;
    if total.is_null() {
        return Ok(None);
    }

    let dtype = aggregate_fn
        .state_dtype(batch.dtype())
        .vortex_expect("Sum accepts the DBP decimal dtype");
    let value = DecimalValue::from(i64::try_from(&total)?);
    let decimal_dtype = *dtype
        .as_decimal_opt()
        .vortex_expect("the Sum state dtype of a decimal input is a decimal");
    let partial = if value.fits_in_precision(decimal_dtype) {
        Scalar::try_decimal(value, decimal_dtype, dtype.nullability())?
    } else {
        Scalar::null(dtype)
    };

    Ok(Some(partial))
}

/// Returns `None` to decline, so the accumulator falls back to decoding the batch.
fn extrema_partial(
    aggregate_fn: &AggregateFnRef,
    batch: &ArrayRef,
    msp: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    let options = aggregate_fn
        .as_opt::<MinMax>()
        .or_else(|| aggregate_fn.as_opt::<Min>())
        .or_else(|| aggregate_fn.as_opt::<Max>());
    let Some(options) = options else {
        return Ok(None);
    };

    let dtype = aggregate_fn
        .state_dtype(batch.dtype())
        .vortex_expect("extrema aggregates accept the DBP decimal dtype");
    let Some(extrema) = min_max(msp, ctx, *options)? else {
        return Ok(Some(Scalar::null(dtype)));
    };

    let element_dtype = batch.dtype().as_nonnullable();
    let min = relabel(&extrema.min, &element_dtype)?;
    let max = relabel(&extrema.max, &element_dtype)?;
    let partial = if aggregate_fn.is::<MinMax>() {
        Scalar::struct_(dtype, vec![min, max])
    } else if aggregate_fn.is::<Min>() {
        min.into_nullable()
    } else {
        max.into_nullable()
    };

    Ok(Some(partial))
}

/// Preserves the unscaled integer value, unlike an integer-to-decimal scalar cast.
fn relabel(value: &Scalar, dtype: &DType) -> VortexResult<Scalar> {
    Scalar::try_decimal(
        DecimalValue::from(i64::try_from(value)?),
        *dtype
            .as_decimal_opt()
            .vortex_expect("DBP has a decimal dtype"),
        dtype.nullability(),
    )
}

#[cfg(test)]
mod tests;
