// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregates over the narrow child of a [`Widen`] array.
//!
//! Results must match aggregation over the widened values. Widening preserves order and equality,
//! but sums also depend on the logical dtype's precision and overflow rules.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateDTypes;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::dtype::DType;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;

/// Computes aggregate partials from a [`Widen`]'s child without materializing widened values.
#[derive(Debug)]
pub(crate) struct WidenArrayAggregate;

impl DynAggregateKernel for WidenArrayAggregate {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(widen) = batch.as_opt::<Widen>() else {
            return Ok(None);
        };

        let values = widen.values();
        let logical_dtype = batch.dtype();

        if logical_dtype.is_decimal() {
            return aggregate_decimal(aggregate_fn, values, logical_dtype, ctx);
        }

        if !logical_dtype.is_int() {
            return Ok(None);
        }

        if aggregate_fn.is::<Sum>() && !is_exact_sum(aggregate_fn, batch, values) {
            return Ok(None);
        }

        let partial = aggregate_values(aggregate_fn, values, logical_dtype, ctx)?;

        Ok(Some(partial))
    }
}

/// Selects decimal aggregates whose child values produce the canonical partial.
fn aggregate_decimal(
    aggregate_fn: &AggregateFnRef,
    values: &ArrayRef,
    logical_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    if let Some(options) = aggregate_fn.as_opt::<Sum>() {
        // This kernel returns a batch total. Only the Stat::Sum path sums each batch from zero.
        // Other variants add to the running sum and can overflow at different points.
        if Stat::from_aggregate_fn(aggregate_fn) != Some(Stat::Sum) {
            return Ok(None);
        }

        let partial = decimal_sum_partial(values, options, logical_dtype, ctx)?;

        return Ok(Some(partial));
    }

    if !(aggregate_fn.is::<MinMax>()
        || aggregate_fn.is::<Min>()
        || aggregate_fn.is::<Max>()
        || aggregate_fn.is::<IsConstant>()
        || aggregate_fn.is::<IsSorted>())
    {
        return Ok(None);
    }

    let partial = aggregate_values(aggregate_fn, values, logical_dtype, ctx)?;

    Ok(Some(partial))
}

/// Computes a child partial and casts it to the logical aggregate state dtype.
fn aggregate_values(
    aggregate_fn: &AggregateFnRef,
    values: &ArrayRef,
    logical_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let partial_dtype = aggregate_fn.state_dtype(logical_dtype).vortex_expect(
        "the served aggregate accepts the logical dtype when it accepts the narrower values",
    );

    if let Some(options) = aggregate_fn.as_opt::<MinMax>() {
        return min_max_partial(values, *options, logical_dtype, partial_dtype, ctx);
    }

    let mut accumulator = aggregate_fn.accumulator(values.dtype())?;
    accumulator.accumulate(values, ctx)?;

    accumulator.flush()?.cast(&partial_dtype)
}

/// Computes a decimal [`Sum`] batch partial without materializing the widened array.
///
/// Checked arithmetic overflow or a batch total outside the result precision produces null.
/// Null rows are skipped, and empty or all-null batches produce zero.
fn decimal_sum_partial(
    values: &ArrayRef,
    options: &NumericalAggregateOpts,
    logical_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    // Sum converts each stored value to the accumulator width selected by these logical dtypes
    // before adding it. The child width never limits the sum. Widen preserves values, scale, and
    // validity, so the additions, precision checks, and output dtype match the widened array.
    let dtypes = AggregateDTypes::try_new(&Sum, options, logical_dtype.clone())?;
    let args = dtypes.args(options);
    let values = values.clone().execute::<Columnar>(ctx)?;

    let mut partial = Sum.empty_partial(args)?;
    Sum.accumulate(args, &mut partial, &values, ctx)?;

    Sum.to_scalar(args, &partial)
}

/// Builds a logical [`MinMax`] partial from the child's cached or computed extrema.
///
/// [`min_max`] uses exact cached extrema before scanning the child. Extrema have the non-nullable
/// logical dtype, even for nullable inputs. A batch with no valid values has a null partial.
fn min_max_partial(
    values: &ArrayRef,
    options: NumericalAggregateOpts,
    logical_dtype: &DType,
    partial_dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let Some(MinMaxResult { min, max }) = min_max(values, ctx, options)? else {
        return Ok(Scalar::null(partial_dtype));
    };

    let element_dtype = logical_dtype.as_nonnullable();
    let extrema = vec![min.cast(&element_dtype)?, max.cast(&element_dtype)?];

    Ok(Scalar::struct_(partial_dtype, extrema))
}

/// Returns whether the [`Sum`] of the values of `batch` equals the sum of the canonical batch.
///
/// - The accumulator **must** sum each batch from zero and then add the total to the running sum.
///   It does so for an aggregate with the legacy [`Stat::Sum`] slot. Otherwise it adds each value
///   to the running sum, which a batch total cannot match near the 64-bit bounds: from a running
///   sum of `i64::MAX - 5`, the values `[10, -10]` overflow, but their total of zero does not.
/// - The sum of the batch **must not** be able to overflow. `Sum` adds every integer type into a
///   64-bit sum, but checks for overflow at different points for different widths: once per block
///   for types narrower than 64 bits, and after every value for 64-bit types. A signed value of
///   `b` bits lies in `[-2^(b-1), 2^(b-1))` and an unsigned one in `[0, 2^b)`, so no sum of at
///   most `2^(64 - b)` of them leaves the range of `i64` or `u64`, and no check fires on either
///   path.
///
/// Only `Sum` with `skip_nans` has the slot. Integers have no NaNs, so `skip_nans` matters here
/// only because it selects the slot.
fn is_exact_sum(aggregate_fn: &AggregateFnRef, batch: &ArrayRef, values: &ArrayRef) -> bool {
    let DType::Primitive(values_ptype, _) = values.dtype() else {
        return false;
    };
    let max_len = 1u64 << (64 - values_ptype.bit_width());

    Stat::from_aggregate_fn(aggregate_fn) == Some(Stat::Sum) && batch.len() as u64 <= max_len
}
