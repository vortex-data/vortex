// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate partials computed from narrow `Widen` values.
//!
//! A child partial is reusable only when it equals the partial of the logical canonical batch.
//! Ordered aggregates preserve that result through widening; sum also depends on the accumulator
//! dtype and whether the aggregate combines batch totals.

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

/// Runs an aggregate of a [`Widen`] on its values and widens the partial to the logical dtype.
///
/// The values run the same accumulator code as the canonical array, so edge cases such as an empty
/// batch or a single null match by construction, and the values keep their own cached statistics
/// and kernels. [`MinMax`] runs [`min_max`] on the values instead, which reads their exact cached
/// extrema before it scans them. Decimal widening preserves order, equality and validity, so
/// extrema, constancy and sortedness have the same partial after widening. A decimal [`Sum`] is
/// served only with `skip_nans`, by [`decimal_sum_partial`] with the sum dtype of the logical
/// dtype. Integer [`Sum`] is served only when [`is_exact_sum`] holds.
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

        if batch.dtype().is_decimal() && aggregate_fn.is::<Sum>() {
            // `decimal_sum_partial` is exact only with the legacy `Stat::Sum` slot.
            if Stat::from_aggregate_fn(aggregate_fn) != Some(Stat::Sum) {
                return Ok(None);
            }

            let options = aggregate_fn.as_::<Sum>();
            let partial = decimal_sum_partial(widen.values(), options, batch.dtype(), ctx)?;
            return Ok(Some(partial));
        }

        match batch.dtype() {
            DType::Decimal(..) => {
                if !(aggregate_fn.is::<MinMax>()
                    || aggregate_fn.is::<Min>()
                    || aggregate_fn.is::<Max>()
                    || aggregate_fn.is::<IsConstant>()
                    || aggregate_fn.is::<IsSorted>())
                {
                    return Ok(None);
                }
            }
            dtype if dtype.is_int() => {}
            _ => return Ok(None),
        }

        let values = widen.values();
        if aggregate_fn.is::<Sum>() && !is_exact_sum(aggregate_fn, batch, values) {
            return Ok(None);
        }

        let partial_dtype = aggregate_fn.state_dtype(batch.dtype()).vortex_expect(
            "the served aggregate accepts the logical dtype when it accepts the narrower values",
        );

        if let Some(options) = aggregate_fn.as_opt::<MinMax>() {
            let partial = min_max_partial(values, *options, batch.dtype(), partial_dtype, ctx)?;
            return Ok(Some(partial));
        }

        let mut accumulator = aggregate_fn.accumulator(values.dtype())?;
        accumulator.accumulate(values, ctx)?;

        Ok(Some(accumulator.flush()?.cast(&partial_dtype)?))
    }
}

/// Returns the [`Sum`] partial of a decimal `Widen` with the logical dtype `logical` from its
/// `values`, in the sum dtype of `logical`.
///
/// The partial **must** equal the partial of the canonical batch. A child accumulator would sum
/// in the [`sum_decimal_dtype`] of the child, whose precision is lower, so it can overflow where
/// the canonical batch does not. So this calls [`Sum`] with the dtypes of `logical`.
/// [`Sum::accumulate`] sends a canonical child to [`accumulate_decimal`], which reads any stored
/// width into the canonical width of the sum dtype of `logical`. It sends a constant child to
/// [`multiply_constant`], so a constant child never materializes.
///
/// The partial is exact only for [`Sum`] with `skip_nans`, which has the legacy [`Stat::Sum`]
/// slot:
///
/// - With the slot, the accumulator sums each batch from an empty partial, and then folds the
///   batch partial into the running sum. So this function and the canonical batch both sum from
///   zero, and their partials are equal, including an overflow to null:
///   - For a canonical child, both make the same checked adds, in the same accumulator width, on
///     the same values, in the same order, and check the precision once at the end. Unlike the
///     integer [`is_exact_sum`], this needs no length bound.
///   - For a constant child, [`multiply_constant`] makes one checked multiply in `i256` and
///     checks the product against the sum precision. The canonical batch adds the same value `n`
///     times. Its running sums all have one sign and grow in magnitude toward the product, and
///     its accumulator width is the canonical width of the sum precision. So the adds overflow
///     that width, or fail the final precision check, exactly when the product does not fit the
///     sum precision. A null constant adds nothing on either path.
/// - Without the slot, as for [`SumV2`] or [`Sum`] without `skip_nans`, the running sum is the
///   start value of [`accumulate_decimal`], which checks the precision only at the end. A batch
///   total can then fail the precision where the running sum does not. For example, a parent
///   `Decimal(3, 0)` over a child `Decimal(2, 0)` sums in `Decimal(13, 0)`. From a running sum of
///   `-(10^13 - 1)`, a batch of `1.5e11` rows of `99` totals `1.485e13`, which fails precision 13
///   alone, while the running sum plus that total fits. So these stay on the default path.
///
/// This function neither reads nor caches [`Stat::Sum`] on the values, because a sum of the values
/// has the [`sum_decimal_dtype`] of the child. The accumulator caches the partial on the `Widen`.
///
/// [`accumulate_decimal`]: crate::aggregate_fn::fns::sum::accumulate_decimal
/// [`multiply_constant`]: crate::aggregate_fn::fns::sum::multiply_constant
/// [`sum_decimal_dtype`]: crate::aggregate_fn::fns::sum::sum_decimal_dtype
/// [`SumV2`]: crate::aggregate_fn::fns::sum_v2::SumV2
fn decimal_sum_partial(
    values: &ArrayRef,
    options: &NumericalAggregateOpts,
    logical: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let dtypes = AggregateDTypes::try_new(&Sum, options, logical.clone())?;
    let args = dtypes.args(options);
    let values = values.clone().execute::<Columnar>(ctx)?;

    let mut partial = Sum.empty_partial(args)?;
    Sum.accumulate(args, &mut partial, &values, ctx)?;

    Sum.to_scalar(args, &partial)
}

/// Returns the [`MinMax`] partial of a `Widen` with the logical dtype `logical` from its values.
///
/// [`min_max`] returns the exact cached extrema of the values before it scans them, so a child
/// that records its extrema needs no scan. A fresh accumulator would always scan. The extrema
/// widen to the non-nullable `logical` dtype, as the canonical accumulator reports them, and no
/// valid value gives a null partial.
fn min_max_partial(
    values: &ArrayRef,
    options: NumericalAggregateOpts,
    logical: &DType,
    partial_dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let Some(MinMaxResult { min, max }) = min_max(values, ctx, options)? else {
        return Ok(Scalar::null(partial_dtype));
    };

    let element_dtype = logical.as_nonnullable();
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
