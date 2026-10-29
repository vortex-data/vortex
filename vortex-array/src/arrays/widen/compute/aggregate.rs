// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
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
/// extrema, constancy and sortedness have the same partial after widening. Decimal sums must use
/// the logical precision to select their accumulator, so they fall back to execution. Integer
/// [`Sum`] is served only when [`is_exact_sum`] holds.
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
