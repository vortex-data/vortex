// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::decimal::valid_range;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_decimal_value_type;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;

pub(super) fn accumulate_decimal(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_decimal_value_type!(array.values_type(), |T| {
        let local = compute_min_max_with_validity::<T>(array, ctx)?;
        partial.merge(args, local)
    })
}

fn compute_min_max_with_validity<D>(
    array: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<MinMaxResult>>
where
    D: Into<DecimalValue> + NativeDecimalType,
{
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    if matches!(&validity, Mask::AllFalse(_)) {
        return Ok(None);
    }

    let decimal_dtype = array.decimal_dtype();

    let Some((min, max)) = valid_range(array.buffer::<D>().as_slice(), &validity) else {
        return Ok(None);
    };

    let min_scalar = Scalar::try_decimal(min.into(), decimal_dtype, NonNullable)?;
    let max_scalar = if min == max {
        min_scalar.clone()
    } else {
        Scalar::try_decimal(max.into(), decimal_dtype, NonNullable)?
    };

    Ok(Some(MinMaxResult {
        min: min_scalar,
        max: max_scalar,
    }))
}
