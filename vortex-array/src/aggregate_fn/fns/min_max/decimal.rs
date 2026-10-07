// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::chunked::MinMax;
use crate::aggregate_fn::chunked::accumulate;
use crate::arrays::DecimalArray;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_decimal_value_type;
use crate::scalar::Scalar;

pub(super) fn accumulate_decimal(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    let decimal_dtype = array.decimal_dtype();
    // Decimals are ordered by their storage integers.
    let local = match_each_decimal_value_type!(array.values_type(), |D| {
        let mut acc = MinMax::<D>::new();
        accumulate(&array.buffer::<D>(), &validity, &mut acc);
        acc.finish().map(|(min, max)| MinMaxResult {
            min: Scalar::decimal(min.into(), decimal_dtype, NonNullable),
            max: Scalar::decimal(max.into(), decimal_dtype, NonNullable),
        })
    });
    partial.merge(args, local);
    Ok(())
}
