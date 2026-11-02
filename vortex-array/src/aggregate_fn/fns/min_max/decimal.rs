// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::DecimalArray;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

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

    Ok(
        valid_range(array.buffer::<D>().as_slice(), &validity).map(|(min, max)| {
            let min_scalar = Scalar::decimal(min.into(), decimal_dtype, NonNullable);
            let max_scalar = if min == max {
                min_scalar.clone()
            } else {
                Scalar::decimal(max.into(), decimal_dtype, NonNullable)
            };

            MinMaxResult {
                min: min_scalar,
                max: max_scalar,
            }
        }),
    )
}

/// Returns the smallest and the largest valid value, or `None` when no value is valid.
///
/// Null slots can hold any value and do not contribute to the range. The scan allocates nothing
/// and does not check decimal precision.
///
/// # Panics
///
/// Panics if `validity` and `values` have different lengths.
pub fn valid_range<T: NativeDecimalType>(values: &[T], validity: &Mask) -> Option<(T, T)> {
    assert_eq!(
        validity.len(),
        values.len(),
        "valid_range requires one validity bit per value"
    );

    match validity {
        Mask::AllTrue(_) => compute_min_max(values.iter()),
        Mask::AllFalse(_) => None,
        Mask::Values(validity) => compute_min_max(
            values
                .iter()
                .zip(validity.bit_buffer().iter())
                .filter_map(|(value, valid)| valid.then_some(value)),
        ),
    }
}

fn compute_min_max<'a, T>(iter: impl Iterator<Item = &'a T>) -> Option<(T, T)>
where
    T: NativeDecimalType,
{
    match iter.minmax_by(|a, b| a.cmp(b)) {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(&value) => Some((value, value)),
        itertools::MinMaxResult::MinMax(&min, &max) => Some((min, max)),
    }
}
