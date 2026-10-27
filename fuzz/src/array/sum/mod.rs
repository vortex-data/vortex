// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::CheckedAdd;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::MAX_PRECISION;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::i256;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

#[cfg(test)]
mod tests;

/// Reference sum independent of grouping and addition order.
/// Returns `None` for floats or mixed-sign inputs whose positive or negative subtotal overflows
/// the native accumulator or decimal precision. Single-sign overflow returns a null scalar;
/// empty and all-null inputs return zero, matching `sum`.
pub fn sum_canonical_array(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    match array.dtype() {
        DType::Bool(_) => accumulate::<u64>(array, |_| true, Scalar::from, ctx),
        DType::Primitive(ptype, _) if ptype.is_unsigned_int() => {
            accumulate::<u64>(array, |_| true, Scalar::from, ctx)
        }
        DType::Primitive(ptype, _) if ptype.is_signed_int() => {
            accumulate::<i64>(array, |_| true, Scalar::from, ctx)
        }
        DType::Primitive(..) => Ok(None),
        DType::Decimal(input_dtype, _) => {
            let output_dtype = DecimalDType::new(
                (input_dtype.precision() + 10).min(MAX_PRECISION),
                input_dtype.scale(),
            );
            let limit = i256::from_i128(10)
                .checked_pow(output_dtype.precision().into())
                .vortex_expect("10^76 fits in i256");
            let values_type = DecimalType::smallest_decimal_value_type(&output_dtype);
            match_each_decimal_value_type!(values_type, |I| {
                let limit = <I as BigCast>::from(limit)
                    .vortex_expect("precision limit fits native accumulator");
                accumulate::<I>(
                    array,
                    |&value| -limit < value && value < limit,
                    |value| match value {
                        Some(value) => {
                            Scalar::decimal(DecimalValue::from(value), output_dtype, Nullable)
                        }
                        None => Scalar::null(DType::Decimal(output_dtype, Nullable)),
                    },
                    ctx,
                )
            })
        }
        _ => vortex_bail!("Unsupported sum dtype: {}", array.dtype()),
    }
}

/// Sum positive and negative values separately to bound every possible partial sum.
/// Reject overflowing bounds with mixed signs, since cancellation can make overflow depend on
/// grouping or addition order.
fn accumulate<T>(
    array: &ArrayRef,
    fits: impl Fn(&T) -> bool,
    scalar: impl Fn(Option<T>) -> Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>>
where
    T: BigCast + Copy + Zero + CheckedAdd + PartialOrd,
{
    let values = native_values::<T>(array, ctx)?;
    let positive = values
        .iter()
        .copied()
        .filter(|&value| value > T::zero())
        .try_fold(T::zero(), |sum, value| sum.checked_add(&value))
        .filter(&fits);
    let negative = values
        .iter()
        .copied()
        .filter(|&value| value < T::zero())
        .try_fold(T::zero(), |sum, value| sum.checked_add(&value))
        .filter(&fits);

    let value = match (positive, negative) {
        (Some(positive), Some(negative)) => positive.checked_add(&negative),
        (None, Some(negative)) if negative.is_zero() => None,
        (Some(positive), None) if positive.is_zero() => None,
        _ => return Ok(None),
    };
    Ok(Some(scalar(value)))
}

fn native_values<T: BigCast>(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<T>> {
    let canonical = array.clone().execute::<Canonical>(ctx)?;
    let valid = canonical
        .clone()
        .into_array()
        .validity()?
        .execute_mask(canonical.len(), ctx)?
        .to_bit_buffer();
    if valid.true_count() == 0 {
        return Ok(Vec::new());
    }
    Ok(match canonical {
        Canonical::Bool(array) => vec![
            T::from((array.to_bit_buffer() & valid).true_count() as u64)
                .vortex_expect("boolean count fits accumulator"),
        ],
        Canonical::Primitive(array) => match_each_integer_ptype!(array.ptype(), |P| {
            array
                .as_slice::<P>()
                .iter()
                .zip(valid.iter())
                .filter_map(|(&value, valid)| valid.then_some(value))
                .map(|value| T::from(value).vortex_expect("integer value fits accumulator"))
                .collect()
        }),
        Canonical::Decimal(array) => match_each_decimal_value_type!(array.values_type(), |D| {
            array
                .buffer::<D>()
                .iter()
                .zip(valid.iter())
                .filter_map(|(&value, valid)| valid.then_some(value))
                .map(|value| T::from(value).vortex_expect("decimal value fits accumulator"))
                .collect()
        }),
        _ => vortex_bail!("Unsupported sum dtype: {}", array.dtype()),
    })
}
