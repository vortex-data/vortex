// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Not;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::cast::upcast_decimal_values;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::validity::Validity;

impl FillNullKernel for Decimal {
    fn fill_null(
        array: ArrayView<'_, Decimal>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let result_validity = Validity::from(fill_value.dtype().nullability());

        Ok(Some(match array.validity()? {
            Validity::Array(is_valid) => {
                let is_invalid = is_valid.execute::<BoolArray>(ctx)?.into_bit_buffer().not();
                let decimal_scalar = fill_value.as_decimal();
                let decimal_value = decimal_scalar
                    .decimal_value()
                    .vortex_expect("fill_null requires a non-null fill value");
                match_each_decimal_value_type!(array.values_type(), |T| {
                    fill_invalid_positions::<T>(
                        array,
                        &is_invalid,
                        &decimal_value,
                        result_validity,
                    )?
                })
            }
            _ => unreachable!("checked in entry point"),
        }))
    }
}

fn fill_invalid_positions<T: NativeDecimalType>(
    array: ArrayView<'_, Decimal>,
    is_invalid: &BitBuffer,
    decimal_value: &DecimalValue,
    result_validity: Validity,
) -> VortexResult<ArrayRef> {
    match decimal_value.cast::<T>() {
        Some(fill_val) => fill_buffer::<T>(array, is_invalid, fill_val, result_validity),
        None => {
            let target = max(array.values_type(), decimal_value.decimal_type());
            let upcasted = upcast_decimal_values(array, target)?;
            match_each_decimal_value_type!(upcasted.values_type(), |U| {
                let upcasted = upcasted.as_view();
                fill_invalid_positions::<U>(upcasted, is_invalid, decimal_value, result_validity)
            })
        }
    }
}

fn fill_buffer<T: NativeDecimalType>(
    array: ArrayView<'_, Decimal>,
    is_invalid: &BitBuffer,
    fill_val: T,
    result_validity: Validity,
) -> VortexResult<ArrayRef> {
    let mut buffer = array.buffer::<T>().into_mut();
    for invalid_index in is_invalid.set_indices() {
        buffer[invalid_index] = fill_val;
    }
    Ok(DecimalArray::new(buffer.freeze(), array.decimal_dtype(), result_validity).into_array())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn fill_null_leading_none() {
        let decimal_dtype = DecimalDType::new(19, 2);
        let arr = DecimalArray::from_option_iter(
            [None, Some(800i128), None, Some(1000i128), None],
            decimal_dtype,
        );
        #[expect(deprecated)]
        let p = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(4200i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .to_decimal();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([4200, 800, 4200, 1000, 4200], decimal_dtype)
        );
        assert_eq!(
            p.buffer::<i128>().as_slice(),
            vec![4200, 800, 4200, 1000, 4200]
        );
        assert!(
            p.as_ref()
                .validity()
                .unwrap()
                .to_mask(p.as_ref().len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .all_true()
        );
    }

    #[test]
    fn fill_null_all_none() {
        let decimal_dtype = DecimalDType::new(19, 2);

        let arr = DecimalArray::from_option_iter(
            [Option::<i128>::None, None, None, None, None],
            decimal_dtype,
        );

        #[expect(deprecated)]
        let p = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(25500i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .to_decimal();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([25500, 25500, 25500, 25500, 25500], decimal_dtype)
        );
    }

    /// fill_null with a value that overflows the array's storage type should upcast the array.
    #[test]
    fn fill_null_overflow_upcasts() {
        let decimal_dtype = DecimalDType::new(3, 0);
        let arr = DecimalArray::from_option_iter([None, Some(10i8), None], decimal_dtype);
        // i8 max is 127, so 200 doesn't fit — the array should be widened to i16.
        #[expect(deprecated)]
        let result = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(200i128),
                DecimalDType::new(3, 0),
                Nullability::NonNullable,
            ))
            .unwrap()
            .to_decimal();
        assert_arrays_eq!(
            result,
            DecimalArray::from_iter([200i16, 10, 200], decimal_dtype)
        );
    }

    #[test]
    fn fill_null_non_nullable() {
        let decimal_dtype = DecimalDType::new(19, 2);

        let arr = DecimalArray::new(
            buffer![800i128, 1000i128, 1200i128, 1400i128, 1600i128],
            decimal_dtype,
            Validity::NonNullable,
        );
        #[expect(deprecated)]
        let p = arr
            .into_array()
            .fill_null(Scalar::decimal(
                DecimalValue::I128(25500i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .to_decimal();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([800i128, 1000, 1200, 1400, 1600], decimal_dtype)
        );
    }
}
