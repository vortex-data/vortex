// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Not;

use vortex_buffer::BitBuffer;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::match_each_decimal_value_type;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_scalar::DecimalValue;
use vortex_scalar::Scalar;

use super::cast::upcast_decimal_values;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::DecimalVTable;
use crate::arrays::decimal::DecimalArray;
use crate::compute::FillNullKernel;
use crate::compute::FillNullKernelAdapter;
use crate::register_kernel;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl FillNullKernel for DecimalVTable {
    fn fill_null(&self, array: &DecimalArray, fill_value: &Scalar) -> VortexResult<ArrayRef> {
        let result_validity = Validity::from(fill_value.dtype().nullability());

        Ok(match array.validity() {
            Validity::Array(is_valid) => {
                let is_invalid = is_valid.to_bool().to_bit_buffer().not();
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
        })
    }
}

fn fill_invalid_positions<T: NativeDecimalType>(
    array: &DecimalArray,
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
                fill_invalid_positions::<U>(&upcasted, is_invalid, decimal_value, result_validity)
            })
        }
    }
}

fn fill_buffer<T: NativeDecimalType>(
    array: &DecimalArray,
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

register_kernel!(FillNullKernelAdapter(DecimalVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::Nullability;
    use vortex_scalar::DecimalValue;
    use vortex_scalar::Scalar;

    use crate::arrays::decimal::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::canonical::ToCanonical;
    use crate::compute::fill_null;
    use crate::validity::Validity;

    #[test]
    fn fill_null_leading_none() {
        let decimal_dtype = DecimalDType::new(19, 2);
        let arr = DecimalArray::from_option_iter(
            [None, Some(800i128), None, Some(1000i128), None],
            decimal_dtype,
        );
        let p = fill_null(
            arr.as_ref(),
            &Scalar::decimal(
                DecimalValue::I128(4200i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ),
        )
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
        assert!(p.validity_mask().unwrap().all_true());
    }

    #[test]
    fn fill_null_all_none() {
        let decimal_dtype = DecimalDType::new(19, 2);

        let arr = DecimalArray::from_option_iter(
            [Option::<i128>::None, None, None, None, None],
            decimal_dtype,
        );

        let p = fill_null(
            arr.as_ref(),
            &Scalar::decimal(
                DecimalValue::I128(25500i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ),
        )
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
        let decimal_dtype = DecimalDType::new(2, 0);
        let arr = DecimalArray::from_option_iter([None, Some(10i8), None], decimal_dtype);
        // i8 max is 127, so 200 doesn't fit — the array should be widened to i16.
        let result = fill_null(
            arr.as_ref(),
            &Scalar::decimal(
                DecimalValue::I128(200i128),
                DecimalDType::new(2, 0),
                Nullability::NonNullable,
            ),
        )
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
        let p = fill_null(
            arr.as_ref(),
            &Scalar::decimal(
                DecimalValue::I128(25500i128),
                DecimalDType::new(19, 2),
                Nullability::NonNullable,
            ),
        )
        .unwrap()
        .to_decimal();
        assert_arrays_eq!(
            p,
            DecimalArray::from_iter([800i128, 1000, 1200, 1400, 1600], decimal_dtype)
        );
    }
}
