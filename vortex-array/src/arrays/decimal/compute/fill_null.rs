// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{Scalar, match_each_decimal_value_type};

use crate::arrays::DecimalVTable;
use crate::arrays::decimal::DecimalArray;
use crate::compute::{FillNullKernel, FillNullKernelAdapter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, ToCanonical, register_kernel};

impl FillNullKernel for DecimalVTable {
    fn fill_null(&self, array: &DecimalArray, fill_value: &Scalar) -> VortexResult<ArrayRef> {
        let result_validity = Validity::from(fill_value.dtype().nullability());

        Ok(match array.validity() {
            Validity::Array(is_valid) => {
                let is_invalid = is_valid.to_bool().bit_buffer().not();
                match_each_decimal_value_type!(array.values_type(), |T| {
                    let mut buffer = array.buffer::<T>().into_mut();
                    let fill_value = fill_value
                        .as_decimal()
                        .decimal_value()
                        .and_then(|v| v.cast::<T>())
                        .vortex_expect("top-level fill_null ensure non-null fill value");
                    for invalid_index in is_invalid.set_indices() {
                        buffer[invalid_index] = fill_value;
                    }
                    DecimalArray::new(buffer.freeze(), array.decimal_dtype(), result_validity)
                        .into_array()
                })
            }
            _ => unreachable!("checked in entry point"),
        })
    }
}

register_kernel!(FillNullKernelAdapter(DecimalVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::{DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

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
        assert!(p.validity_mask().all_true());
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
