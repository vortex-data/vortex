use itertools::Itertools;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::{DecimalValue, Scalar, i256};

use crate::arrays::decimal::serde::DecimalValueType;
use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::{Array, register_kernel};

macro_rules! sum_decimal {
    ($ty:ty, $values:expr) => {{
        let mut sum: $ty = <$ty>::default();
        for v in $values {
            sum = num_traits::CheckedAdd::checked_add(&sum, &v).expect("overflow");
        }
        sum
    }};
    ($ty:ty, $values:expr, $validity:expr) => {{
        let mut sum: $ty = <$ty>::default();
        for (v, valid) in $values.iter().zip_eq($validity.iter()) {
            if valid {
                sum = num_traits::CheckedAdd::checked_add(&sum, &v).expect("overflow");
            }
        }
        sum
    }};
}

impl SumKernel for DecimalEncoding {
    fn sum(&self, array: &DecimalArray) -> VortexResult<Scalar> {
        let decimal_dtype = array.decimal_dtype();
        let nullability = array.dtype.nullability();

        match (array.values_type, array.validity_mask()?) {
            (_, Mask::AllFalse(_)) => {
                vortex_bail!("invalid state, all-null array should be checked by top-level sum fn")
            }

            // fast paths: no validity checks needed
            (DecimalValueType::I128, Mask::AllTrue(_)) => Ok(Scalar::decimal(
                DecimalValue::I128(sum_decimal!(i128, array.buffer::<i128>())),
                decimal_dtype,
                nullability,
            )),
            (DecimalValueType::I256, Mask::AllTrue(_)) => Ok(Scalar::decimal(
                DecimalValue::I256(sum_decimal!(i256, array.buffer::<i256>())),
                decimal_dtype,
                nullability,
            )),
            // Variant that requires validity checks
            (DecimalValueType::I128, Mask::Values(mask_values)) => Ok(Scalar::decimal(
                DecimalValue::I128(sum_decimal!(
                    i128,
                    array.buffer::<i128>(),
                    mask_values.boolean_buffer()
                )),
                decimal_dtype,
                nullability,
            )),
            (DecimalValueType::I256, Mask::Values(mask_values)) => Ok(Scalar::decimal(
                DecimalValue::I256(sum_decimal!(
                    i256,
                    array.buffer::<i256>(),
                    mask_values.boolean_buffer()
                )),
                decimal_dtype,
                nullability,
            )),
        }
    }
}

register_kernel!(SumKernelAdapter(DecimalEncoding).lift());
