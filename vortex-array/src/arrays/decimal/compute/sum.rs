use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::{DecimalValue, Scalar, match_each_decimal_value_type};

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::register_kernel;

macro_rules! sum_decimal {
    ($ty:ty, $values:expr) => {{
        let mut sum: $ty = <$ty>::default();
        for v in $values {
            sum = num_traits::CheckedAdd::checked_add(&sum, &v).expect("overflow");
        }
        sum
    }};
    ($ty:ty, $values:expr, $validity:expr) => {{
        use itertools::Itertools;

        let mut sum: $ty = <$ty>::default();
        for (v, valid) in $values.iter().zip_eq($validity.iter()) {
            if valid {
                sum = num_traits::CheckedAdd::checked_add(&sum, &v).expect("overflow");
            }
        }
        sum
    }};
}

impl SumKernel for DecimalVTable {
    fn sum(&self, array: &DecimalArray) -> VortexResult<Scalar> {
        let decimal_dtype = array.decimal_dtype();
        let nullability = array.dtype.nullability();

        match array.validity_mask()? {
            Mask::AllFalse(_) => {
                vortex_bail!("invalid state, all-null array should be checked by top-level sum fn")
            }
            Mask::AllTrue(_) => {
                match_each_decimal_value_type!(array.values_type(), |D| {
                    Ok(Scalar::decimal(
                        DecimalValue::from(sum_decimal!(D, array.buffer::<D>())),
                        decimal_dtype,
                        nullability,
                    ))
                })
            }
            Mask::Values(mask_values) => {
                match_each_decimal_value_type!(array.values_type(), |D| {
                    Ok(Scalar::decimal(
                        DecimalValue::from(sum_decimal!(
                            D,
                            array.buffer::<D>(),
                            mask_values.boolean_buffer()
                        )),
                        decimal_dtype,
                        nullability,
                    ))
                })
            }
        }
    }
}

register_kernel!(SumKernelAdapter(DecimalVTable).lift());
