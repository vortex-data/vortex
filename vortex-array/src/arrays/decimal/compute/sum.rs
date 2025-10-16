// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DECIMAL256_MAX_PRECISION;
use vortex_dtype::DecimalDType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::{DecimalValue, Scalar, match_each_decimal_value_type};

use crate::arrays::{DecimalArray, DecimalVTable, smallest_decimal_value_type};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::register_kernel;
use num_traits::{AsPrimitive, CheckedAdd};

// Its safe to use `AsPrimitive` here because we always cast up.
macro_rules! sum_decimal {
    ($ty:ty, $values:expr) => {{
        let mut sum: $ty = <$ty>::default();
        for v in $values.iter() {
            let v: $ty = (*v).as_();
            sum = CheckedAdd::checked_add(&sum, &v)
                .ok_or_else(|| vortex_err!("Overflow when summing decimal {sum:?} + {v:?}"))?;
        }
        sum
    }};
    ($ty:ty, $values:expr, $validity:expr) => {{
        use itertools::Itertools;

        let mut sum: $ty = <$ty>::default();
        for (v, valid) in $values.iter().zip_eq($validity.iter()) {
            if valid {
                let v: $ty = (*v).as_();
                sum = CheckedAdd::checked_add(&sum, &v)
                    .ok_or_else(|| vortex_err!("Overflow when summing decimal {sum:?} + {v:?}"))?
            }
        }
        sum
    }};
}

impl SumKernel for DecimalVTable {
    fn sum(&self, array: &DecimalArray) -> VortexResult<Scalar> {
        let decimal_dtype = array.decimal_dtype();
        let nullability = array.dtype().nullability();

        // Both Spark and DataFusion use this heuristic.
        // - https://github.com/apache/spark/blob/fcf636d9eb8d645c24be3db2d599aba2d7e2955a/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/Sum.scala#L66
        // - https://github.com/apache/datafusion/blob/4153adf2c0f6e317ef476febfdc834208bd46622/datafusion/functions-aggregate/src/sum.rs#L188
        let new_precision = u8::min(DECIMAL256_MAX_PRECISION, decimal_dtype.precision() + 10);
        let new_scale = decimal_dtype.scale();
        let return_dtype = DecimalDType::new(new_precision, new_scale);

        match array.validity_mask() {
            Mask::AllFalse(_) => {
                vortex_bail!("invalid state, all-null array should be checked by top-level sum fn")
            }
            Mask::AllTrue(_) => {
                let values_type = smallest_decimal_value_type(&return_dtype);
                match_each_decimal_value_type!(array.values_type(), |I| {
                    match_each_decimal_value_type!(values_type, |O| {
                        Ok(Scalar::decimal(
                            DecimalValue::from(sum_decimal!(O, array.buffer::<I>())),
                            return_dtype,
                            nullability,
                        ))
                    })
                })
            }
            Mask::Values(mask_values) => {
                let values_type = smallest_decimal_value_type(&return_dtype);
                match_each_decimal_value_type!(array.values_type(), |I| {
                    match_each_decimal_value_type!(values_type, |O| {
                        Ok(Scalar::decimal(
                            DecimalValue::from(sum_decimal!(
                                O,
                                array.buffer::<I>(),
                                mask_values.boolean_buffer()
                            )),
                            return_dtype,
                            nullability,
                        ))
                    })
                })
            }
        }
    }
}

register_kernel!(SumKernelAdapter(DecimalVTable).lift());
