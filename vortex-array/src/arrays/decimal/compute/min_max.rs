// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_decimal_value_type;
use crate::register_kernel;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

impl MinMaxKernel for DecimalVTable {
    fn min_max(&self, array: &DecimalArray) -> VortexResult<Option<MinMaxResult>> {
        match_each_decimal_value_type!(array.values_type(), |T| {
            compute_min_max_with_validity::<T>(array)
        })
    }
}

register_kernel!(MinMaxKernelAdapter(DecimalVTable).lift());

#[inline]
fn compute_min_max_with_validity<D>(array: &DecimalArray) -> VortexResult<Option<MinMaxResult>>
where
    D: Into<DecimalValue> + NativeDecimalType,
{
    Ok(match array.validity_mask()? {
        Mask::AllTrue(_) => compute_min_max(array.buffer::<D>().iter(), array.decimal_dtype()),
        Mask::AllFalse(_) => None,
        Mask::Values(v) => compute_min_max(
            array
                .buffer::<D>()
                .iter()
                .zip(v.bit_buffer().iter())
                .filter_map(|(v, m)| m.then_some(v)),
            array.decimal_dtype(),
        ),
    })
}

fn compute_min_max<'a, T>(
    iter: impl Iterator<Item = &'a T>,
    decimal_dtype: DecimalDType,
) -> Option<MinMaxResult>
where
    T: Into<DecimalValue> + NativeDecimalType + Ord + Copy + 'a,
{
    match iter.minmax_by(|a, b| a.cmp(b)) {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(&x) => {
            let scalar = Scalar::decimal(x.into(), decimal_dtype, NonNullable);
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(&min, &max) => Some(MinMaxResult {
            min: Scalar::decimal(min.into(), decimal_dtype, NonNullable),
            max: Scalar::decimal(max.into(), decimal_dtype, NonNullable),
        }),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::arrays::DecimalArray;
    use crate::compute::MinMaxResult;
    use crate::compute::min_max;
    use crate::dtype::DecimalDType;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[test]
    fn min_max_test() {
        let decimal = DecimalArray::new(
            buffer![100i32, 2000i32, 200i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true]),
        );

        let min_max = min_max(decimal.as_ref()).unwrap();

        let non_nullable_dtype = decimal.dtype().as_nonnullable();
        let expected = MinMaxResult {
            min: Scalar::try_new(
                non_nullable_dtype.clone(),
                Some(ScalarValue::from(DecimalValue::from(100i32))),
            )
            .unwrap(),
            max: Scalar::try_new(
                non_nullable_dtype,
                Some(ScalarValue::from(DecimalValue::from(200i32))),
            )
            .unwrap(),
        };

        assert_eq!(Some(expected), min_max)
    }
}
