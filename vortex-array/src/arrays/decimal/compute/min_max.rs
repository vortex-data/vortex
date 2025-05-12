use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::{
    DecimalValue, NativeDecimalType, Scalar, ScalarValue, match_each_decimal_value_type,
};

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for DecimalVTable {
    fn min_max(&self, array: &DecimalArray) -> VortexResult<Option<MinMaxResult>> {
        match_each_decimal_value_type!(array.values_type(), |$T| {
            compute_min_max_with_validity::<$T>(array)
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
        Mask::AllTrue(_) => compute_min_max(array.buffer::<D>().iter(), array.dtype()),
        Mask::AllFalse(_) => None,
        Mask::Values(v) => compute_min_max(
            array
                .buffer::<D>()
                .iter()
                .zip(v.boolean_buffer().iter())
                .filter_map(|(v, m)| m.then_some(v)),
            array.dtype(),
        ),
    })
}

fn compute_min_max<'a, T>(iter: impl Iterator<Item = &'a T>, dtype: &DType) -> Option<MinMaxResult>
where
    T: Into<DecimalValue> + NativeDecimalType + Ord + Copy + 'a,
{
    match iter.minmax_by(|a, b| a.cmp(b)) {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(&x) => {
            let scalar = Scalar::new(dtype.clone(), ScalarValue::from(x.into()));
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(&min, &max) => Some(MinMaxResult {
            min: Scalar::new(dtype.clone(), ScalarValue::from(min.into())),
            max: Scalar::new(dtype.clone(), ScalarValue::from(max.into())),
        }),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;
    use vortex_scalar::{DecimalValue, Scalar, ScalarValue};

    use crate::arrays::DecimalArray;
    use crate::compute::{MinMaxResult, min_max};
    use crate::validity::Validity;

    #[test]
    fn min_max_test() {
        let decimal = DecimalArray::new(
            buffer![100i32, 2000i32, 200i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true]),
        );

        let min_max = min_max(decimal.as_ref()).unwrap();

        let expected = MinMaxResult {
            min: Scalar::new(
                decimal.dtype().clone(),
                ScalarValue::from(DecimalValue::from(100i32)),
            ),
            max: Scalar::new(
                decimal.dtype().clone(),
                ScalarValue::from(DecimalValue::from(200i32)),
            ),
        };

        assert_eq!(Some(expected), min_max)
    }
}
