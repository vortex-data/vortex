use itertools::Itertools;
use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, register_kernel};

impl MinMaxKernel for PrimitiveEncoding {
    fn min_max(&self, array: &PrimitiveArray) -> VortexResult<Option<MinMaxResult>> {
        match_each_native_ptype!(array.ptype(), |$T| {
            compute_min_max_with_validity::<$T>(array)
        })
    }
}

register_kernel!(MinMaxKernelAdapter(PrimitiveEncoding).lift());

#[inline]
fn compute_min_max_with_validity<T>(array: &PrimitiveArray) -> VortexResult<Option<MinMaxResult>>
where
    T: Into<ScalarValue> + NativePType,
{
    Ok(match array.validity_mask()? {
        Mask::AllTrue(_) => compute_min_max(array.as_slice::<T>().iter(), array.dtype()),
        Mask::AllFalse(_) => None,
        Mask::Values(v) => compute_min_max(
            array
                .as_slice::<T>()
                .iter()
                .zip(v.boolean_buffer().iter())
                .filter_map(|(v, m)| m.then_some(v)),
            array.dtype(),
        ),
    })
}

fn compute_min_max<'a, T>(iter: impl Iterator<Item = &'a T>, dtype: &DType) -> Option<MinMaxResult>
where
    T: Into<ScalarValue> + NativePType,
{
    // `total_compare` function provides a total ordering (even for NaN values).
    // However, we exclude NaNs from min max as they're not useful for any purpose where min/max would be used
    match iter
        .filter(|v| !v.is_nan())
        .minmax_by(|a, b| a.total_compare(**b))
    {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(&x) => {
            let scalar = Scalar::new(dtype.clone(), x.into());
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(&min, &max) => Some(MinMaxResult {
            min: Scalar::new(dtype.clone(), min.into()),
            max: Scalar::new(dtype.clone(), max.into()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::arrays::PrimitiveArray;
    use crate::compute::min_max;
    use crate::validity::Validity;

    #[test]
    fn min_max_nan() {
        let array = PrimitiveArray::new(
            buffer![f32::NAN, -f32::NAN, -1.0, 1.0],
            Validity::NonNullable,
        );
        let min_max = min_max(&array).unwrap().unwrap();
        assert_eq!(f32::try_from(min_max.min).unwrap(), -1.0);
        assert_eq!(f32::try_from(min_max.max).unwrap(), 1.0);
    }

    #[test]
    fn min_max_inf() {
        let array = PrimitiveArray::new(
            buffer![f32::INFINITY, f32::NEG_INFINITY, -1.0, 1.0],
            Validity::NonNullable,
        );
        let min_max = min_max(&array).unwrap().unwrap();
        assert_eq!(f32::try_from(min_max.min).unwrap(), f32::NEG_INFINITY);
        assert_eq!(f32::try_from(min_max.max).unwrap(), f32::INFINITY);
    }
}
