use itertools::Itertools;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::{MinMaxFn, MinMaxResult};
use crate::variants::PrimitiveArrayTrait;

impl MinMaxFn<PrimitiveArray> for PrimitiveEncoding {
    fn min_max(&self, array: &PrimitiveArray) -> VortexResult<MinMaxResult> {
        match_each_native_ptype!(array.ptype(), |$T| {
            compute_min_max_with_validity::<$T>(array)
        })
    }
}

#[inline]
fn compute_min_max_with_validity<T: NativePType>(
    array: &PrimitiveArray,
) -> VortexResult<MinMaxResult>
where
    vortex_scalar::ScalarValue: From<T>,
{
    Ok(match array.validity_mask()? {
        Mask::AllTrue(_) => compute_min_max(array.as_slice::<T>().iter(), array.dtype()),
        Mask::AllFalse(_) => (None, None),
        Mask::Values(v) => compute_min_max(
            array
                .as_slice::<T>()
                .into_iter()
                .zip(v.boolean_buffer().iter())
                .filter_map(|(v, m)| if m { Some(v) } else { None }),
            array.dtype(),
        ),
    })
}

fn compute_min_max<'a, T: NativePType + Copy>(
    iter: impl Iterator<Item = &'a T>,
    dtype: &DType,
) -> MinMaxResult
where
    vortex_scalar::ScalarValue: From<T>,
{
    let dtype = dtype.clone().with_nullability(NonNullable);

    // this `compare` function provides a total ordering (even for NaN values)
    match iter.minmax_by(|a, b| a.total_compare(**b)) {
        itertools::MinMaxResult::NoElements => (None, None),
        itertools::MinMaxResult::OneElement(x) => {
            let scalar = Scalar::new(dtype.clone(), (*x).into());
            (Some(scalar.clone()), Some(scalar))
        }
        itertools::MinMaxResult::MinMax(min, max) => (
            Some(Scalar::new(dtype.clone(), (*min).into())),
            Some(Scalar::new(dtype.clone(), (*max).into())),
        ),
    }
}
