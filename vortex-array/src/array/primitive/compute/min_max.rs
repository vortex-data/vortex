use itertools::Itertools;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::{MinMaxFn, MinMaxResult};
use crate::stats::{Precision, Stat};
use crate::variants::PrimitiveArrayTrait;

impl MinMaxFn<PrimitiveArray> for PrimitiveEncoding {
    fn min_max(&self, array: &PrimitiveArray) -> VortexResult<MinMaxResult> {
        let min = array.statistics().get_scalar(Stat::Min, array.dtype());
        let max = array.statistics().get_scalar(Stat::Max, array.dtype());

        if let (Some(min), Some(max)) = (min, max) {
            if min.is_exact() && max.is_exact() {
                return Ok(Some((min, max)));
            }
        }

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
        Mask::AllFalse(_) => None,
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
    // this `compare` function provides a total ordering (even for NaN values)
    match iter.minmax_by(|a, b| a.total_compare(**b)) {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(x) => {
            let scalar = Scalar::new(dtype.clone(), (*x).into());
            Some((Precision::exact(scalar.clone()), Precision::exact(scalar)))
        }
        itertools::MinMaxResult::MinMax(min, max) => Some((
            Precision::exact(Scalar::new(dtype.clone(), (*min).into())),
            Precision::exact(Scalar::new(dtype.clone(), (*max).into())),
        )),
    }
}
