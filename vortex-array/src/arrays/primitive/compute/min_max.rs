use itertools::Itertools;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::{MinMaxFn, MinMaxResult};
use crate::variants::PrimitiveArrayTrait;

impl MinMaxFn<PrimitiveArray> for PrimitiveEncoding {
    fn min_max(&self, array: &PrimitiveArray) -> VortexResult<Option<MinMaxResult>> {
        match_each_native_ptype!(array.ptype(), |$T| {
            compute_min_max_with_validity::<$T>(array)
        })
    }
}

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
    T: Into<ScalarValue> + NativePType + Copy,
{
    // this `compare` function provides a total ordering (even for NaN values)
    match iter.minmax_by(|a, b| a.total_compare(**b)) {
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
