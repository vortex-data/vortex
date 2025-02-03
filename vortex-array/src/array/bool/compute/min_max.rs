use std::ops::BitAnd;

use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::{MinMaxFn, MinMaxResult};

impl MinMaxFn<BoolArray> for BoolEncoding {
    fn min_max(&self, array: &BoolArray) -> VortexResult<Option<MinMaxResult>> {
        let x = match array.validity_mask()? {
            Mask::AllTrue(_) => array.boolean_buffer(),
            Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => array.boolean_buffer().bitand(v.boolean_buffer()),
        };
        let mut slices = x.set_slices();
        // If there are no slices, then all values are false
        // if there is a single slice that covers the entire array, then all values are true
        // otherwise, we have a mix of true and false values

        let Some(slice) = slices.next() else {
            // all false
            return Ok(Some(MinMaxResult {
                min: Scalar::new(DType::Bool(NonNullable), false.into()),
                max: Scalar::new(DType::Bool(NonNullable), false.into()),
            }));
        };
        if slice.0 == 0 && slice.1 == x.len() {
            // all true
            return Ok(Some(MinMaxResult {
                min: Scalar::new(DType::Bool(NonNullable), true.into()),
                max: Scalar::new(DType::Bool(NonNullable), true.into()),
            }));
        };

        Ok(Some(MinMaxResult {
            min: Scalar::new(DType::Bool(NonNullable), false.into()),
            max: Scalar::new(DType::Bool(NonNullable), true.into()),
        }))
    }
}
