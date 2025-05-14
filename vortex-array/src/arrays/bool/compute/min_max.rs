use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for BoolVTable {
    fn min_max(&self, array: &BoolArray) -> VortexResult<Option<MinMaxResult>> {
        let x = match array.validity_mask()? {
            Mask::AllTrue(_) => array.boolean_buffer().clone(),
            Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => array.boolean_buffer().bitand(v.boolean_buffer()),
        };

        // TODO(ngates): we should be able to bail out earlier as soon as we have one true and
        //  one false value.
        let mut slices = x.set_slices();
        // If there are no slices, then all values are false
        // if there is a single slice that covers the entire array, then all values are true
        // otherwise, we have a mix of true and false values

        let Some(slice) = slices.next() else {
            // all false
            return Ok(Some(MinMaxResult {
                min: Scalar::new(array.dtype().clone(), false.into()),
                max: Scalar::new(array.dtype().clone(), false.into()),
            }));
        };
        if slice.0 == 0 && slice.1 == x.len() {
            // all true
            return Ok(Some(MinMaxResult {
                min: Scalar::new(array.dtype().clone(), true.into()),
                max: Scalar::new(array.dtype().clone(), true.into()),
            }));
        };

        Ok(Some(MinMaxResult {
            min: Scalar::new(array.dtype().clone(), false.into()),
            max: Scalar::new(array.dtype().clone(), true.into()),
        }))
    }
}

register_kernel!(MinMaxKernelAdapter(BoolVTable).lift());
