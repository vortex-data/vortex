use vortex_array::compute::{FilterKernel, filter};
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl FilterKernel for DateTimePartsEncoding {
    fn filter(&self, array: &DateTimePartsArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            filter(array.days().as_ref(), mask)?,
            filter(array.seconds().as_ref(), mask)?,
            filter(array.subseconds().as_ref(), mask)?,
        )?
        .into_array())
    }
}
