use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl FilterKernel for DateTimePartsVTable {
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
register_kernel!(FilterKernelAdapter(DateTimePartsVTable).lift());
