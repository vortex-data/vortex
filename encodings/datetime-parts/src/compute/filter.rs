use vortex_array::compute::{filter, FilterFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl FilterFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn filter(&self, array: &DateTimePartsArray, mask: &Mask) -> VortexResult<Array> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            filter(array.days().as_ref(), mask)?,
            filter(array.seconds().as_ref(), mask)?,
            filter(array.subsecond().as_ref(), mask)?,
        )?
        .into_array())
    }
}
