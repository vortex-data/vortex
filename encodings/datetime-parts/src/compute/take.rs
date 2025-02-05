use vortex_array::compute::{take, TakeFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl TakeFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn take(&self, array: &DateTimePartsArray, indices: &Array) -> VortexResult<Array> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            take(array.days(), indices)?,
            take(array.seconds(), indices)?,
            take(array.subseconds(), indices)?,
        )?
        .into_array())
    }
}
