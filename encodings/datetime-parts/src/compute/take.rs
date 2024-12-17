use vortex_array::compute::{take, TakeFn};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl TakeFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn take(&self, array: &DateTimePartsArray, indices: &ArrayData) -> VortexResult<ArrayData> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            take(array.days(), indices)?,
            take(array.seconds(), indices)?,
            take(array.subsecond(), indices)?,
        )?
        .into_array())
    }
}
