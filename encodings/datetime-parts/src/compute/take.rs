use vortex_array::compute::{take, TakeFn, TakeOptions};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl TakeFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn take(
        &self,
        array: &DateTimePartsArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            take(array.days(), indices, options)?,
            take(array.seconds(), indices, options)?,
            take(array.subsecond(), indices, options)?,
        )?
        .into_array())
    }
}
