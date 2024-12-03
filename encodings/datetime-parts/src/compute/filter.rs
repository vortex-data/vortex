use vortex_array::compute::{filter, FilterFn, FilterMask};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl FilterFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn filter(&self, array: &DateTimePartsArray, mask: FilterMask) -> VortexResult<ArrayData> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            filter(array.days().as_ref(), mask.clone())?,
            filter(array.seconds().as_ref(), mask.clone())?,
            filter(array.subsecond().as_ref(), mask)?,
        )?
        .into_array())
    }
}
