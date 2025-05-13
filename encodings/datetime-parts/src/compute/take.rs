use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl TakeKernel for DateTimePartsVTable {
    fn take(&self, array: &DateTimePartsArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            take(array.days(), indices)?,
            take(array.seconds(), indices)?,
            take(array.subseconds(), indices)?,
        )?
        .into_array())
    }
}

register_kernel!(TakeKernelAdapter(DateTimePartsVTable).lift());
