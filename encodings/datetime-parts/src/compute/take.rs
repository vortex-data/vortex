use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl TakeKernel for DateTimePartsEncoding {
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

register_kernel!(TakeKernelAdapter(DateTimePartsEncoding).lift());
