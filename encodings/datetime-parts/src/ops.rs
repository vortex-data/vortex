use vortex_array::compute::slice;
use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;

use crate::DateTimePartsArray;

impl ArrayOperationsImpl for DateTimePartsArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            self.dtype().clone(),
            slice(self.days(), start, stop)?,
            slice(self.seconds(), start, stop)?,
            slice(self.subseconds(), start, stop)?,
        )?
        .into_array())
    }
}
