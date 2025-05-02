use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;

use crate::DateTimePartsArray;

impl ArrayOperationsImpl for DateTimePartsArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            self.dtype().clone(),
            self.days().slice(start, stop)?,
            self.seconds().slice(start, stop)?,
            self.subseconds().slice(start, stop)?,
        )?
        .into_array())
    }
}
