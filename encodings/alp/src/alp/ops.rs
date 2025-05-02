use vortex_array::compute::slice;
use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;

use crate::ALPArray;

impl ArrayOperationsImpl for ALPArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ALPArray::try_new(
            slice(self.encoded(), start, stop)?,
            self.exponents(),
            self.patches()
                .map(|p| p.slice(start, stop))
                .transpose()?
                .flatten(),
        )?
        .into_array())
    }
}
