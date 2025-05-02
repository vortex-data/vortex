use vortex_array::compute::slice;
use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;

use crate::DictArray;

impl ArrayOperationsImpl for DictArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DictArray::try_new(slice(self.codes(), start, stop)?, self.values().clone())
            .map(|a| a.into_array())
    }
}
