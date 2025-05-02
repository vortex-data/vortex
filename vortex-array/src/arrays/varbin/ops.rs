use vortex_error::VortexResult;

use crate::arrays::VarBinArray;
use crate::{Array, ArrayOperationsImpl, ArrayRef};

impl ArrayOperationsImpl for VarBinArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        VarBinArray::try_new(
            self.offsets().slice(start, stop + 1)?,
            self.bytes().clone(),
            self.dtype().clone(),
            self.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}
