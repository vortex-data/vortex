use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{VarBinArray, varbin_scalar};
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

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(self.bytes_at(index)?, self.dtype()))
    }
}
