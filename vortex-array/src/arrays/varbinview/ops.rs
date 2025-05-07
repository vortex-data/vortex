use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{VarBinViewArray, varbin_scalar};
use crate::{Array, ArrayOperationsImpl, ArrayRef};

impl ArrayOperationsImpl for VarBinViewArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let views = self.views().slice(start..stop);

        Ok(VarBinViewArray::try_new(
            views,
            self.buffers().to_vec(),
            self.dtype().clone(),
            self.validity().slice(start, stop)?,
        )?
        .into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(self.bytes_at(index), self.dtype()))
    }
}
