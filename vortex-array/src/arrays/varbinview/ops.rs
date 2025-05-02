use vortex_error::VortexResult;

use crate::arrays::VarBinViewArray;
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
}
