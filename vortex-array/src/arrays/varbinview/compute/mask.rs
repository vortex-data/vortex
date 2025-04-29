use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl MaskKernel for VarBinViewEncoding {
    fn mask(&self, array: &VarBinViewArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(VarBinViewArray::try_new(
            array.views().clone(),
            array.buffers().to_vec(),
            array.dtype().as_nullable(),
            array.validity().mask(mask)?,
        )?
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(VarBinViewEncoding).lift());
