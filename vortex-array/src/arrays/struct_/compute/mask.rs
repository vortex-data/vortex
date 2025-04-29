use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{StructArray, StructEncoding};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl MaskKernel for StructEncoding {
    fn mask(&self, array: &StructArray, filter_mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().mask(filter_mask)?;

        StructArray::try_new_with_dtype(
            array.fields().to_vec(),
            array.struct_dtype().clone(),
            array.len(),
            validity,
        )
        .map(|a| a.into_array())
    }
}
register_kernel!(MaskKernelAdapter(StructEncoding).lift());
