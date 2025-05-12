use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{StructArray, StructEncoding};
use crate::compute::{FilterKernel, FilterKernelAdapter, filter};
use crate::{Array, ArrayRef, register_kernel};

impl FilterKernel for StructEncoding {
    fn filter(&self, array: &StructArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().filter(mask)?;

        let fields: Vec<ArrayRef> = array
            .fields()
            .iter()
            .map(|field| filter(field, mask))
            .try_collect()?;
        let length = fields
            .first()
            .map(|a| a.len())
            .unwrap_or_else(|| mask.true_count());

        StructArray::try_new_with_dtype(fields, array.struct_dtype().clone(), length, validity)
            .map(|a| a.into_array())
    }
}

register_kernel!(FilterKernelAdapter(StructEncoding).lift());
