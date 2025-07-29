use vortex_error::VortexResult;

use crate::arrays::VarBinViewArray;
use crate::builders::VarBinViewBuilder;
use crate::compute::{ZipKernelAdapter, zip_impl_with_builder, zip_return_dtype};
use crate::{Array, ArrayRef, register_kernel};
use crate::{arrays::VarBinViewVTable, compute::ZipKernel};

impl ZipKernel for VarBinViewVTable {
    fn zip(
        &self,
        mask: &vortex_mask::Mask,
        if_true: &VarBinViewArray,
        if_false: &dyn Array,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<VarBinViewVTable>() else {
            return Ok(None);
        };
        Ok(Some(zip_impl_with_builder(
            mask,
            if_true.as_ref(),
            if_false.as_ref(),
            Box::new(VarBinViewBuilder::with_buffer_deduplication(
                zip_return_dtype(if_true.as_ref(), if_false.as_ref()),
                if_true.len(),
            )),
        )?))
    }
}

register_kernel!(ZipKernelAdapter(VarBinViewVTable).lift());
