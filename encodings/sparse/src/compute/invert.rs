use vortex_array::compute::{InvertKernel, InvertKernelAdapter, invert};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{SparseArray, SparseVTable};

impl InvertKernel for SparseVTable {
    fn invert(&self, array: &SparseArray) -> VortexResult<ArrayRef> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array
            .patches()
            .clone()
            .map_values(|values| invert(&values))?;
        SparseArray::try_new_from_patches(inverted_patches, inverted_fill).map(|a| a.into_array())
    }
}

register_kernel!(InvertKernelAdapter(SparseVTable).lift());
