mod compare;
mod is_constant;

use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, TakeKernel, TakeKernelAdapter, filter, take,
};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{FoRArray, FoREncoding};

impl TakeKernel for FoREncoding {
    fn take(&self, array: &FoRArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            take(array.encoded(), indices)?,
            array.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(TakeKernelAdapter(FoREncoding).lift());

impl FilterKernel for FoREncoding {
    fn filter(&self, array: &FoRArray, mask: &Mask) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            filter(array.encoded(), mask)?,
            array.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(FilterKernelAdapter(FoREncoding).lift());
