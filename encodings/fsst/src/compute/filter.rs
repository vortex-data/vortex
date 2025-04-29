use vortex_array::arrays::VarBinArray;
use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{Array, ArrayExt, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{FSSTArray, FSSTEncoding};

impl FilterKernel for FSSTEncoding {
    // Filtering an FSSTArray filters the codes array, leaving the symbols array untouched
    fn filter(&self, array: &FSSTArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols().clone(),
            array.symbol_lengths().clone(),
            filter(array.codes(), mask)?.as_::<VarBinArray>().clone(),
            filter(array.uncompressed_lengths(), mask)?,
        )?
        .into_array())
    }
}

register_kernel!(FilterKernelAdapter(FSSTEncoding).lift());
