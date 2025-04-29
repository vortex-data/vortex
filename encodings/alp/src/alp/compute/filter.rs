use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPArray, ALPEncoding};

impl FilterKernel for ALPEncoding {
    fn filter(&self, array: &ALPArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let patches = array
            .patches()
            .map(|p| p.filter(mask))
            .transpose()?
            .flatten();

        Ok(
            ALPArray::try_new(filter(array.encoded(), mask)?, array.exponents(), patches)?
                .into_array(),
        )
    }
}

register_kernel!(FilterKernelAdapter(ALPEncoding).lift());
