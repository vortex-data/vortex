mod between;
mod compare;
mod nan_count;

use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;

use crate::{ALPArray, ALPEncoding};

impl ComputeVTable for ALPEncoding {}

impl TakeKernel for ALPEncoding {
    fn take(&self, array: &ALPArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_encoded = take(array.encoded(), indices)?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices))
            .transpose()?
            .flatten()
            .map(|p| {
                p.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::try_new(taken_encoded, array.exponents(), taken_patches)?.into_array())
    }
}

register_kernel!(TakeKernelAdapter(ALPEncoding).lift());
