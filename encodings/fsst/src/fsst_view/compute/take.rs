// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use vortex_array::compute::{TakeKernel, TakeKernelAdapter};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexResult, VortexUnwrap};

use crate::{FSSTViewArray, FSSTViewVTable};

impl TakeKernel for FSSTViewVTable {
    #[allow(clippy::unnecessary_fallible_conversions)]
    fn take(&self, array: &FSSTViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let validity = array.validity.take(indices)?;

        let primitive_indices = indices.to_primitive();

        let views = array.views().deref();
        let taken_views = match_each_integer_ptype!(primitive_indices.ptype(), |I| {
            primitive_indices
                .as_slice::<I>()
                .iter()
                .map(|&idx| views[usize::try_from(idx).vortex_unwrap()])
                .collect()
        });

        // SAFETY: purely taking views doesn't modify internal pointers or compressed data buffer
        Ok(unsafe {
            FSSTViewArray::new_unchecked(
                taken_views,
                array.buffer().clone(),
                array.symbols.clone(),
                array.symbol_lengths.clone(),
                array.compressed_offsets.clone(),
                array.uncompressed_offsets.clone(),
                array.dtype.clone(),
                validity,
            )
            .into_array()
        })
    }
}

register_kernel!(TakeKernelAdapter(FSSTViewVTable).lift());
