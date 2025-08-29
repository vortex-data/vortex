// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{FilterKernel, FilterKernelAdapter};
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_buffer::{Buffer, BufferMut};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};

use crate::fsst_view::{FSSTViewArray, FSSTViewVTable};

const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl FilterKernel for FSSTViewVTable {
    fn filter(&self, array: &FSSTViewArray, mask: &Mask) -> VortexResult<ArrayRef> {
        // TODO(aduffy): do we want to compact the buffer if the filter is below some threshold?
        // Apply the mask to the views alone.
        let validity = array.validity().filter(mask)?;

        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        let filtered_views = match mask_values.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
            MaskIter::Indices(indices) => {
                filter_primitive_indices(array.views().as_slice(), indices.iter().copied())
            }
            MaskIter::Slices(slices) => filter_primitive_slices(
                array.views().as_slice(),
                mask.true_count(),
                slices.iter().copied(),
            ),
        };

        Ok(unsafe {
            FSSTViewArray::new_unchecked(
                filtered_views,
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

register_kernel!(FilterKernelAdapter(FSSTViewVTable).lift());

fn filter_primitive_indices<T: Copy>(
    values: &[T],
    indices: impl Iterator<Item = usize>,
) -> Buffer<T> {
    indices
        .map(|idx| *unsafe { values.get_unchecked(idx) })
        .collect()
}

fn filter_primitive_slices<T: Copy>(
    values: &[T],
    indices_len: usize,
    indices: impl Iterator<Item = (usize, usize)>,
) -> Buffer<T> {
    let mut output = BufferMut::with_capacity(indices_len);
    for (start, end) in indices {
        output.extend_from_slice(&values[start..end]);
    }
    output.freeze()
}
