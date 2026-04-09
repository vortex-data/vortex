// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`BitBuffer`] filtering algorithms.

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::get_bit;
use vortex_mask::MaskValues;

/// Filter a [`BitBuffer`] by [`MaskValues`], returning a new [`BitBuffer`].
pub(super) fn filter_bit_buffer(bb: &BitBuffer, mask: &MaskValues) -> BitBuffer {
    assert_eq!(
        mask.len(),
        bb.len(),
        "Selection mask length must equal the mask length"
    );

    // BitBuffer filtering always uses indices for simplicity.
    filter_bitbuffer_by_indices(bb, mask.indices())
}

fn filter_bitbuffer_by_indices(bb: &BitBuffer, indices: &[usize]) -> BitBuffer {
    if indices.is_empty() {
        return BitBuffer::empty();
    }

    let mut out = BitBufferMut::with_capacity(indices.len());
    let bools = bb.inner().as_ref();
    let bit_offset = bb.offset();

    // Scan for contiguous runs in the indices and copy them in bulk.
    let mut i = 0;
    while i < indices.len() {
        let run_start = indices[i];
        let mut run_end = run_start + 1;
        let mut j = i + 1;
        while j < indices.len() && indices[j] == run_end {
            run_end += 1;
            j += 1;
        }

        let run_len = j - i;
        if run_len >= 64 {
            // Bulk copy for long contiguous runs.
            out.append_buffer(&bb.slice(run_start..run_end));
        } else {
            // Gather individual bits for short/scattered indices.
            for k in i..j {
                let idx = unsafe { *indices.get_unchecked(k) };
                out.append(get_bit(bools, bit_offset + idx));
            }
        }

        i = j;
    }

    out.freeze()
}

#[allow(unused)]
fn filter_bitbuffer_by_slices(bb: &BitBuffer, slices: &[(usize, usize)]) -> BitBuffer {
    let output_len: usize = slices.iter().map(|(start, end)| end - start).sum();
    let mut out = BitBufferMut::with_capacity(output_len);

    for &(start, end) in slices {
        out.append_buffer(&bb.slice(start..end));
    }

    out.freeze()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;

    use crate::arrays::filter::execute::bitbuffer::filter_bitbuffer_by_indices;

    #[test]
    fn filter_bool_by_index_test() {
        let buf = bitbuffer![1 1 0];
        let indices = [0usize, 2];
        let filtered = filter_bitbuffer_by_indices(&buf, &indices);
        assert_eq!(2, filtered.len());
        assert_eq!(filtered, bitbuffer![1 0])
    }
}
