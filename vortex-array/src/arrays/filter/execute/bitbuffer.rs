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
    let bools = bb.inner().as_ref();
    let bit_offset = bb.offset();

    // FIXME(ngates): this is slower than it could be!
    BitBufferMut::collect_bool(indices.len(), |idx| {
        let idx = *unsafe { indices.get_unchecked(idx) };
        get_bit(bools, bit_offset + idx) // Panics if out of bounds.
    })
    .freeze()
}

#[expect(unused)]
fn filter_bitbuffer_by_slices(bb: &BitBuffer, slices: &[(usize, usize)]) -> BitBuffer {
    let bools = bb.inner().as_ref();
    let bit_offset = bb.offset();
    let output_len: usize = slices.iter().map(|(start, end)| end - start).sum();

    let mut out = BitBufferMut::with_capacity(output_len);

    // FIXME(ngates): this is slower than it could be!
    for &(start, end) in slices {
        for idx in start..end {
            out.append(get_bit(bools, bit_offset + idx)); // Panics if out of bounds.
        }
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
