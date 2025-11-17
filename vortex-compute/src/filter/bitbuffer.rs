// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{BitBuffer, BitBufferMut, get_bit};
use vortex_mask::Mask;

use crate::filter::{Filter, MaskIndices};

impl Filter<Mask> for &BitBuffer {
    type Output = BitBuffer;

    fn filter(self, selection_mask: &Mask) -> BitBuffer {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        match selection_mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => BitBuffer::empty(),
            Mask::Values(v) => {
                filter_indices(self.inner().as_ref(), self.offset(), v.indices()).freeze()
            }
        }
    }
}

impl Filter<Mask> for &mut BitBufferMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        match selection_mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(v) => {
                *self = filter_indices(self.inner().as_slice(), self.offset(), v.indices())
            }
        }
    }
}

impl Filter<MaskIndices<'_>> for &BitBuffer {
    type Output = BitBuffer;

    fn filter(self, indices: &MaskIndices) -> BitBuffer {
        filter_indices(self.inner().as_ref(), self.offset(), indices).freeze()
    }
}

impl Filter<MaskIndices<'_>> for &mut BitBufferMut {
    type Output = ();

    fn filter(self, indices: &MaskIndices) {
        *self = filter_indices(self.inner().as_ref(), self.offset(), indices)
    }
}

fn filter_indices(bools: &[u8], bit_offset: usize, indices: &[usize]) -> BitBufferMut {
    // FIXME(ngates): this is slower than it could be!
    BitBufferMut::collect_bool(indices.len(), |idx| {
        let idx = *unsafe { indices.get_unchecked(idx) };
        get_bit(bools, bit_offset + idx)
    })
}

#[cfg(test)]
mod test {
    use vortex_buffer::bitbuffer;

    use super::*;

    #[test]
    fn filter_bool_by_index_test() {
        let buf = bitbuffer![1 1 0];
        let filtered = filter_indices(buf.inner().as_ref(), 0, &[0, 2]).freeze();
        assert_eq!(2, filtered.len());
        assert_eq!(filtered, bitbuffer![1 0])
    }
}
