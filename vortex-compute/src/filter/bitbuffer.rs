// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{
    BitBuffer, BitBufferMut, BitView, get_bit, get_bit_unchecked, set_bit_unchecked,
    unset_bit_unchecked,
};
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

impl<const NB: usize> Filter<BitView<'_, NB>> for &BitBuffer {
    type Output = BitBuffer;

    fn filter(self, selection: &BitView<'_, NB>) -> BitBuffer {
        let bits = self.inner().as_ptr();
        let mut out = BitBufferMut::with_capacity(selection.true_count());
        let mut out_idx = 0;
        selection.iter_ones(|idx| {
            let value = unsafe { get_bit_unchecked(bits, self.offset() + idx) };
            unsafe { out.set_to_unchecked(out_idx, value) };
            out_idx += 1;
        });
        out.freeze()
    }
}

impl<const NB: usize> Filter<BitView<'_, NB>> for &mut BitBufferMut {
    type Output = ();

    fn filter(self, selection: &BitView<'_, NB>) {
        assert_eq!(
            self.len(),
            BitView::<NB>::N,
            "Selection mask length must equal the mask length"
        );

        let this = std::mem::take(self);

        let offset = this.offset();
        let mut buffer = this.into_inner();

        let buffer_ptr = buffer.as_mut_ptr();
        let mut out_idx = 0;
        selection.iter_ones(|idx| {
            let value = unsafe { get_bit_unchecked(buffer_ptr, offset + idx) };

            // NOTE(ngates): we don't call out.set_bit_unchecked here because it's nice that we
            //  can shift away any non-zero offset by writing directly into the bits buffer.
            if value {
                unsafe { set_bit_unchecked(buffer_ptr, out_idx) };
            } else {
                unsafe { unset_bit_unchecked(buffer_ptr, out_idx) };
            }
            out_idx += 1;
        });

        *self = BitBufferMut::from_buffer(buffer, 0, selection.true_count());
    }
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
