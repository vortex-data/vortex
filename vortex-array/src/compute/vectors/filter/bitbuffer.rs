// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{BitBuffer, BitBufferMut, get_bit};
use vortex_mask::{Mask, MaskIter};

use crate::compute::vectors::filter::Filter;

/// If the filter density is above 80%, we use slices to filter the array instead of indices.
// TODO(ngates): we need more experimentation to determine the best threshold here.
const FILTER_SLICES_DENSITY_THRESHOLD: f64 = 0.8;

impl Filter for BitBuffer {
    type Mutable = BitBufferMut;

    fn filter(&self, mask: &Mask) -> Self {
        assert_eq!(mask.len(), self.len());
        match mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => Self::empty(),
            Mask::Values(v) => match v.threshold_iter(FILTER_SLICES_DENSITY_THRESHOLD) {
                MaskIter::Indices(indices) => filter_indices(self, indices),
                MaskIter::Slices(slices) => filter_slices(self, mask.true_count(), slices),
            },
        }
    }

    fn filter_into(&self, mask: &Mask, _out: Self::Mutable) -> Self {
        self.filter(mask)
    }
}

fn filter_indices(bools: &BitBuffer, indices: &[usize]) -> BitBuffer {
    let buffer = bools.inner().as_ref();
    BitBuffer::collect_bool(indices.len(), |idx| {
        let idx = *unsafe { indices.get_unchecked(idx) };
        get_bit(buffer, bools.offset() + idx)
    })
}

fn filter_slices(buffer: &BitBuffer, output_len: usize, slices: &[(usize, usize)]) -> BitBuffer {
    let mut builder = BitBufferMut::with_capacity(output_len);
    for (start, end) in slices {
        // TODO(ngates): we probably want a borrowed slice for things like this.
        builder.append_buffer(&buffer.slice(*start..*end));
    }
    builder.freeze()
}

#[cfg(test)]
mod test {
    use vortex_buffer::bitbuffer;

    use super::*;

    #[test]
    fn filter_bool_by_slice_test() {
        let bits = bitbuffer![true, true, false];

        let filtered = filter_slices(&bits, 2, &[(0, 1), (2, 3)]);
        assert_eq!(2, filtered.len());

        assert_eq!(filtered, bitbuffer![true, false])
    }

    #[test]
    fn filter_bool_by_index_test() {
        let buf = bitbuffer![true, true, false];
        let filtered = filter_indices(&buf, &[0, 2]);
        assert_eq!(2, filtered.len());
        assert_eq!(bitbuffer![true, false], filtered)
    }
}
