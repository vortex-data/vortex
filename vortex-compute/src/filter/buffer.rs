// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_mask::{Mask, MaskIter};

use crate::filter::Filter;

// This is modeled after the constant with the equivalent name in arrow-rs.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl<T: Copy> Filter for Buffer<T> {
    fn filter(&self, mask: &Mask) -> Self {
        assert_eq!(mask.len(), self.len());
        match mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => Self::empty(),
            Mask::Values(v) => match v.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
                MaskIter::Indices(indices) => filter_indices(self.as_slice(), indices),
                MaskIter::Slices(slices) => {
                    filter_slices(self.as_slice(), mask.true_count(), slices)
                }
            },
        }
    }
}

fn filter_indices<T: Copy>(values: &[T], indices: &[usize]) -> Buffer<T> {
    Buffer::<T>::from_trusted_len_iter(indices.iter().map(|&idx| values[idx]))
}

fn filter_slices<T>(values: &[T], output_len: usize, slices: &[(usize, usize)]) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity(output_len);
    for (start, end) in slices {
        out.extend_from_slice(&values[*start..*end]);
    }
    out.freeze()
}
