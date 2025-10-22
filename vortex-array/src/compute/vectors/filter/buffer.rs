// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute::vectors::filter::Filter;
use vortex_buffer::{Buffer, BufferMut};
use vortex_mask::{Mask, MaskIter};

// This is modeled after the constant with the equivalent name in arrow-rs.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl<T: Copy> Filter for Buffer<T> {
    type Mutable = BufferMut<T>;

    fn filter(&self, mask: &Mask) -> Self {
        self.filter_into(mask, BufferMut::empty())
    }

    fn filter_into(&self, mask: &Mask, out: Self::Mutable) -> Self {
        assert_eq!(mask.len(), self.len());
        match mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => {
                assert!(out.is_empty());
                out.freeze()
            }
            Mask::Values(v) => match v.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
                MaskIter::Indices(indices) => filter_indices_into(self.as_slice(), indices, out),
                MaskIter::Slices(slices) => {
                    filter_slices_into(self.as_slice(), mask.true_count(), slices, out)
                }
            },
        }
    }
}

fn filter_indices_into<T: Copy>(
    values: &[T],
    indices: &[usize],
    mut out: BufferMut<T>,
) -> Buffer<T> {
    out.extend_trusted(indices.iter().map(|&idx| values[idx]));
    out.freeze()
}

fn filter_slices_into<T>(
    values: &[T],
    output_len: usize,
    slices: &[(usize, usize)],
    mut out: BufferMut<T>,
) -> Buffer<T> {
    out.reserve(output_len);
    for (start, end) in slices {
        out.extend_from_slice(&values[*start..*end]);
    }
    out.freeze()
}
