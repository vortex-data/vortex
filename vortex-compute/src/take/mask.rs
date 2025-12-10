// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::get_bit;
use vortex_dtype::UnsignedPType;
use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PVector;

use crate::take::LINUX_PAGE_SIZE;
use crate::take::Take;

impl<I: UnsignedPType> Take<[I]> for &Mask {
    type Output = Mask;

    fn take(self, indices: &[I]) -> Mask {
        match self {
            Mask::AllTrue(_) => Mask::AllTrue(indices.len()),
            Mask::AllFalse(_) => Mask::AllFalse(indices.len()),
            Mask::Values(mask_values) => {
                let taken_bit_buffer = mask_values.bit_buffer().take(indices);
                Mask::from_buffer(taken_bit_buffer)
            }
        }
    }
}

impl<I: UnsignedPType> Take<PVector<I>> for &Mask {
    type Output = Mask;

    /// Implementation of take on [`Mask`] that is null-aware.
    ///
    /// If an index is specified as null by the [`PVector`], then the taken mask value is set to
    /// `false`.
    ///
    /// This is useful for many of the `take` implementations for vectors.
    fn take(self, indices: &PVector<I>) -> Mask {
        let indices_validity = indices.validity();
        let indices_len = indices.len();

        let indices_validity_values = match indices_validity {
            Mask::AllTrue(_) => return self.take(indices.elements().as_slice()),
            Mask::AllFalse(_) => return Mask::AllFalse(indices_len),
            Mask::Values(indices_validity_values) => indices_validity_values,
        };

        match self {
            // Since all the values are true, the only false values will be from the indices.
            Mask::AllTrue(_) => Mask::Values(indices_validity_values.clone()),
            // Since all the values are already false, the indices nullability wont change anything.
            Mask::AllFalse(_) => Mask::AllFalse(indices_len),
            Mask::Values(mask_values) => {
                // For boolean arrays that roughly fit into a single page (at least, on Linux), it's
                // worth the overhead to convert to a `Vec<bool>`.
                if self.len() <= LINUX_PAGE_SIZE {
                    let bools = mask_values.bit_buffer().iter().collect();
                    Mask::from_buffer(take_byte_bool_nullable(bools, indices))
                } else {
                    Mask::from_buffer(take_bool_nullable(mask_values.bit_buffer(), indices))
                }
            }
        }
    }
}

fn take_byte_bool_nullable<I: UnsignedPType>(bools: Vec<bool>, indices: &PVector<I>) -> BitBuffer {
    BitBuffer::collect_bool(indices.len(), |idx| {
        indices
            .get(idx)
            .is_some_and(|bool_idx| bools[bool_idx.as_()])
    })
}

fn take_bool_nullable<I: UnsignedPType>(bools: &BitBuffer, indices: &PVector<I>) -> BitBuffer {
    // We dereference to the underlying buffer to avoid incurring an access cost on every index.
    let buffer = bools.inner().as_ref();
    let offset = bools.offset();

    BitBuffer::collect_bool(indices.len(), |idx| {
        indices
            .get(idx)
            .is_some_and(|bool_idx| get_bit(buffer, offset + bool_idx.as_()))
    })
}
