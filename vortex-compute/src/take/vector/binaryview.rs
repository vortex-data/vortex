// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementations of `take` on [`BinaryViewVector`].
//!
//! take` on the binary view simply performs a `take` on the views of the vector, which means the
//! resulting vector may have "garbage" data in its child buffers.
//!
//! Note that it is on the outer array type to perform compaction / garbage collection.

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::binaryview::BinaryView;
use vortex_vector::binaryview::BinaryViewType;
use vortex_vector::binaryview::BinaryViewVector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<T: BinaryViewType, I: UnsignedPType> Take<PVector<I>> for &BinaryViewVector<T> {
    type Output = BinaryViewVector<T>;

    fn take(self, indices: &PVector<I>) -> BinaryViewVector<T> {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<T: BinaryViewType, I: UnsignedPType> Take<[I]> for &BinaryViewVector<T> {
    type Output = BinaryViewVector<T>;

    fn take(self, indices: &[I]) -> BinaryViewVector<T> {
        let taken_views = take_views(self.views(), indices);
        let taken_validity = self.validity().take(indices);

        debug_assert_eq!(taken_views.len(), taken_validity.len());

        // SAFETY: We called take on views and validity with the same indices, so the new components
        // must have the same length. The views still point into the same buffers which we clone via
        // Arc, so all view references remain valid.
        unsafe {
            BinaryViewVector::new_unchecked(taken_views, self.buffers().clone(), taken_validity)
        }
    }
}

fn take_nullable<T: BinaryViewType, I: UnsignedPType>(
    binary_view: &BinaryViewVector<T>,
    indices: &PVector<I>,
) -> BinaryViewVector<T> {
    // We ignore nullability when taking the views since we can let the `Mask` implementation
    // determine which elements are null.
    let taken_views = take_views(binary_view.views(), indices.elements().as_slice());

    // Note that this is **not** the same as the `indices: &[I]` `take` implementation above.
    let taken_validity = binary_view.validity().take(indices);

    debug_assert_eq!(taken_views.len(), taken_validity.len());

    // SAFETY: We used the same indices to take from both components, so they should still have the
    // same length. The views still point into the same buffers which we clone via Arc, so all view
    // references remain valid.
    unsafe {
        BinaryViewVector::new_unchecked(taken_views, binary_view.buffers().clone(), taken_validity)
    }
}

/// Takes views at the given indices.
fn take_views<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    Buffer::<BinaryView>::from_trusted_len_iter(indices.iter().map(|i| views[i.as_()]))
}
