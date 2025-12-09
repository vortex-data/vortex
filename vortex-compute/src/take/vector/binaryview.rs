// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// use std::ops::Deref;

// use num_traits::AsPrimitive;
// use vortex_buffer::Buffer;
use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
// use vortex_vector::binaryview::BinaryView;
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

    fn take(self, _indices: &[I]) -> BinaryViewVector<T> {
        todo!("TODO(connor): Implement `take` for `BinaryViewVector` and figure out rebuilding");

        /*

        let taken_views = take_views(self.views(), indices);
        let taken_validity = self.validity().take(indices);

        debug_assert_eq!(taken_views.len(), taken_validity.len());

        // SAFETY: We called take on views and validity with the same indices, so the new components
        // must have the same length. The views still point into the same buffers which we clone via
        // Arc, so all view references remain valid.
        unsafe {
            BinaryViewVector::new_unchecked(taken_views, self.buffers().clone(), taken_validity)
        }

        */
    }
}

fn take_nullable<T: BinaryViewType, I: UnsignedPType>(
    _bvector: &BinaryViewVector<T>,
    _indices: &PVector<I>,
) -> BinaryViewVector<T> {
    todo!("TODO(connor): Implement `take` for `BinaryViewVector` and figure out rebuilding");

    /*

    // We ignore nullability when taking the views since we can let the `Mask` implementation
    // determine which elements are null.
    let taken_views = take_views(bvector.views(), indices.elements().as_slice());
    let taken_validity = bvector.validity().take(indices);

    debug_assert_eq!(taken_views.len(), taken_validity.len());

    // SAFETY: We used the same indices to take from both components, so they should still have the
    // same length. The views still point into the same buffers which we clone via Arc, so all view
    // references remain valid.
    unsafe {
        BinaryViewVector::new_unchecked(taken_views, bvector.buffers().clone(), taken_validity)
    }

    */
}

/*

/// Takes views at the given indices.
fn take_views<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    let views_ref = views.deref();
    Buffer::<BinaryView>::from_trusted_len_iter(indices.iter().map(|i| views_ref[(*i).as_()]))
}

*/
