// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::mask::{BitMask, BitMaskView};
use crate::experiment::vector::{BitVector, N, Selection, VType, Vector};
use arrow_buffer::BooleanBuffer;
use bitvec::array::BitArray;
use bitvec::order::Msb0;
use bitvec::slice::BitSlice;
use std::mem::take;
use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

impl<'v> Vector<'v> {
    /// Create a new `Vector` holding primitive elements.
    pub fn new_primitive<T: NativePType>(
        elements: &'v mut [T],
        validity: Option<&'v mut BitVector>,
    ) -> Self {
        assert_eq!(elements.len(), N);
        assert!(validity.as_ref().map(|v| v.len() == N).unwrap_or(true));
        Vector {
            vtype: VType::Primitive(T::PTYPE),
            elements: elements.as_mut_ptr().cast(),
            validity,
            selection: Selection::Empty,
            data: vec![],
            children: vec![],
            _marker: Default::default(),
        }
    }

    /// Access this vector as primitive.
    pub fn as_primitive<'a, T: NativePType>(&'a mut self) -> PrimitiveVector<'a, 'v, T> {
        // assert_eq!(
        //     self.vtype,
        //     VType::Primitive(T::PTYPE),
        //     "Invalid type for primitive view"
        // );
        PrimitiveVector {
            view: self,
            phantom: std::marker::PhantomData,
        }
    }
}

/// A primitive vector holds elements of type `T` in the elements buffer.
///
// TODO(ngates): it would be really nice if we could do something like
//  `PrimitiveArray::as_vector(offset) -> Vector`, this would make it easy to allocate a primitive
//  array to use as a buffer for an export pipeline.
pub struct PrimitiveVector<'a, 'v, T> {
    view: &'a mut Vector<'v>,
    phantom: std::marker::PhantomData<T>,
}

impl<T: NativePType> PrimitiveVector<'_, '_, T> {
    /// Flatten the primitive vector in-place.
    pub fn flatten(&mut self) {
        let selection = take(&mut self.view.selection);
        let len = match selection {
            Selection::Empty => 0,
            Selection::Prefix { len } => len,
            Selection::Constant { element, len } => {
                let value = self.as_ref()[element];
                self.as_mut()[0..len].fill(value);
                len
            }
            Selection::Indices(indices) => {
                println!("INDICES: {:?}", indices);
                let mut buffer = [T::default(); N];
                for (i, &index) in indices.iter().enumerate() {
                    if index >= N {
                        vortex_panic!("Index out of bounds: {} >= {}", index, N);
                    }
                    buffer[i] = self.as_ref()[index];
                }
                self.as_mut().copy_from_slice(&buffer[0..indices.len()]);
                indices.len()
            }
            Selection::Mask(mask) => {
                let mut offset = 0;
                BitMaskView::Some(&mask).iter_ones(|idx| {
                    unsafe {
                        // SAFETY: We assume that the elements are of type T and that the view is valid.
                        *self.as_mut().get_unchecked_mut(offset) =
                            *self.as_ref().get_unchecked(idx);
                        offset += 1;
                    }
                });
                offset
                // FIXME(ngates): this "naive" implementation is way slower annoyingly. Which
                //  suggests we may not want to rely on bitvec crate for this :(
                // let mut offset = 0;
                // for idx in mask.iter_ones() {
                //     self.as_mut()[offset] = self.as_ref()[idx];
                //     offset += 1;
                // }
                // offset
            }
        };
        self.view.selection = Selection::Prefix { len };
    }
}

impl<T: NativePType> AsRef<[T]> for PrimitiveVector<'_, '_, T> {
    fn as_ref(&self) -> &[T] {
        // SAFETY: We assume that the elements are of type T and that the view is valid.
        unsafe { std::slice::from_raw_parts(self.view.elements.cast::<T>(), N) }
    }
}

impl<T: NativePType> AsMut<[T]> for PrimitiveVector<'_, '_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        // SAFETY: We assume that the elements are of type T and that the view is valid.
        unsafe { std::slice::from_raw_parts_mut(self.view.elements.cast::<T>(), N) }
    }
}
