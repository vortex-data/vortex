// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::vector::{N, Selection, VType, Vector, VectorMask};
use vortex_dtype::NativePType;

impl<'v> Vector<'v> {
    /// Create a new `Vector` holding primitive elements.
    pub fn new_primitive<T: NativePType>(
        elements: &'v mut [T],
        validity: &'v mut VectorMask,
    ) -> Vector<'v> {
        assert_eq!(elements.len(), N);
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
}

impl<'v> Vector<'v> {
    /// Access this vector as primitive.
    pub fn as_primitive<'a, T: NativePType>(&'a mut self) -> PrimitiveVector<'a, 'v, T> {
        assert_eq!(
            self.vtype,
            VType::Primitive(T::PTYPE),
            "Invalid type for primitive view"
        );
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

impl<T: NativePType> AsMut<[T]> for PrimitiveVector<'_, '_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        // SAFETY: We assume that the elements are of type T and that the view is valid.
        unsafe { std::slice::from_raw_parts_mut(self.view.elements.cast::<T>(), N) }
    }
}
