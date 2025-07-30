// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::vector::VType;
use vortex_dtype::NativePType;

/// A type-erased view of canonical data.
///
/// We can't have `View` as a trait, since downcasting view `Any` requires a static lifetime.
/// We can't really have `View` as an enum, since we cannot downcast using generics, we need
/// as_u32, as_u64, etc.
/// Instead, we have a generic `View` struct that can hold any type of vector data, then
/// downcasting happens by applying a view over the struct, where we validate the vtype.
///
pub struct View<'a> {
    vtype: VType,
    capacity: usize,
    elements: &'a mut [u8],
}

impl<'a> View<'a> {
    /// Create a new `PrimitiveView` from a `View`.
    pub fn new_primitive<T: NativePType>(elements: &'a mut [T]) -> View<'a> {
        View {
            vtype: VType::Primitive(T::PTYPE),
            capacity: elements.len(),
            elements: unsafe {
                // SAFETY: We assume that the elements are of type T and that the view is valid.
                std::slice::from_raw_parts_mut(
                    elements.as_mut_ptr() as *mut u8,
                    elements.len() * size_of::<T>(),
                )
            },
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn as_primitive<T: NativePType>(&'a mut self) -> PrimitiveView<'a, T> {
        assert_eq!(
            self.vtype,
            VType::Primitive(T::PTYPE),
            "Invalid type for view"
        );
        PrimitiveView {
            view: self,
            phantom: std::marker::PhantomData,
        }
    }
}

pub struct PrimitiveView<'a, T> {
    view: &'a mut View<'a>,
    phantom: std::marker::PhantomData<T>,
}

impl<T: NativePType> AsMut<[T]> for PrimitiveView<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        // SAFETY: We assume that the elements are of type T and that the view is valid.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.view.elements.as_mut_ptr() as *mut T,
                self.view.capacity,
            )
        }
    }
}
