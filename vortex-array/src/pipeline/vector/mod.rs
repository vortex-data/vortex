// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vectors contain owned fixed-size canonical arrays of elements.
//!

// TODO(ngates): Currently, the data in a vector is Arc'd. We should consider whether we want the
//  performance hit for as_mut(), or whether we want zero-copy cloning. Not clear that we ever
//  need the clone behavior.

use crate::pipeline::N;
use crate::pipeline::bits::BitVector;
use crate::pipeline::selection::Selection;
use crate::pipeline::types::{Element, VType};
use crate::pipeline::view::{TypedViewMut, View, ViewMut};
use std::fmt::Debug;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};

pub struct TypedVector<T> {
    vector: Vector,
    _marker: std::marker::PhantomData<T>,
}

impl<T> TypedVector<T> {
    pub fn as_typed_view_mut(&mut self) -> TypedViewMut<'_, T> {
        todo!()
    }

    pub fn as_view_mut(&mut self) -> ViewMut<'_> {
        todo!()
    }
}

impl<T> Default for TypedVector<T> {
    fn default() -> Self {
        todo!()
    }
}

/// A vector contains fixed-size owned data in canonical form.
#[derive(Debug)]
pub struct Vector {
    /// The physical type of the vector, which defines how the elements are stored.
    vtype: VType,
    /// The allocated elements buffer.
    /// Alignment is at least the size of the element type.
    /// The capacity of the elements buffer is N * size_of::<T>() where T is the element type.
    elements: ByteBufferMut,
    /// The validity mask for the vector, indicating which elements in the buffer are valid.
    validity: BitVector,
    // A selection mask over the elements and validity of the vector.
    selection: Selection,

    /// Additional buffers of data used by the vector, such as string data.
    // TODO(ngates): ideally these buffers are compressed somehow? E.g. using FSST?
    #[allow(dead_code)]
    data: Vec<ByteBuffer>,
}

impl Vector {
    pub fn new_with_vtype(vtype: VType) -> Self {
        let elements = ByteBufferMut::with_capacity_aligned(
            vtype.byte_width() * N,
            Alignment::new(vtype.byte_width()),
        );
        Self {
            vtype,
            elements,
            validity: BitVector::full().clone(),
            selection: Selection::default(),
            data: vec![],
        }
    }

    pub fn as_mut<T: Element>(&mut self) -> &mut [T; N] {
        assert_eq!(self.vtype, T::vtype());
        unsafe { &mut *(self.elements.as_mut_ptr().cast::<T>().cast::<[T; N]>()) }
    }

    pub fn as_view_mut(&mut self) -> ViewMut<'_> {
        ViewMut {
            vtype: self.vtype,
            elements: self.elements.as_mut_ptr().cast(),
            validity: Some(self.validity.as_view_mut()),
            selection: self.selection.clone(),
            data: vec![],
            children: vec![],
            _marker: Default::default(),
        }
    }

    pub fn as_view(&self) -> View<'_> {
        todo!()
    }
}
