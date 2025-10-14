// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vectors contain owned fixed-size canonical arrays of elements.
//!

// TODO(ngates): Currently, the data in a vector is Arc'd. We should consider whether we want the
//  performance hit for as_mut(), or whether we want zero-copy cloning. Not clear that we ever
//  need the clone behavior.

use std::cell::{Ref, RefMut};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};

use crate::pipeline::N;
use crate::pipeline::bits::BitVector;
use crate::pipeline::types::{Element, VType};
use crate::pipeline::view::{View, ViewMut};

/// A vector contains fixed-size owned data in canonical form.
#[derive(Debug)]
pub struct Vector {
    /// The physical type of the vector, which defines how the elements are stored.
    vtype: VType,
    /// The allocated elements buffer.
    /// Alignment is at least the size of the element type.
    /// The capacity of the elements buffer is N * `size_of::<T>()` where T is the element type.
    elements: ByteBufferMut,
    /// The validity mask for the vector, indicating which elements in the buffer are valid.
    validity: BitVector,
    // The position of the selected values in the vector.
    selection: Selection,

    /// Additional buffers of data used by the vector, such as string data.
    // TODO(ngates): ideally these buffers are compressed somehow? E.g. using FSST?
    #[allow(dead_code)]
    data: Vec<ByteBuffer>,
}

impl Vector {
    pub fn new<T: Element>() -> Self {
        Self::new_with_vtype(T::vtype())
    }

    pub fn new_with_vtype(vtype: VType) -> Self {
        let mut elements = ByteBufferMut::with_capacity_aligned(
            vtype.byte_width() * N,
            Alignment::new(vtype.byte_width()),
        );
        unsafe { elements.set_len(vtype.byte_width() * N) };

        Self {
            vtype,
            elements,
            validity: BitVector::full().clone(),
            selection: Selection::Prefix,
            data: vec![],
        }
    }

    pub fn set_selection(&mut self, selection: Selection) {
        self.selection = selection;
    }

    pub fn as_mut_array<T: Element>(&mut self) -> &mut [T; N] {
        assert_eq!(self.vtype, T::vtype());
        unsafe { &mut *(self.elements.as_mut_ptr().cast::<T>().cast::<[T; N]>()) }
    }

    pub fn as_view_mut(&mut self) -> ViewMut<'_> {
        ViewMut {
            vtype: self.vtype,
            elements: self.elements.as_mut_ptr().cast(),
            validity: Some(self.validity.as_view_mut()),
            data: vec![],
            selection: self.selection,
            _marker: Default::default(),
        }
    }

    pub fn as_view(&self) -> View<'_> {
        View {
            vtype: self.vtype,
            elements: self.elements.as_ptr().cast(),
            validity: Some(self.validity.as_view()),
            selection: self.selection,
            data: vec![],
            _marker: Default::default(),
        }
    }
}

/// A [`VectorRef`] provides a small wrapper to allow accessing a [`View`] with the same lifetime
/// as the borrowed vector, rather than the lifetime of the [`Ref`].
pub struct VectorRef<'a> {
    // Use to ensure that view and borrow have the same lifetime.
    #[allow(dead_code)]
    borrow: Ref<'a, Vector>,
    view: View<'a>,
}

impl<'a> VectorRef<'a> {
    pub fn new(borrow: Ref<'a, Vector>) -> Self {
        let view = borrow.as_view();
        // SAFETY: we continue to hold onto the [`Ref`], so it is safe to erase the lifetime.
        let view = unsafe { std::mem::transmute::<View<'_>, View<'a>>(view) };
        Self { borrow, view }
    }

    pub fn as_view(&self) -> &View<'a> {
        &self.view
    }
}

impl<'a> Deref for VectorRef<'a> {
    type Target = View<'a>;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

/// A [`VectorRefMut`] provides a small wrapper to allow accessing a [`ViewMut`] with the same
/// lifetime as the borrowed vector, rather than the lifetime of the [`RefMut`].
pub struct VectorRefMut<'a> {
    // Use to ensure that view and borrow have the same lifetime.
    #[allow(dead_code)]
    borrow: RefMut<'a, Vector>,
    view: ViewMut<'a>,
}

impl<'a> VectorRefMut<'a> {
    pub fn new(mut borrow: RefMut<'a, Vector>) -> Self {
        let view = borrow.as_view_mut();
        // SAFETY: we continue to hold onto the [`Ref`], so it is safe to erase the lifetime.
        let view = unsafe { std::mem::transmute::<ViewMut<'_>, ViewMut<'a>>(view) };
        Self { borrow, view }
    }
}

impl<'a> Deref for VectorRefMut<'a> {
    type Target = ViewMut<'a>;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl<'a> DerefMut for VectorRefMut<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.view
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    Prefix,
    Mask,
}
