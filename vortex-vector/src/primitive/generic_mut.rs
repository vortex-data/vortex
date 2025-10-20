// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`GenericPVectorMut<T>`].

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_mask::MaskMut;

use crate::{GenericPVector, VectorMutOps, VectorOps};

/// A mutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`BufferMut<T>`]
/// that stores the elements of the vector. Additionally, an optional [`MaskMut`] is stored to track
/// null primitive elements.
///
/// The immutable equivalent of this type is [`GenericPVector<T>`].
#[derive(Debug, Clone)]
pub struct GenericPVectorMut<T> {
    pub(super) elements: BufferMut<T>,
    pub(super) validity: Option<MaskMut>,
}

impl<T: NativePType> GenericPVectorMut<T> {
    /// Create a new mutable primitive vector with the given capacity and nullability.
    pub fn with_capacity(capacity: usize, nullability: Nullability) -> Self {
        let validity = nullability
            .is_nullable()
            .then(|| MaskMut::with_capacity(capacity));

        Self {
            elements: BufferMut::with_capacity(capacity),
            validity,
        }
    }
}

impl<T: NativePType> VectorMutOps for GenericPVectorMut<T> {
    type Immutable = GenericPVector<T>;

    fn nullability(&self) -> Nullability {
        Nullability::from(self.validity.is_some())
    }

    fn dtype(&self) -> DType {
        DType::Primitive(T::PTYPE, self.nullability())
    }

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
        if let Some(v) = self.validity.as_mut() {
            v.reserve(additional);
        }
    }

    /// Extends the vector by appending elements from another vector.
    fn extend_from_vector(&mut self, other: &GenericPVector<T>) {
        assert!(
            self.is_nullable() || !other.is_nullable(),
            "tried to extend a non-nullable `GenericPVector` with a nullable vector"
        );

        self.elements.extend_from_slice(other.elements.as_slice());

        match (&mut self.validity, &other.validity) {
            (Some(self_v), Some(other_v)) => self_v.append_mask(other_v),
            (Some(self_v), None) => self_v.append_n(true, other.elements.len()),
            (None, Some(other_v)) => {
                let mut new_validity =
                    MaskMut::new_true(self.elements.len() - other.elements.len());
                new_validity.append_mask(other_v);
                self.validity = Some(new_validity);
            }
            (None, None) => {}
        }
    }

    /// Freeze the vector into an immutable one.
    fn freeze(self) -> GenericPVector<T> {
        GenericPVector {
            elements: self.elements.freeze(),
            validity: self.validity.map(|v| v.freeze()),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        GenericPVectorMut {
            elements: self.elements.split_off(at),
            validity: self.validity.as_mut().map(|v| v.split_off(at)),
        }
    }

    fn unsplit(&mut self, other: Self) {
        let other_len = other.elements.len();
        self.elements.unsplit(other.elements);
        match (&mut self.validity, other.validity) {
            (Some(self_v), Some(other_v)) => self_v.unsplit(other_v),
            (Some(self_v), None) => self_v.append_n(true, other_len),
            (None, Some(other_v)) => {
                let mut new_validity = MaskMut::new_true(self.elements.len() - other_len);
                new_validity.unsplit(other_v);
                self.validity = Some(new_validity);
            }
            (None, None) => {}
        }
    }
}
