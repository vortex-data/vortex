// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVectorMut<T>`].

use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_mask::MaskMut;

use crate::{PVector, VectorMutOps, VectorOps};

/// A mutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`BufferMut<T>`]
/// that stores the elements of the vector.
///
/// The immutable equivalent of this type is [`PVector<T>`].
#[derive(Debug, Clone)]
pub struct PVectorMut<T> {
    pub(super) elements: BufferMut<T>,
    pub(super) validity: MaskMut,
}

impl<T: NativePType> PVectorMut<T> {
    /// Create a new mutable primitive vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: BufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }

    // TODO(connor): Make this a proper Rust test after we replace `BooleanBuffer` with `BitBuffer`
    // in `MaskValues`.
    /// Creates a new [`PVectorMut<T>`] from an iterator of `Option<T>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// # Examples
    ///
    /// ```text
    /// use vortex_vector::{PVectorMut, VectorMutOps};
    ///
    /// let mut vec = PVectorMut::<i32>::from_option_iter([Some(1), None, Some(3)]);
    /// assert_eq!(vec.len(), 3);
    /// ```
    pub fn from_option_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
    {
        match PVector::from_option_iter(iter).try_into_mut() {
            Ok(res) => res,
            Err(_) => unreachable!("We just created the `PVector`, so we must own it"),
        }
    }
}

impl<T: NativePType> VectorMutOps for PVectorMut<T> {
    type Immutable = PVector<T>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
        self.validity.reserve(additional);
    }

    /// Extends the vector by appending elements from another vector.
    fn extend_from_vector(&mut self, other: &PVector<T>) {
        self.elements.extend_from_slice(other.elements.as_slice());
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.push_n(T::zero(), n);
        self.validity.append_n(false, n);
    }

    /// Freeze the vector into an immutable one.
    fn freeze(self) -> PVector<T> {
        PVector {
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        PVectorMut {
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }
}
