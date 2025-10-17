// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`VectorOps`] and [`VectorOpsMut`] for [`Vector`] and
//! [`VectorMut`], respecitively.

use vortex_dtype::{DType, Nullability};

use crate::{
    Vector, VectorMut, match_each_vector, match_each_vector_mut, match_each_vector_mut_immut_pair,
    match_each_vector_mut_pair, private,
};

/// Common operations for immutable vectors.
pub trait VectorOps: private::Sealed + Into<Vector> {
    /// The mutable equivalent of this immutable vector.
    type Mutable: VectorMutOps<Immutable = Self>;

    /// Returns the [`Nullability`] of the vector.
    fn nullability(&self) -> Nullability;

    /// Returns the [`DType`] (or data type) of the vector.
    fn dtype(&self) -> DType;

    /// Returns the number of elements in the vector, also referred to as its "length".
    fn len(&self) -> usize;

    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Try to convert `self` into a mutable vector (implementing [`VectorMutOps`]).
    ///
    /// If `self` is the only unique strong reference (it effectively "owns" the buffer), this
    /// method will succeed and return a mutable vector with the contents of `self` **without**
    /// copying.
    ///
    /// If `self` is not unique, this will fail and return `self` back.
    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized;
}

impl VectorOps for Vector {
    type Mutable = VectorMut;

    fn nullability(&self) -> Nullability {
        match_each_vector!(self, |v| { v.nullability() })
    }

    fn dtype(&self) -> DType {
        match_each_vector!(self, |v| { v.dtype() })
    }

    fn len(&self) -> usize {
        match_each_vector!(self, |v| { v.len() })
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        match_each_vector!(self, |v| {
            v.try_into_mut().map(VectorMut::from).map_err(Vector::from)
        })
    }
}

/// Common operations for mutable vectors.
pub trait VectorMutOps: private::Sealed + Into<VectorMut> {
    /// The immutable equivalent of this mutable vector.
    type Immutable: VectorOps<Mutable = Self>;

    /// Returns the [`Nullability`] of the vector.
    fn nullability(&self) -> Nullability;

    /// Returns the [`DType`] (or data type) of the vector.
    fn dtype(&self) -> DType;

    /// Returns the number of elements in the vector, also referred to as its "length".
    fn len(&self) -> usize;

    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    fn capacity(&self) -> usize;

    /// Reserves capacity for at least `additional` more elements to be inserted in the given
    /// vector.
    ///
    /// The collection may reserve more space to speculatively avoid frequent reallocations. After
    /// calling `reserve`, the capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// Please let us know if you need `reserve_exact` functionality!
    fn reserve(&mut self, additional: usize);

    /// Extends the vector by appending elements from another vector.
    fn extend_from_vector(&mut self, other: &Self::Immutable);

    /// Converts `self` into an immutable vector.
    fn freeze(self) -> Self::Immutable;

    /// Splits the vector into two at the given index.
    ///
    /// Afterward, `self` contains elements `[0, at)`, and the returned vector contains elements
    /// `[at, capacity)`. It's guaranteed that the memory does not move, that is, the address of
    /// `self` does not change, and the address of the returned slice is at bytes after that.
    ///
    /// This is an `O(1)` operation that just increases the reference count and sets a few indices.
    ///
    /// # Panics
    ///
    /// Panics if we try to split off more than the current capacity of the vector (if
    /// `at > capacity`).
    fn split_off(&mut self, at: usize) -> Self;

    /// Absorbs a mutable vector that was previously split off.
    ///
    /// If the two vectors were previously contiguous and not mutated in a way that causes
    /// re-allocation i.e., if other was created by calling split_off on this vector, then this is
    /// an O(1) operation that just decreases a reference count and sets a few indices.
    ///
    /// Otherwise, this method degenerates to `self.extend_from_vector(other.as_ref())`.
    fn unsplit(&mut self, other: Self);
}

impl VectorMutOps for VectorMut {
    type Immutable = Vector;

    fn nullability(&self) -> Nullability {
        match_each_vector_mut!(self, |v| { v.nullability() })
    }

    fn dtype(&self) -> DType {
        match_each_vector_mut!(self, |v| { v.dtype() })
    }

    fn len(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.len() })
    }

    fn capacity(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_vector_mut!(self, |v| { v.reserve(additional) })
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        match_each_vector_mut_immut_pair!(self, other, |a, b| {
            a.extend_from_vector(b);
        });
    }

    fn freeze(self) -> Self::Immutable {
        match_each_vector_mut!(self, |v| { v.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_vector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match_each_vector_mut_pair!(self, other, |a, b| {
            a.unsplit(b);
        });
    }
}
