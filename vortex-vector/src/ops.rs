// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`VectorOps`] and [`VectorMutOps`] for [`Vector`] and
//! [`VectorMut`], respectively.

use vortex_mask::{Mask, MaskMut};

use crate::{Vector, VectorMut, private};

/// Common operations for immutable vectors (all the variants of [`Vector`]).
pub trait VectorOps: private::Sealed + Into<Vector> + Sized {
    /// The mutable equivalent of this immutable vector.
    type Mutable: VectorMutOps<Immutable = Self>;

    /// Returns the number of elements in the vector, also referred to as its "length".
    fn len(&self) -> usize;

    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the validity mask of the vector, where `true` represents a _valid_ element and
    /// `false` represents a `null` element.
    ///
    /// Note that vectors are **always** considered nullable. "Non-nullable" data will simply have a
    /// [`Mask`] of [`AllTrue(len)`](Mask::AllTrue). It is on the caller to ensure that they do not
    /// add nullable data to a vector they want to keep as non-nullable.
    fn validity(&self) -> &Mask;

    /// Tries to convert `self` into a mutable vector (implementing [`VectorMutOps`]).
    ///
    /// This method will only succeed if `self` is the only unique strong reference (it effectively
    /// "owns" the buffer). If this is true, this method will return a mutable vector with the
    /// contents of `self` **without** any copying of data.
    ///
    /// # Errors
    ///
    /// If `self` is not unique, this will fail and return `self` back to the caller.
    fn try_into_mut(self) -> Result<Self::Mutable, Self>;
}

/// Common operations for mutable vectors (all the variants of [`VectorMut`]).
pub trait VectorMutOps: private::Sealed + Into<VectorMut> + Sized {
    /// The immutable equivalent of this mutable vector.
    type Immutable: VectorOps<Mutable = Self>;

    /// Returns the number of elements in the vector, also referred to as its "length".
    fn len(&self) -> usize;

    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the validity mask of the vector, where `true` represents a _valid_ element and
    /// `false` represents a `null` element.
    ///
    /// Note that while this returns a [`MaskMut`] (which is typically an owned type), the caller is
    /// only allowed to inspect it via the shared reference.
    fn validity(&self) -> &MaskMut;

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
    ///
    /// # Panics
    ///
    /// Panics if the `other` vector has the wrong type (for example, a
    /// [`StructVector`](crate::struct_::StructVector) might have incorrect fields).
    fn extend_from_vector(&mut self, other: &Self::Immutable);

    /// Appends `n` null elements to the vector.
    ///
    /// Implementors should ensure that they correctly append "null" or garbage values to their
    /// elements in addition to adding nulls to their validity mask.
    fn append_nulls(&mut self, n: usize);

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
    /// re-allocation i.e., if other was created by calling [`split_off()`] on this vector, then
    /// this is an `O(1)` operation (simply decreases a reference count and sets a few indices).
    ///
    /// Otherwise, this method falls back to `self.extend_from_vector(other)`.
    ///
    /// [`split_off()`]: Self::split_off
    fn unsplit(&mut self, other: Self);
}
