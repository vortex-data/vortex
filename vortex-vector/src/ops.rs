// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::{Vector, VectorMut};

/// Common operations for immutable vectors.
pub trait VectorOps: Into<Vector> {
    type Mutable: VectorMutOps<Immutable = Self>;

    /// Returns the length of the vector.
    fn len(&self) -> usize;

    /// Returns the data type of the vector.
    fn dtype(&self) -> &DType;

    /// Try to convert self into a mutable vector.
    //
    // If self is the only unique strong reference, this will succeed and return a mutable vector
    // with the contents of self without copying. If self is not unique, this will fail and return
    // self.
    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized;
}

/// Common operations for mutable vectors.
pub trait VectorMutOps: Into<VectorMut> {
    type Immutable: VectorOps<Mutable = Self>;

    /// Returns the length of the vector.
    fn len(&self) -> usize;

    /// Returns the data type of the vector.
    fn dtype(&self) -> &DType;

    /// Returns the capacity of the vector.
    fn capacity(&self) -> usize;

    /// Reserves capacity for at least `additional` more elements to be inserted.
    fn reserve(&mut self, additional: usize);

    /// Splits the vector into two at the given index.
    ///
    /// Afterward, self contains elements `[0, at)`, and the returned vector contains elements
    /// `[at, capacity)`. It’s guaranteed that the memory does not move, that is, the address of
    /// self does not change, and the address of the returned slice is at bytes after that.
    ///
    /// This is an O(1) operation that just increases the reference count and sets a few indices.
    fn split_off(&mut self, at: usize) -> Self;

    /// Absorbs a mutable vector that was previously split off.
    ///
    /// If the two vectors were previously contiguous and not mutated in a way that causes
    /// re-allocation i.e., if other was created by calling split_off on this vector, then this is
    /// an O(1) operation that just decreases a reference count and sets a few indices.
    ///
    // Otherwise, this method degenerates to self.extend_from_vector(other.as_ref()).
    fn unsplit(&mut self, other: Self);

    /// Extends the vector by appending elements from another vector.
    fn extend_from_vector(&mut self, other: &Self::Immutable);

    /// Freeze the vector into an immutable one.
    fn freeze(self) -> Self::Immutable;
}
