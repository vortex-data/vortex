// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`VectorOps`] and [`VectorOps`] for [`Vector`] and
//! [`Vector`], respectively.

use std::fmt::Debug;
use std::ops::RangeBounds;

use vortex_mask::Mask;

use crate::cow::Cow;
use crate::{private, Scalar, Vector};

/// Common operations for mutable vectors (all the variants of [`Vector`]).
pub trait VectorOps: private::Sealed + Into<Vector> + Sized {
    /// Returns the number of elements in the vector, also referred to as its "length".
    fn len(&self) -> usize;

    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the validity mask of the vector, where `true` represents a _valid_ element and
    /// `false` represents a `null` element.
    fn validity(&self) -> &Cow<Mask>;

    /// Returns the mutable validity mask of the vector, where `true` represents a _valid_ element
    /// and `false` represents a `null` element.
    fn validity_mut(&mut self) -> &mut Cow<Mask>;

    /// Return the scalar at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    fn scalar_at(&self, index: usize) -> Scalar;

    /// Slice the vector from `start` to `end` (exclusive).
    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self;

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

    /// Clears the buffer, removing all data. Existing capacity is preserved.
    fn clear(&mut self);

    /// Shortens the buffer, keeping the first len bytes and dropping the rest.
    ///
    /// If len is greater than the buffer’s current length, this has no effect.
    ///
    /// Existing underlying capacity is preserved.
    fn truncate(&mut self, len: usize);

    /// Extends the vector by appending elements from another vector.
    ///
    /// # Panics
    ///
    /// Panics if the `other` vector has the wrong type (for example, a
    /// [`StructVector`](crate::struct_::StructVector) might have incorrect fields).
    fn extend_from_vector(&mut self, other: &Self);

    /// Appends `n` null elements to the vector.
    ///
    /// Implementors should ensure that they correctly append "null" or garbage values to their
    /// elements in addition to adding nulls to their validity mask.
    fn append_nulls(&mut self, n: usize);

    /// Converts `self` into an immutable vector by recursively calling [`Cow::freeze`].
    fn freeze(self) -> Self;

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

/// Converts a range bounds into a length, given the total length of the vector.
pub(crate) fn range_bounds_to_len(bounds: impl RangeBounds<usize> + Debug, len: usize) -> usize {
    use std::ops::Bound;

    let start = match bounds.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };

    let end = match bounds.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };

    assert!(
        start <= end && end <= len,
        "Range {:?} out of bounds for length {}",
        bounds,
        len
    );

    end - start
}
