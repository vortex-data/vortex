// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVecMut<T>`].

use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::MaskMut;

use crate::{PVec, VectorMutOps, VectorOps};

/// A mutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`BufferMut<T>`]
/// that stores the elements of the vector.
///
/// `PVecMut<T>` is the primary way to construct primitive vectors. It provides efficient methods
/// for building vectors incrementally before converting them to an immutable [`PVec<T>`] using
/// the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// # Examples
///
/// ## Creating and building a vector
///
/// ```
/// use vortex_vector::{PVecMut, VectorMutOps};
///
/// // Create with initial capacity for i32 values.
/// let mut vec = PVecMut::<i32>::with_capacity(10);
/// assert_eq!(vec.len(), 0);
/// assert!(vec.capacity() >= 10);
///
/// // Create from an iterator of optional values.
/// let mut vec = PVecMut::<i32>::from_iter([Some(1), None, Some(3)]);
/// assert_eq!(vec.len(), 3);
///
/// // Works with different primitive types.
/// let mut f64_vec = PVecMut::<f64>::from_iter([1.5, 2.5, 3.5].map(Some));
/// assert_eq!(f64_vec.len(), 3);
/// ```
///
/// ## Extending and appending
///
/// ```
/// use vortex_vector::{PVecMut, VectorMutOps};
///
/// let mut vec1 = PVecMut::<i32>::from_iter([1, 2].map(Some));
/// let vec2 = PVecMut::<i32>::from_iter([3, 4].map(Some)).freeze();
///
/// // Extend from another vector.
/// vec1.extend_from_vector(&vec2);
/// assert_eq!(vec1.len(), 4);
///
/// // Append null values.
/// vec1.append_nulls(2);
/// assert_eq!(vec1.len(), 6);
/// ```
///
/// ## Splitting and unsplitting
///
/// ```
/// use vortex_vector::{PVecMut, VectorMutOps};
///
/// let mut vec = PVecMut::<i64>::from_iter([10, 20, 30, 40, 50].map(Some));
///
/// // Split the vector at index 3.
/// let mut second_half = vec.split_off(3);
/// assert_eq!(vec.len(), 3);
/// assert_eq!(second_half.len(), 2);
///
/// // Rejoin the vectors.
/// vec.unsplit(second_half);
/// assert_eq!(vec.len(), 5);
/// ```
///
/// ## Working with nulls
///
/// ```
/// use vortex_vector::{PVecMut, VectorMutOps};
///
/// // Create a vector with some null values.
/// let mut vec = PVecMut::<u32>::from_iter([Some(100), None, Some(200), None]);
/// assert_eq!(vec.len(), 4);
///
/// // Add more nulls.
/// vec.append_nulls(3);
/// assert_eq!(vec.len(), 7);
/// ```
///
/// ## Converting to immutable
///
/// ```
/// use vortex_vector::{PVecMut, VectorMutOps, VectorOps};
///
/// let mut vec = PVecMut::<f32>::from_iter([1.0, 2.0, 3.0].map(Some));
///
/// // Freeze into an immutable vector.
/// let immutable = vec.freeze();
/// assert_eq!(immutable.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct PVecMut<T: NativePType> {
    /// The mutable buffer representing the vector elements.
    pub(super) elements: BufferMut<T>,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: MaskMut,
}

impl<T: NativePType> PVecMut<T> {
    /// Creates a new [`PVecMut<T>`] from the given elements buffer and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the elements buffer.
    pub fn new(elements: BufferMut<T>, validity: MaskMut) -> Self {
        Self::try_new(elements, validity)
            .vortex_expect("`PVecMut` validity mask must have the same length as elements")
    }

    /// Tries to create a new [`PVecMut<T>`] from the given elements buffer and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the validity mask does not match the length of the
    /// elements buffer.
    pub fn try_new(elements: BufferMut<T>, validity: MaskMut) -> VortexResult<Self> {
        vortex_ensure!(
            validity.len() == elements.len(),
            "`PVecMut` validity mask must have the same length as elements"
        );

        Ok(Self { elements, validity })
    }

    /// Creates a new [`PVecMut<T>`] from the given elements buffer and validity mask without
    /// validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity mask has the same length as the elements buffer.
    ///
    /// Ideally, they are taken from `into_parts`, mutated in a way that doesn't re-allocate, and
    /// then passed back to this function.
    pub unsafe fn new_unchecked(elements: BufferMut<T>, validity: MaskMut) -> Self {
        debug_assert_eq!(
            elements.len(),
            validity.len(),
            "`PVecMut` validity mask must have the same length as elements"
        );

        Self { elements, validity }
    }

    /// Create a new mutable primitive vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: BufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }
}

impl<T: NativePType> VectorMutOps for PVecMut<T> {
    type Immutable = PVec<T>;

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
    fn extend_from_vector(&mut self, other: &PVec<T>) {
        self.elements.extend_from_slice(other.elements.as_slice());
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.push_n(T::zero(), n);
        self.validity.append_n(false, n);
    }

    /// Freeze the vector into an immutable one.
    fn freeze(self) -> PVec<T> {
        PVec {
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        PVecMut {
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }
}
