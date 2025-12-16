// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVectorMut<T>`].

use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::MaskMut;

use crate::VectorMutOps;
use crate::VectorOps;
use crate::primitive::PScalar;
use crate::primitive::PVector;

/// A mutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`BufferMut<T>`]
/// that stores the elements of the vector.
#[derive(Debug, Clone)]
pub struct PVectorMut<T> {
    /// The mutable buffer representing the vector elements.
    pub(super) elements: BufferMut<T>,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: MaskMut,
}

impl<T> PVectorMut<T> {
    /// Creates a new [`PVectorMut<T>`] from the given elements buffer and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the elements buffer.
    pub fn new(elements: BufferMut<T>, validity: MaskMut) -> Self {
        Self::try_new(elements, validity).vortex_expect("Failed to create `PVectorMut`")
    }

    /// Tries to create a new [`PVectorMut<T>`] from the given elements buffer and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the validity mask does not match the length of the
    /// elements buffer.
    pub fn try_new(elements: BufferMut<T>, validity: MaskMut) -> VortexResult<Self> {
        vortex_ensure!(
            validity.len() == elements.len(),
            "`PVectorMut` validity mask must have the same length as elements"
        );

        Ok(Self { elements, validity })
    }

    /// Creates a new [`PVectorMut<T>`] from the given elements buffer and validity mask without
    /// validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity mask has the same length as the elements buffer.
    ///
    /// Ideally, they are taken from `into_parts`, mutated in a way that doesn't re-allocate, and
    /// then passed back to this function.
    pub unsafe fn new_unchecked(elements: BufferMut<T>, validity: MaskMut) -> Self {
        if cfg!(debug_assertions) {
            Self::new(elements, validity)
        } else {
            Self { elements, validity }
        }
    }

    /// Create a new mutable primitive vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: BufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }

    /// Set the length of the vector.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: Self::capacity
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len < self.elements.capacity());
        debug_assert!(new_len < self.validity.capacity());
        unsafe { self.elements.set_len(new_len) };
        unsafe { self.validity.set_len(new_len) };
    }

    /// Returns a mutable reference to the elements buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that any mutations to the elements do not violate the
    /// invariants of the vector (e.g., the length must remain consistent with the elements buffer).
    pub unsafe fn elements_mut(&mut self) -> &mut BufferMut<T> {
        &mut self.elements
    }

    /// Returns a mutable reference to the validity mask.
    ///
    /// # Safety
    ///
    /// The caller must ensure that any mutations to the validity mask do not violate the
    /// invariants of the vector (e.g., the length must remain consistent with the elements buffer).
    pub unsafe fn validity_mut(&mut self) -> &mut MaskMut {
        &mut self.validity
    }

    /// Decomposes the primitive vector into its constituent parts (buffer and validity).
    pub fn into_parts(self) -> (BufferMut<T>, MaskMut) {
        (self.elements, self.validity)
    }

    /// Append n values to the vector.
    pub fn append_values(&mut self, value: T, n: usize)
    where
        T: Copy,
    {
        self.elements.push_n(value, n);
        self.validity.append_n(true, n);
    }

    /// Transmute a `PVectorMut<T>` into a `PVectorMut<U>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all values of type `T` in this vector are valid as type `U`.
    /// See [`std::mem::transmute`] for more details.
    ///
    /// # Panics
    ///
    /// Panics if the type `U` does not have the same size and alignment as `T`.
    pub unsafe fn transmute<U: NativePType>(self) -> PVectorMut<U> {
        let (buffer, mask) = self.into_parts();

        // SAFETY: same guarantees as this function.
        let buffer = unsafe { buffer.transmute::<U>() };

        PVectorMut::new(buffer, mask)
    }
}

impl<T: NativePType> VectorMutOps for PVectorMut<T> {
    type Immutable = PVector<T>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
        self.validity.reserve(additional);
    }

    fn clear(&mut self) {
        self.elements.clear();
        self.validity.clear();
    }

    fn truncate(&mut self, len: usize) {
        self.elements.truncate(len);
        self.validity.truncate(len);
    }

    /// Extends the vector by appending elements from another vector.
    fn extend_from_vector(&mut self, other: &PVector<T>) {
        self.elements.extend_from_slice(other.elements.as_slice());
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.push_n(T::zero(), n); // Note that the value we push doesn't actually matter.
        self.validity.append_n(false, n);
    }

    fn append_zeros(&mut self, n: usize) {
        self.elements.push_n(T::zero(), n);
        self.validity.append_n(true, n);
    }

    fn append_scalars(&mut self, scalar: &PScalar<T>, n: usize) {
        match scalar.value() {
            None => {
                self.append_nulls(n);
            }
            Some(v) => {
                self.append_values(v, n);
            }
        }
    }

    /// Freeze the vector into an immutable one.
    fn freeze(self) -> PVector<T> {
        PVector {
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        Self {
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        if self.is_empty() {
            *self = other;
            return;
        }
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }
}
