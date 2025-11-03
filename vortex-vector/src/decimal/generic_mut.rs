// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DVectorMut<D>`].

use vortex_buffer::BufferMut;
use vortex_dtype::{DecimalDType, NativeDecimalType, PrecisionScale};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::MaskMut;

use crate::{DVector, VectorMutOps, VectorOps};

/// A specifically typed mutable decimal vector.
#[derive(Debug, Clone)]
pub struct DVectorMut<D> {
    /// The precision and scale of each decimal in the decimal vector.
    pub(super) ps: PrecisionScale<D>,
    /// The mutable buffer representing the vector decimal elements.
    pub(super) elements: BufferMut<D>,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: MaskMut,
}

impl<D: NativeDecimalType> DVectorMut<D> {
    /// Creates a new [`DVectorMut<D>`] from the given [`PrecisionScale`], elements buffer, and
    /// validity mask.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - The lengths of the `elements` and `validity` do not match.
    /// - Any of the elements are out of bounds for the given [`PrecisionScale`].
    pub fn new(ps: PrecisionScale<D>, elements: BufferMut<D>, validity: MaskMut) -> Self {
        Self::try_new(ps, elements, validity).vortex_expect("Failed to create `DVector`")
    }

    /// Tries to create a new [`DVectorMut<D>`] from the given [`PrecisionScale`], elements buffer,
    /// and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - The lengths of the `elements` and `validity` do not match.
    /// - Any of the elements are out of bounds for the given [`PrecisionScale`].
    pub fn try_new(
        ps: PrecisionScale<D>,
        elements: BufferMut<D>,
        validity: MaskMut,
    ) -> VortexResult<Self> {
        if elements.len() != validity.len() {
            vortex_bail!(
                "Elements length {} does not match validity length {}",
                elements.len(),
                validity.len()
            );
        }

        // We assert that each element is within bounds for the given precision/scale.
        if !elements.iter().all(|e| ps.is_valid(*e)) {
            vortex_bail!(
                "One or more elements are out of bounds for precision {} and scale {}",
                ps.precision(),
                ps.scale()
            );
        }

        Ok(Self {
            ps,
            elements,
            validity,
        })
    }

    /// Creates a new [`DVectorMut<D>`] from the given [`PrecisionScale`], elements buffer, and
    /// validity mask, _without_ validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// - The lengths of the elements and validity are equal.
    /// - All elements are in bounds for the given [`PrecisionScale`].
    pub unsafe fn new_unchecked(
        ps: PrecisionScale<D>,
        elements: BufferMut<D>,
        validity: MaskMut,
    ) -> Self {
        if cfg!(debug_assertions) {
            Self::try_new(ps, elements, validity).vortex_expect("Failed to create `DVectorMut`")
        } else {
            Self {
                ps,
                elements,
                validity,
            }
        }
    }

    /// Create a new mutable primitive vector with the given capacity.
    pub fn with_capacity(decimal_dtype: &DecimalDType, capacity: usize) -> Self {
        Self {
            ps: PrecisionScale::try_from(decimal_dtype)
                .vortex_expect("TODO(someone): This definitely should not be fallible"),
            elements: BufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }

    /// Decomposes the decimal vector into its constituent parts ([`PrecisionScale`], decimal
    /// buffer, and validity).
    pub fn into_parts(self) -> (PrecisionScale<D>, BufferMut<D>, MaskMut) {
        (self.ps, self.elements, self.validity)
    }

    /// Get the precision/scale of the decimal vector.
    pub fn precision_scale(&self) -> PrecisionScale<D> {
        self.ps
    }

    /// Returns a reference to the underlying elements buffer containing the decimal data.
    pub fn elements(&self) -> &BufferMut<D> {
        &self.elements
    }

    /// Returns a mutable reference to the underlying elements buffer containing the decimal data.
    ///
    /// # Safety
    ///
    /// Modifying the elements buffer directly may violate the precision/scale constraints.
    /// The caller must ensure that any modifications maintain these invariants.
    pub unsafe fn elements_mut(&mut self) -> &mut BufferMut<D> {
        &mut self.elements
    }

    /// Gets a nullable element at the given index, panicking on out-of-bounds.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: D`.
    ///
    /// Note that this `get` method is different from the standard library [`slice::get`], which
    /// returns `None` if the index is out of bounds. This method will panic if the index is out of
    /// bounds, and return `None` if the elements is null.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&D> {
        self.validity.value(index).then(|| &self.elements[index])
    }

    /// Appends a new element to the end of the vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is out of bounds for the vector's precision/scale.
    pub fn try_push(&mut self, value: D) -> VortexResult<()> {
        if !self.ps.is_valid(value) {
            vortex_bail!("Value {:?} is out of bounds for {}", value, self.ps,);
        }

        self.elements.push(value);
        self.validity.append_n(true, 1);
        Ok(())
    }
}

impl<D: NativeDecimalType> AsRef<[D]> for DVectorMut<D> {
    fn as_ref(&self) -> &[D] {
        &self.elements
    }
}

impl<D: NativeDecimalType> VectorMutOps for DVectorMut<D> {
    type Immutable = DVector<D>;

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

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        self.elements.extend_from_slice(&other.elements);
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.extend((0..n).map(|_| D::default()));
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> Self::Immutable {
        DVector {
            ps: self.ps,
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        DVectorMut {
            ps: self.ps,
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }
}
