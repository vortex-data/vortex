// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DVector<D>`].

use std::fmt::Debug;
use std::ops::RangeBounds;

use vortex_buffer::Buffer;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::PrecisionScale;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::VectorOps;
use crate::decimal::DScalar;
use crate::decimal::DVectorMut;

/// An immutable vector of generic decimal values.
///
/// `D` is bound by [`NativeDecimalType`], which can be one of the native integer types (`i8`,
/// `i16`, `i32`, `i64`, `i128`) or `i256`. `D` is used to store the decimal values.
///
/// The decimal vector maintains a [`PrecisionScale<D>`] that defines the precision (total number of
/// digits) and scale (digits after the decimal point) for all values in the vector.
#[derive(Debug, Clone)]
pub struct DVector<D> {
    /// The precision and scale of each decimal in the decimal vector.
    pub(super) ps: PrecisionScale<D>,
    /// The buffer representing the vector decimal elements.
    pub(super) elements: Buffer<D>,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
}

impl<D: NativeDecimalType> DVector<D> {
    /// Creates a new [`DVector<D>`] from the given [`PrecisionScale`], elements buffer, and
    /// validity mask.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - The lengths of the `elements` and `validity` do not match.
    /// - Any of the elements are out of bounds for the given [`PrecisionScale`].
    pub fn new(ps: PrecisionScale<D>, elements: Buffer<D>, validity: Mask) -> Self {
        Self::try_new(ps, elements, validity).vortex_expect("Failed to create `DVector`")
    }

    /// Tries to create a new [`DVector<D>`] from the given [`PrecisionScale`], elements buffer, and
    /// validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - The lengths of the `elements` and `validity` do not match.
    /// - Any of the elements are out of bounds for the given [`PrecisionScale`].
    pub fn try_new(
        ps: PrecisionScale<D>,
        elements: Buffer<D>,
        validity: Mask,
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

    /// Creates a new [`DVector<D>`] from the given [`PrecisionScale`], elements buffer, and
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
        elements: Buffer<D>,
        validity: Mask,
    ) -> Self {
        if cfg!(debug_assertions) {
            Self::try_new(ps, elements, validity).vortex_expect("Failed to create `DVector`")
        } else {
            Self {
                ps,
                elements,
                validity,
            }
        }
    }

    /// Decomposes the decimal vector into its constituent parts ([`PrecisionScale`], decimal
    /// buffer, and validity).
    pub fn into_parts(self) -> (PrecisionScale<D>, Buffer<D>, Mask) {
        (self.ps, self.elements, self.validity)
    }

    /// Get the precision/scale of the decimal vector.
    pub fn precision_scale(&self) -> PrecisionScale<D> {
        self.ps
    }

    /// Returns the precision of the decimal vector.
    pub fn precision(&self) -> u8 {
        self.ps.precision()
    }

    /// Returns the scale of the decimal vector.
    pub fn scale(&self) -> i8 {
        self.ps.scale()
    }

    /// Returns a reference to the underlying elements buffer containing the decimal data.
    pub fn elements(&self) -> &Buffer<D> {
        &self.elements
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
}

impl<D: NativeDecimalType> AsRef<[D]> for DVector<D> {
    fn as_ref(&self) -> &[D] {
        &self.elements
    }
}

impl<D: NativeDecimalType> VectorOps for DVector<D> {
    type Mutable = DVectorMut<D>;
    type Scalar = DScalar<D>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn scalar_at(&self, index: usize) -> DScalar<D> {
        assert!(index < self.len());

        let is_valid = self.validity.value(index);
        let value = is_valid.then(|| self.elements[index]);

        // SAFETY: We have already checked the validity on construction of the vector
        unsafe { DScalar::<D>::new_unchecked(self.ps, value) }
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        let elements = self.elements.slice(range.clone());
        let validity = self.validity.slice(range);
        Self {
            ps: self.ps,
            elements,
            validity,
        }
    }

    fn clear(&mut self) {
        self.elements.clear();
        self.validity.clear();
    }

    fn try_into_mut(self) -> Result<DVectorMut<D>, Self> {
        let elements = match self.elements.try_into_mut() {
            Ok(elements) => elements,
            Err(elements) => {
                return Err(Self {
                    ps: self.ps,
                    elements,
                    validity: self.validity,
                });
            }
        };

        match self.validity.try_into_mut() {
            Ok(validity_mut) => Ok(DVectorMut {
                ps: self.ps,
                elements,
                validity: validity_mut,
            }),
            Err(validity) => Err(Self {
                ps: self.ps,
                elements: elements.freeze(),
                validity,
            }),
        }
    }

    fn into_mut(self) -> DVectorMut<D> {
        DVectorMut {
            ps: self.ps,
            elements: self.elements.into_mut(),
            validity: self.validity.into_mut(),
        }
    }
}
