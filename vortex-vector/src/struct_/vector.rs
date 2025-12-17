// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`].

use std::fmt::Debug;
use std::ops::BitAnd;
use std::ops::RangeBounds;
use std::sync::Arc;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::Vector;
use crate::VectorMutOps;
use crate::VectorOps;
use crate::struct_::StructScalar;
use crate::struct_::StructVectorMut;

/// An immutable vector of struct values.
///
/// Struct values are stored column-wise in the vector, so values in the same field are stored next
/// to each other (rather than values in the same struct stored next to each other).
#[derive(Debug, Clone)]
pub struct StructVector {
    /// The fields of the `StructVector`, each stored column-wise as a [`Vector`].
    ///
    /// We store these as an [`Arc<Box<_>>`] because we need to call [`try_unwrap()`] in our
    /// [`try_into_mut()`] implementation, and since slices are unsized it is not implemented for
    /// [`Arc<[Vector]>`].
    ///
    /// [`try_unwrap()`]: Arc::try_unwrap
    /// [`try_into_mut()`]: Self::try_into_mut
    pub(super) fields: Arc<Box<[Vector]>>,

    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,

    /// The length of the vector (which is the same as all field vectors).
    ///
    /// This is stored here as a convenience, as the validity also tracks this information.
    pub(super) len: usize,
}

impl PartialEq for StructVector {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        // Number of fields must match
        if self.fields.len() != other.fields.len() {
            return false;
        }
        // Validity patterns must match
        if self.validity != other.validity {
            return false;
        }
        // For each field pair: clone the fields, call mask_validity(&combined_mask) on both clones
        // where combined_mask = self.validity AND other.validity, then compare with ==
        let combined_mask = self.validity.bitand(&other.validity);

        // Each field must match with the combined mask applied
        for (self_field, other_field) in self.fields.iter().zip(other.fields.iter()) {
            let mut self_field_masked = self_field.clone();
            let mut other_field_masked = other_field.clone();
            self_field_masked.mask_validity(&combined_mask);
            other_field_masked.mask_validity(&combined_mask);

            if self_field_masked != other_field_masked {
                return false;
            }
        }
        true
    }
}

impl StructVector {
    /// Creates a new [`StructVector`] from the given fields and validity mask.
    ///
    /// Note that we take [`Arc<Box<[_]>>`] in order to enable easier conversion to
    /// [`StructVectorMut`] via [`try_into_mut()`](Self::try_into_mut).
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - Any field vector has a length that does not match the length of other fields.
    /// - The validity mask length does not match the field length.
    pub fn new(fields: Arc<Box<[Vector]>>, validity: Mask) -> Self {
        Self::try_new(fields, validity).vortex_expect("Failed to create `StructVector`")
    }

    /// Tries to create a new [`StructVector`] from the given fields and validity mask.
    ///
    /// Note that we take [`Arc<Box<[_]>>`] in order to enable easier conversion to
    /// [`StructVectorMut`] via [`try_into_mut()`](Self::try_into_mut).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - Any field vector has a length that does not match the length of other fields.
    /// - The validity mask length does not match the field length.
    pub fn try_new(fields: Arc<Box<[Vector]>>, validity: Mask) -> VortexResult<Self> {
        let len = validity.len();

        // Validate that all fields have the correct length.
        for (i, field) in fields.iter().enumerate() {
            vortex_ensure!(
                field.len() == len,
                "Field {} has length {} but expected length {}",
                i,
                field.len(),
                len
            );
        }

        Ok(Self {
            fields,
            validity,
            len,
        })
    }

    /// Creates a new [`StructVector`] from the given fields and validity mask without validation.
    ///
    /// Note that we take [`Arc<Box<[_]>>`] in order to enable easier conversion to
    /// [`StructVectorMut`] via [`try_into_mut()`](Self::try_into_mut).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - All field vectors have the same length.
    /// - The validity mask has a length equal to the field length.
    pub unsafe fn new_unchecked(fields: Arc<Box<[Vector]>>, validity: Mask) -> Self {
        let len = validity.len();

        if cfg!(debug_assertions) {
            Self::new(fields, validity)
        } else {
            Self {
                fields,
                validity,
                len,
            }
        }
    }

    /// Decomposes the struct vector into its constituent parts (fields and validity).
    pub fn into_parts(self) -> (Arc<Box<[Vector]>>, Mask) {
        (self.fields, self.validity)
    }

    /// Returns the fields of the `StructVector`, each stored column-wise as a [`Vector`].
    pub fn fields(&self) -> &Arc<Box<[Vector]>> {
        &self.fields
    }
}

impl Eq for StructVector {}

impl VectorOps for StructVector {
    type Mutable = StructVectorMut;
    type Scalar = StructScalar;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn mask_validity(&mut self, mask: &Mask) {
        self.validity = self.validity.bitand(mask);
    }

    fn scalar_at(&self, index: usize) -> StructScalar {
        assert!(index < self.len());
        StructScalar::new(self.slice(index..index + 1))
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        let sliced_fields: Box<[_]> = self
            .fields
            .iter()
            .map(|field| field.slice(range.clone()))
            .collect();

        let sliced_validity = self.validity.slice(range);
        let len = sliced_validity.len();

        StructVector {
            fields: Arc::new(sliced_fields),
            validity: sliced_validity,
            len,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.validity.clear();
        Arc::make_mut(&mut self.fields)
            .iter_mut()
            .for_each(|f| f.clear());
    }

    fn try_into_mut(self) -> Result<StructVectorMut, Self> {
        let len = self.len;

        let fields = match Arc::try_unwrap(self.fields) {
            Ok(fields) => fields,
            Err(fields) => return Err(Self { fields, ..self }),
        };

        let validity = match self.validity.try_into_mut() {
            Ok(validity) => validity,
            Err(validity) => {
                return Err(Self {
                    fields: Arc::new(fields),
                    validity,
                    len,
                });
            }
        };

        // Convert all the remaining fields to mutable, if possible.
        let mut mutable_fields = Vec::with_capacity(fields.len());
        let mut fields_iter = fields.into_iter();

        while let Some(field) = fields_iter.next() {
            match field.try_into_mut() {
                Ok(mutable_field) => {
                    // We were able to take ownership of the field vector, so add it and keep going.
                    mutable_fields.push(mutable_field);
                }
                Err(immutable_field) => {
                    // We were unable to take ownership, so we must re-freeze all of the fields
                    // vectors we took ownership over and reconstruct the original `StructVector`.
                    let mut all_fields: Vec<Vector> = mutable_fields
                        .into_iter()
                        .map(|mut_field| mut_field.freeze())
                        .collect();

                    all_fields.push(immutable_field);
                    all_fields.extend(fields_iter);

                    return Err(Self {
                        fields: Arc::new(all_fields.into_boxed_slice()),
                        len: self.len,
                        validity: validity.freeze(),
                    });
                }
            }
        }

        Ok(StructVectorMut {
            fields: mutable_fields.into_boxed_slice(),
            len: self.len,
            validity,
        })
    }

    fn into_mut(self) -> StructVectorMut {
        let len = self.len;
        let validity = self.validity.into_mut();

        // If someone else has a strong reference to the `Arc`, clone the underlying data (which is
        // just a **different** reference count increment).
        let fields = Arc::try_unwrap(self.fields).unwrap_or_else(|arc| (*arc).clone());

        let mutable_fields: Box<[_]> = fields
            .into_vec()
            .into_iter()
            .map(|field| field.into_mut())
            .collect();

        StructVectorMut {
            fields: mutable_fields,
            len,
            validity,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_mask::Mask;

    use super::*;
    use crate::bool::BoolVectorMut;
    use crate::null::NullVector;
    use crate::primitive::PVectorMut;

    #[test]
    fn test_struct_vector_eq_identical() {
        // Two identical struct vectors should be equal.
        let v1 = StructVector::new(
            Arc::new(Box::new([
                NullVector::new(3).into(),
                BoolVectorMut::from_iter([true, false, true])
                    .freeze()
                    .into(),
                PVectorMut::<i32>::from_iter([10, 20, 30]).freeze().into(),
            ])),
            Mask::AllTrue(3),
        );

        let v2 = StructVector::new(
            Arc::new(Box::new([
                NullVector::new(3).into(),
                BoolVectorMut::from_iter([true, false, true])
                    .freeze()
                    .into(),
                PVectorMut::<i32>::from_iter([10, 20, 30]).freeze().into(),
            ])),
            Mask::AllTrue(3),
        );

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_different_length() {
        // Struct vectors with different lengths should not be equal.
        let v1 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 20, 30])
                .freeze()
                .into()])),
            Mask::AllTrue(3),
        );

        let v2 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 20])
                .freeze()
                .into()])),
            Mask::AllTrue(2),
        );

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_different_field_count() {
        // Struct vectors with different number of fields should not be equal.
        let v1 = StructVector::new(
            Arc::new(Box::new([
                PVectorMut::<i32>::from_iter([10, 20, 30]).freeze().into(),
                BoolVectorMut::from_iter([true, false, true])
                    .freeze()
                    .into(),
            ])),
            Mask::AllTrue(3),
        );

        let v2 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 20, 30])
                .freeze()
                .into()])),
            Mask::AllTrue(3),
        );

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_different_validity() {
        // Struct vectors with different validity patterns should not be equal.
        let v1 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 20, 30])
                .freeze()
                .into()])),
            Mask::AllTrue(3),
        );

        let v2 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 20, 30])
                .freeze()
                .into()])),
            Mask::from_iter([true, false, true]),
        );

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_different_field_values() {
        // Struct vectors with different field values should not be equal.
        let v1 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 20, 30])
                .freeze()
                .into()])),
            Mask::AllTrue(3),
        );

        let v2 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 99, 30])
                .freeze()
                .into()])),
            Mask::AllTrue(3),
        );

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_ignores_invalid_positions() {
        // Two struct vectors with different values at invalid positions should be equal
        // as long as they have the same validity pattern and same values at valid positions.
        //
        // validity = [true, false, true] means position 1 is invalid
        let validity = Mask::from_iter([true, false, true]);

        let v1 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 20, 30])
                .freeze()
                .into()])),
            validity.clone(),
        );

        // Different value at position 1 (which is invalid)
        let v2 = StructVector::new(
            Arc::new(Box::new([PVectorMut::<i32>::from_iter([10, 99, 30])
                .freeze()
                .into()])),
            validity,
        );

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_combined_mask_applied() {
        // Test that the combined mask (self.validity AND other.validity) is applied.
        // Both vectors have the same validity, so the combined mask equals that validity.
        //
        // validity = [true, false, true, false, true] means positions 1,3 are invalid
        let validity = Mask::from_iter([true, false, true, false, true]);

        let v1 = StructVector::new(
            Arc::new(Box::new([
                PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5])
                    .freeze()
                    .into(),
                BoolVectorMut::from_iter([true, true, true, true, true])
                    .freeze()
                    .into(),
            ])),
            validity.clone(),
        );

        // Different values at invalid positions (1 and 3)
        let v2 = StructVector::new(
            Arc::new(Box::new([
                PVectorMut::<i32>::from_iter([1, 999, 3, 888, 5])
                    .freeze()
                    .into(),
                BoolVectorMut::from_iter([true, false, true, false, true])
                    .freeze()
                    .into(),
            ])),
            validity,
        );

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_nested() {
        // Test equality with nested struct vectors.
        let inner1 = StructVector::new(
            Arc::new(Box::new([BoolVectorMut::from_iter([true, false, true])
                .freeze()
                .into()])),
            Mask::AllTrue(3),
        );

        let inner2 = StructVector::new(
            Arc::new(Box::new([BoolVectorMut::from_iter([true, false, true])
                .freeze()
                .into()])),
            Mask::AllTrue(3),
        );

        let v1 = StructVector::new(Arc::new(Box::new([inner1.into()])), Mask::AllTrue(3));

        let v2 = StructVector::new(Arc::new(Box::new([inner2.into()])), Mask::AllTrue(3));

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_struct_vector_eq_empty() {
        // Two empty struct vectors should be equal.
        let v1 = StructVector::new(Arc::new(Box::new([])), Mask::AllTrue(0));
        let v2 = StructVector::new(Arc::new(Box::new([])), Mask::AllTrue(0));

        assert_eq!(v1, v2);
    }
}
