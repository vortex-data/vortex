// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`].

use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::{StructVectorMut, Vector, VectorMutOps, VectorOps};

/// An immutable vector of boolean values.
///
/// `StructVector` can be considered a borrowed / frozen version of [`StructVectorMut`], which is
/// created via the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// See the documentation for [`StructVectorMut`] for more information.
#[derive(Debug, Clone)]
pub struct StructVector {
    /// The fields of the `StructVector`, each stored column-wise as a [`Vector`].
    pub(super) fields: Box<[Vector]>,

    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,

    /// The length of the vector (which is the same as all field vectors).
    ///
    /// This is stored here as a convenience, and also helps in the case that the `StructVector` has
    /// no fields.
    pub(super) len: usize,
}

impl StructVector {
    /// Creates a new [`StructVector`] from the given fields and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - Any field vector has a length that does not match the length of other fields.
    /// - The validity mask length does not match the field length.
    pub fn new(fields: Box<[Vector]>, validity: Mask) -> Self {
        Self::try_new(fields, validity).vortex_expect(
            "`StructVector` fields must have matching length and validity constraints",
        )
    }

    /// Tries to create a new [`StructVector`] from the given fields and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - Any field vector has a length that does not match the length of other fields.
    /// - The validity mask length does not match the field length.
    pub fn try_new(fields: Box<[Vector]>, validity: Mask) -> VortexResult<Self> {
        let len = if fields.is_empty() {
            validity.len()
        } else {
            fields[0].len()
        };

        Self::validate(&fields, len, &validity)?;

        Ok(Self {
            fields,
            validity,
            len,
        })
    }

    /// Creates a new [`StructVector`] from the given fields and validity mask without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - All field vectors have the same length.
    /// - The validity mask has a length equal to the field length.
    pub unsafe fn new_unchecked(fields: Box<[Vector]>, validity: Mask) -> Self {
        let len = if fields.is_empty() {
            validity.len()
        } else {
            fields[0].len()
        };

        debug_assert!(
            Self::validate(&fields, len, &validity).is_ok(),
            "`StructVector` fields must have matching length and validity constraints"
        );

        Self {
            fields,
            validity,
            len,
        }
    }

    /// Validates the fields and validity mask for a [`StructVector`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - Any field vector has a length that does not match the length of other fields.
    /// - The validity mask length does not match the field length.
    fn validate(fields: &[Vector], len: usize, validity: &Mask) -> VortexResult<()> {
        // Validate that the validity mask has the correct length.
        vortex_ensure!(
            validity.len() == len,
            "Validity mask length ({}) does not match expected length ({})",
            validity.len(),
            len
        );

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

        Ok(())
    }

    /// Decomposes the struct vector into its constituent parts (fields, validity, and length).
    pub fn into_parts(self) -> (Box<[Vector]>, Mask, usize) {
        (self.fields, self.validity, self.len)
    }

    /// Returns the fields of the `StructVector`, each stored column-wise as a [`Vector`].
    pub fn fields(&self) -> &[Vector] {
        self.fields.as_ref()
    }
}

impl VectorOps for StructVector {
    type Mutable = StructVectorMut;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        let validity = match self.validity.try_into_mut() {
            Ok(validity) => validity,
            Err(validity) => {
                return Err(StructVector { validity, ..self });
            }
        };

        // Convert all of the remaining fields to mutable, if possible.
        let mut mutable_fields = Vec::with_capacity(self.fields.len());
        let mut fields_iter = self.fields.into_iter();

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

                    return Err(StructVector {
                        fields: all_fields.into_boxed_slice(),
                        len: self.len,
                        validity: validity.freeze(),
                    });
                }
            }
        }

        Ok(StructVectorMut {
            fields: mutable_fields,
            len: self.len,
            validity,
        })
    }
}
