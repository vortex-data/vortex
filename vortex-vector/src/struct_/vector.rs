// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`].

use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::{StructVectorMut, Vector, VectorMutOps, VectorOps};

/// An immutable vector of struct values.
///
/// `StructVector` can be considered a borrowed / frozen version of [`StructVectorMut`], which is
/// created via the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// See the documentation for [`StructVectorMut`] for more information.
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

impl VectorOps for StructVector {
    type Mutable = StructVectorMut;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn try_into_mut(self) -> Result<StructVectorMut, Self>
    where
        Self: Sized,
    {
        let len = self.len;

        let fields = match Arc::try_unwrap(self.fields) {
            Ok(fields) => fields,
            Err(fields) => return Err(StructVector { fields, ..self }),
        };

        let validity = match self.validity.try_into_mut() {
            Ok(validity) => validity,
            Err(validity) => {
                return Err(StructVector {
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

                    return Err(StructVector {
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
}
