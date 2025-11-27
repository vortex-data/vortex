// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`].

use std::fmt::Debug;
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
    type Scalar = StructScalar;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn scalar_at(&self, index: usize) -> StructScalar {
        assert!(index < self.len());
        StructScalar::new(self.slice(index..index + 1))
    }

    fn slice(&self, _range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        todo!()
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
