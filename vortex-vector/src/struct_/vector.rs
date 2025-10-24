// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`].

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

    /// The length of the vector (which is the same as all field vectors).
    ///
    /// This is stored here as a convenience, and also helps in the case that the `StructVector` has
    /// no fields.
    pub(super) len: usize,

    /// The capacity of the vector (which is the less than or equal to the capacity of all field
    /// vectors).
    ///
    /// This is stored here as a convenience for converting to/from a [`StructVectorMut`], and also
    /// helps in the case that the `StructVector` has no fields.
    pub(super) minimum_capacity: usize,

    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
}

impl StructVector {
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
                        minimum_capacity: self.minimum_capacity,
                        validity: validity.freeze(),
                    });
                }
            }
        }

        Ok(StructVectorMut {
            fields: mutable_fields,
            len: self.len,
            minimum_capacity: self.minimum_capacity,
            validity,
        })
    }
}
