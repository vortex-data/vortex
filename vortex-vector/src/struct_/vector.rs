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
    pub(super) capacity: usize,

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
                        capacity: self.capacity,
                        validity: validity.freeze(),
                    });
                }
            }
        }

        Ok(StructVectorMut {
            fields: mutable_fields,
            len: self.len,
            capacity: self.capacity,
            validity,
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;

    use super::*;
    use crate::{BoolVectorMut, NullVector, PVectorMut, VectorMut};

    // TODO(connor): Make sure to test actual logic instead of just lengths and capacity.

    /// Helper function to create a `StructVector` with the given fields.
    fn create_struct_vector(fields: Vec<Vector>, len: usize, capacity: usize) -> StructVector {
        StructVector {
            fields: fields.into_boxed_slice(),
            len,
            capacity,
            validity: Mask::AllTrue(len),
        }
    }

    #[test]
    fn test_try_into_mut_success() {
        // Create a `StructVector` with 3 different field types (null, bool, primitive).
        let null_field: Vector = NullVector::new(5).into();
        let bool_field: Vector = BoolVectorMut::from_iter([true, false, true, false, true])
            .freeze()
            .into();
        let prim_field: Vector = PVectorMut::<i32>::from_iter([10, 20, 30, 40, 50])
            .freeze()
            .into();

        let struct_vec = create_struct_vector(vec![null_field, bool_field, prim_field], 5, 5);

        // Attempt to convert to mutable.
        let result = struct_vec.try_into_mut();

        // Should succeed since all fields have unique ownership.
        assert!(result.is_ok());

        let mut_struct = result.unwrap();

        // Verify the mutable struct has correct length and capacity.
        assert_eq!(mut_struct.len(), 5);
        assert_eq!(mut_struct.capacity(), 5);

        // Verify that we have 3 fields.
        assert_eq!(mut_struct.fields.len(), 3);

        // Verify each field has the correct type and length.
        assert!(matches!(mut_struct.fields[0], VectorMut::Null(_)));
        assert_eq!(mut_struct.fields[0].len(), 5);

        assert!(matches!(mut_struct.fields[1], VectorMut::Bool(_)));
        assert_eq!(mut_struct.fields[1].len(), 5);
        assert!(mut_struct.fields[1].capacity() >= 5);

        assert!(matches!(mut_struct.fields[2], VectorMut::Primitive(_)));
        assert_eq!(mut_struct.fields[2].len(), 5);
        assert!(mut_struct.fields[2].capacity() >= 5);
    }

    #[test]
    fn test_try_into_mut_fails_first_field() {
        // Create a bool field with shared ownership (cloned to have Arc count > 1).
        let bool_field: Vector = BoolVectorMut::from_iter([true, false, true])
            .freeze()
            .into();
        let bool_field_clone = bool_field.clone();

        let null_field: Vector = NullVector::new(3).into();
        let prim_field: Vector = PVectorMut::<i32>::from_iter([1, 2, 3]).freeze().into();

        // Create a struct with the cloned bool field as the first field.
        let struct_vec = create_struct_vector(vec![bool_field_clone, null_field, prim_field], 3, 3);

        // Attempt to convert to mutable.
        let result = struct_vec.try_into_mut();

        // Should fail since the first field has shared ownership.
        assert!(result.is_err());

        // Keep the original alive to ensure shared ownership is maintained.
        drop(bool_field);

        let recovered_struct = result.unwrap_err();

        // Verify the recovered struct has all 3 fields.
        assert_eq!(recovered_struct.fields.len(), 3);
        assert_eq!(recovered_struct.len(), 3);
        assert_eq!(recovered_struct.capacity, 3);

        // Verify field types are preserved.
        assert!(matches!(recovered_struct.fields[0], Vector::Bool(_)));
        assert!(matches!(recovered_struct.fields[1], Vector::Null(_)));
        assert!(matches!(recovered_struct.fields[2], Vector::Primitive(_)));
    }

    #[test]
    fn test_try_into_mut_fails_middle_field() {
        // Create a primitive field with shared ownership (cloned).
        let prim_field: Vector = PVectorMut::<i32>::from_iter([100, 200, 300, 400])
            .freeze()
            .into();
        let prim_field_clone = prim_field.clone();

        let bool_field: Vector = BoolVectorMut::from_iter([true, false, true, false])
            .freeze()
            .into();
        let null_field: Vector = NullVector::new(4).into();

        // Create a struct with the cloned primitive field as the middle field.
        let struct_vec = create_struct_vector(vec![bool_field, prim_field_clone, null_field], 4, 4);

        // Attempt to convert to mutable.
        let result = struct_vec.try_into_mut();

        // Should fail since the middle field has shared ownership.
        assert!(result.is_err());

        // Keep the original alive to ensure shared ownership is maintained.
        drop(prim_field);

        let recovered_struct = result.unwrap_err();

        // Verify all 3 fields are present in the recovered struct.
        assert_eq!(recovered_struct.fields.len(), 3);
        assert_eq!(recovered_struct.len(), 4);
        assert_eq!(recovered_struct.capacity, 4);

        // Verify field types and order are preserved.
        // The first field was converted to mutable then frozen back, so it should still be bool.
        assert!(matches!(recovered_struct.fields[0], Vector::Bool(_)));
        assert!(matches!(recovered_struct.fields[1], Vector::Primitive(_)));
        assert!(matches!(recovered_struct.fields[2], Vector::Null(_)));
    }

    #[test]
    fn test_try_into_mut_fails_last_field() {
        // Create a null field with "shared ownership" (though NullVector always succeeds, we test
        // the pattern by cloning a bool vector).
        let bool_field_last: Vector = BoolVectorMut::from_iter([true, true]).freeze().into();
        let bool_field_last_clone = bool_field_last.clone();

        let null_field: Vector = NullVector::new(2).into();
        let prim_field: Vector = PVectorMut::<u64>::from_iter([1000, 2000]).freeze().into();

        // Create a struct with the cloned bool field as the last field.
        let struct_vec =
            create_struct_vector(vec![null_field, prim_field, bool_field_last_clone], 2, 2);

        // Attempt to convert to mutable.
        let result = struct_vec.try_into_mut();

        // Should fail since the last field has shared ownership.
        assert!(result.is_err());

        // Keep the original alive to ensure shared ownership is maintained.
        drop(bool_field_last);

        let recovered_struct = result.unwrap_err();

        // Verify all 3 fields are present.
        assert_eq!(recovered_struct.fields.len(), 3);
        assert_eq!(recovered_struct.len(), 2);
        assert_eq!(recovered_struct.capacity, 2);

        // Verify field types are preserved.
        // The first two fields were converted to mutable then frozen back.
        assert!(matches!(recovered_struct.fields[0], Vector::Null(_)));
        assert!(matches!(recovered_struct.fields[1], Vector::Primitive(_)));
        assert!(matches!(recovered_struct.fields[2], Vector::Bool(_)));
    }

    #[test]
    fn test_try_into_mut_nested_struct() {
        // Create a nested struct: a `StructVector` containing other `StructVector`s as fields.
        let inner_struct1: Vector = create_struct_vector(
            vec![
                NullVector::new(3).into(),
                BoolVectorMut::from_iter([true, false, true])
                    .freeze()
                    .into(),
            ],
            3,
            3,
        )
        .into();

        let inner_struct2: Vector = create_struct_vector(
            vec![PVectorMut::<i32>::from_iter([1, 2, 3]).freeze().into()],
            3,
            3,
        )
        .into();

        let outer_struct = create_struct_vector(vec![inner_struct1, inner_struct2], 3, 3);

        // Attempt to convert to mutable.
        let result = outer_struct.try_into_mut();

        // Should succeed once `StructVectorMut` is fully implemented.
        assert!(result.is_ok());

        let mut_struct = result.unwrap();
        assert_eq!(mut_struct.len(), 3);
        assert_eq!(mut_struct.fields.len(), 2);

        // Verify nested structs are converted correctly.
        assert!(matches!(mut_struct.fields[0], VectorMut::Struct(_)));
        assert!(matches!(mut_struct.fields[1], VectorMut::Struct(_)));
    }
}
