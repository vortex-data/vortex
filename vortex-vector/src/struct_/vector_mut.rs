// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVectorMut`].

use vortex_error::{VortexResult, vortex_ensure, vortex_panic};
use vortex_mask::MaskMut;

use crate::{StructVector, Vector, VectorMut, VectorMutOps, VectorOps};

/// A mutable vector of struct values (values with named fields).
///
/// Struct values are stored column-wise in the vector, so values in the same field are stored next
/// to each other (rather than values in the same struct stored next to each other).
///
/// TODO examples.
#[derive(Debug, Clone)]
pub struct StructVectorMut {
    /// The fields of the `StructVectorMut`, each stored column-wise as a [`VectorMut`].
    ///
    /// We store this as a mutable vector instead of a fixed-sized type since vectors do not have an
    /// associated [`DType`](vortex_dtype::DType), thus users can add field columns if they need.
    pub(super) fields: Vec<VectorMut>,

    /// The length of the vector (which is the same as all field vectors).
    ///
    /// This is stored here as a convenience, and also helps in the case that the `StructVector` has
    /// no fields.
    pub(super) len: usize,

    /// The capacity of the vector (which is the less than or equal to the capacity of all field
    /// vectors).
    ///
    /// This is stored here as a convenience, and also helps in the case that the `StructVector` has
    /// no fields.
    pub(super) minimum_capacity: usize,

    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: MaskMut,
}

impl StructVectorMut {
    /// Creates a new [`StructVectorMut`] with the given fields, length, and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any field vector has a length that does not match `len`
    /// - The validity mask length does not match `len`
    pub fn try_new(fields: Vec<VectorMut>, len: usize, validity: MaskMut) -> VortexResult<Self> {
        // Validate that the validity mask has the correct length.
        vortex_ensure!(
            validity.len() == len,
            "Validity mask length ({}) does not match expected length ({})",
            validity.len(),
            len
        );

        // Validate that all fields have the correct length and compute the minimum capacity.
        let mut minimum_capacity = usize::MAX;
        for (i, field) in fields.iter().enumerate() {
            vortex_ensure!(
                field.len() == len,
                "Field {} has length {} but expected length {}",
                i,
                field.len(),
                len
            );

            minimum_capacity = minimum_capacity.min(field.capacity());
        }

        // Handle the case where there are no fields.
        if fields.is_empty() {
            minimum_capacity = len;
        }

        debug_assert!(minimum_capacity >= len);

        Ok(Self {
            fields,
            len,
            minimum_capacity,
            validity,
        })
    }

    /// Returns the fields of the `StructVectorMut`, each stored column-wise as a [`VectorMut`].
    pub fn fields_mut(&mut self) -> &mut [VectorMut] {
        self.fields.as_mut_slice()
    }
}

impl VectorMutOps for StructVectorMut {
    type Immutable = StructVector;

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.minimum_capacity
    }

    fn reserve(&mut self, additional: usize) {
        // Note that the `capacity` stored in `self` is just a lower bound on all fields, so it does
        // not need to be exactly the minimum capacity of all fields.
        self.minimum_capacity = self.minimum_capacity.max(self.len + additional);

        // Reserve the additional capacity in each field vector.
        for field in &mut self.fields {
            field.reserve(additional);

            debug_assert_eq!(
                field.len(),
                self.len,
                "Field length must match `StructVectorMut` length"
            );
            debug_assert!(
                field.capacity() >= self.minimum_capacity,
                "Field capacity must be at least the `StructVectorMut` capacity"
            );
        }

        self.validity.reserve(additional);
    }

    fn extend_from_vector(&mut self, other: &StructVector) {
        assert_eq!(
            self.fields.len(),
            other.fields().len(),
            "Cannot extend StructVectorMut: field count mismatch ({} vs {})",
            self.fields.len(),
            other.fields().len()
        );

        // Extend each field from the corresponding field in `other`.
        let pairs = self.fields.iter_mut().zip(other.fields());
        for (self_mut_vector, other_vec) in pairs {
            match (self_mut_vector, other_vec) {
                (VectorMut::Null(a), Vector::Null(b)) => a.extend_from_vector(b),
                (VectorMut::Bool(a), Vector::Bool(b)) => a.extend_from_vector(b),
                (VectorMut::Primitive(a), Vector::Primitive(b)) => a.extend_from_vector(b),
                (VectorMut::Struct(a), Vector::Struct(b)) => a.extend_from_vector(b),
                _ => {
                    vortex_panic!("Mismatched field types in `StructVectorMut::extend_from_vector`")
                }
            }
        }

        // Extend the validity mask.
        self.validity.append_mask(other.validity());

        self.len += other.len();

        // Note that the `capacity` stored in `self` is just a lower bound on all fields, so it does
        // not need to be exactly the minimum capacity of all fields.
        self.minimum_capacity = self.minimum_capacity.max(self.len);
    }

    fn append_nulls(&mut self, n: usize) {
        for field in &mut self.fields {
            field.append_nulls(n); // Note that the value we push to each doesn't actually matter.
        }
        self.validity.append_n(false, n);

        self.len += n;
        self.minimum_capacity = self.minimum_capacity.max(self.len);
    }

    fn freeze(self) -> Self::Immutable {
        let frozen_fields: Vec<Vector> = self
            .fields
            .into_iter()
            .map(|mut_field| mut_field.freeze())
            .collect();

        StructVector {
            fields: frozen_fields.into_boxed_slice(),
            len: self.len,
            minimum_capacity: self.minimum_capacity,
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        assert!(
            at <= self.capacity(),
            "split_off out of bounds: {} > {}",
            at,
            self.capacity()
        );

        // Split off each field vector.
        let split_fields: Vec<VectorMut> = self
            .fields
            .iter_mut()
            .map(|field| field.split_off(at))
            .collect();

        // Split off the validity mask.
        let split_validity = self.validity.split_off(at);

        // Compute the minimum capacity of the split-off vector.
        let mut split_minimum_capacity = usize::MAX;
        for field in &split_fields {
            split_minimum_capacity = split_minimum_capacity.min(field.capacity());
        }

        // Handle the case where there are no fields.
        if split_fields.is_empty() {
            split_minimum_capacity = self.len.saturating_sub(at);
        }

        // Update self's state.
        let split_len = self.len.saturating_sub(at);
        self.len = at;
        self.minimum_capacity = self.minimum_capacity.min(at);

        Self {
            fields: split_fields,
            len: split_len,
            minimum_capacity: split_minimum_capacity,
            validity: split_validity,
        }
    }

    fn unsplit(&mut self, other: Self) {
        assert_eq!(
            self.fields.len(),
            other.fields.len(),
            "Cannot unsplit StructVectorMut: field count mismatch ({} vs {})",
            self.fields.len(),
            other.fields.len()
        );

        // Unsplit each field with the corresponding field in `other`.
        let pairs = self.fields.iter_mut().zip(other.fields);
        for (self_mut_vector, other_mut_vec) in pairs {
            match (self_mut_vector, other_mut_vec) {
                (VectorMut::Null(a), VectorMut::Null(b)) => a.unsplit(b),
                (VectorMut::Bool(a), VectorMut::Bool(b)) => a.unsplit(b),
                (VectorMut::Primitive(a), VectorMut::Primitive(b)) => a.unsplit(b),
                (VectorMut::Struct(a), VectorMut::Struct(b)) => a.unsplit(b),
                _ => {
                    vortex_panic!("Mismatched field types in `StructVectorMut::unsplit`")
                }
            }
        }

        // Unsplit the validity mask.
        self.validity.unsplit(other.validity);

        // Update length and capacity.
        self.len += other.len;
        self.minimum_capacity = self.minimum_capacity.max(self.len);
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
            minimum_capacity: capacity,
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
        assert_eq!(recovered_struct.minimum_capacity, 3);

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
        assert_eq!(recovered_struct.minimum_capacity, 4);

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
        assert_eq!(recovered_struct.minimum_capacity, 2);

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

    #[test]
    fn test_split_off_basic() {
        // Create a `StructVectorMut` with 3 different field types.
        let null_field: VectorMut = NullVector::new(10).try_into_mut().unwrap().into();
        let bool_field: VectorMut = BoolVectorMut::from_iter([
            true, false, true, false, true, false, true, false, true, false,
        ])
        .into();
        let prim_field: VectorMut =
            PVectorMut::<i32>::from_iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).into();

        let mut struct_vec = StructVectorMut::try_new(
            vec![null_field, bool_field, prim_field],
            10,
            MaskMut::new_true(10),
        )
        .unwrap();

        // Split at position 6.
        let second_half = struct_vec.split_off(6);

        // Verify the first half.
        assert_eq!(struct_vec.len(), 6);
        assert_eq!(struct_vec.fields.len(), 3);
        assert!(struct_vec.capacity() >= 6);

        // Verify the second half.
        assert_eq!(second_half.len(), 4);
        assert_eq!(second_half.fields.len(), 3);
        assert!(second_half.capacity() >= 4);

        // Verify field types are preserved in both halves.
        assert!(matches!(struct_vec.fields[0], VectorMut::Null(_)));
        assert!(matches!(struct_vec.fields[1], VectorMut::Bool(_)));
        assert!(matches!(struct_vec.fields[2], VectorMut::Primitive(_)));

        assert!(matches!(second_half.fields[0], VectorMut::Null(_)));
        assert!(matches!(second_half.fields[1], VectorMut::Bool(_)));
        assert!(matches!(second_half.fields[2], VectorMut::Primitive(_)));

        // Verify field lengths.
        assert_eq!(struct_vec.fields[0].len(), 6);
        assert_eq!(struct_vec.fields[1].len(), 6);
        assert_eq!(struct_vec.fields[2].len(), 6);

        assert_eq!(second_half.fields[0].len(), 4);
        assert_eq!(second_half.fields[1].len(), 4);
        assert_eq!(second_half.fields[2].len(), 4);
    }

    #[test]
    fn test_split_off_boundaries() {
        // Create a small struct vector.
        let null_field: VectorMut = NullVector::new(5).try_into_mut().unwrap().into();
        let bool_field: VectorMut =
            BoolVectorMut::from_iter([true, false, true, false, true]).into();

        let mut struct_vec =
            StructVectorMut::try_new(vec![null_field, bool_field], 5, MaskMut::new_true(5))
                .unwrap();

        // Split at 0 (empty first half).
        let all_elements = struct_vec.split_off(0);
        assert_eq!(struct_vec.len(), 0);
        assert_eq!(all_elements.len(), 5);

        // Create another struct vector for the second boundary test.
        let null_field2: VectorMut = NullVector::new(5).try_into_mut().unwrap().into();
        let bool_field2: VectorMut =
            BoolVectorMut::from_iter([true, false, true, false, true]).into();

        let mut struct_vec2 =
            StructVectorMut::try_new(vec![null_field2, bool_field2], 5, MaskMut::new_true(5))
                .unwrap();

        // Split at length (empty second half).
        let empty_second = struct_vec2.split_off(5);
        assert_eq!(struct_vec2.len(), 5);
        assert_eq!(empty_second.len(), 0);
    }

    #[test]
    fn test_unsplit_contiguous() {
        // Create a `StructVectorMut` with multiple field types.
        let null_field: VectorMut = NullVector::new(8).try_into_mut().unwrap().into();
        let bool_field: VectorMut =
            BoolVectorMut::from_iter([true, false, true, false, true, false, true, false]).into();
        let prim_field: VectorMut =
            PVectorMut::<i64>::from_iter([10, 20, 30, 40, 50, 60, 70, 80]).into();

        let mut struct_vec = StructVectorMut::try_new(
            vec![null_field, bool_field, prim_field],
            8,
            MaskMut::new_true(8),
        )
        .unwrap();

        // Split at position 5.
        let second_half = struct_vec.split_off(5);

        // Verify the split.
        assert_eq!(struct_vec.len(), 5);
        assert_eq!(second_half.len(), 3);

        // Unsplit to rejoin.
        struct_vec.unsplit(second_half);

        // Verify the result.
        assert_eq!(struct_vec.len(), 8);
        assert_eq!(struct_vec.fields.len(), 3);
        assert!(struct_vec.capacity() >= 8);

        // Verify field types and lengths.
        assert!(matches!(struct_vec.fields[0], VectorMut::Null(_)));
        assert!(matches!(struct_vec.fields[1], VectorMut::Bool(_)));
        assert!(matches!(struct_vec.fields[2], VectorMut::Primitive(_)));

        assert_eq!(struct_vec.fields[0].len(), 8);
        assert_eq!(struct_vec.fields[1].len(), 8);
        assert_eq!(struct_vec.fields[2].len(), 8);
    }

    #[test]
    fn test_unsplit_noncontiguous() {
        // Create the first struct vector.
        let null_field1: VectorMut = NullVector::new(3).try_into_mut().unwrap().into();
        let bool_field1: VectorMut = BoolVectorMut::from_iter([true, false, true]).into();

        let mut struct_vec1 =
            StructVectorMut::try_new(vec![null_field1, bool_field1], 3, MaskMut::new_true(3))
                .unwrap();

        // Create a separate struct vector.
        let null_field2: VectorMut = NullVector::new(2).try_into_mut().unwrap().into();
        let bool_field2: VectorMut = BoolVectorMut::from_iter([false, true]).into();

        let struct_vec2 =
            StructVectorMut::try_new(vec![null_field2, bool_field2], 2, MaskMut::new_true(2))
                .unwrap();

        // Unsplit to join them.
        struct_vec1.unsplit(struct_vec2);

        // Verify the result.
        assert_eq!(struct_vec1.len(), 5);
        assert_eq!(struct_vec1.fields.len(), 2);

        // Verify field types and lengths.
        assert!(matches!(struct_vec1.fields[0], VectorMut::Null(_)));
        assert!(matches!(struct_vec1.fields[1], VectorMut::Bool(_)));

        assert_eq!(struct_vec1.fields[0].len(), 5);
        assert_eq!(struct_vec1.fields[1].len(), 5);
    }

    #[test]
    fn test_split_unsplit_nested() {
        // Create nested structs.
        let inner1_null: VectorMut = NullVector::new(4).try_into_mut().unwrap().into();
        let inner1_bool: VectorMut = BoolVectorMut::from_iter([true, false, true, false]).into();
        let inner_struct1: VectorMut =
            StructVectorMut::try_new(vec![inner1_null, inner1_bool], 4, MaskMut::new_true(4))
                .unwrap()
                .into();

        let inner2_prim: VectorMut = PVectorMut::<u32>::from_iter([100, 200, 300, 400]).into();
        let inner_struct2: VectorMut =
            StructVectorMut::try_new(vec![inner2_prim], 4, MaskMut::new_true(4))
                .unwrap()
                .into();

        let mut outer_struct =
            StructVectorMut::try_new(vec![inner_struct1, inner_struct2], 4, MaskMut::new_true(4))
                .unwrap();

        // Split the outer struct.
        let second_half = outer_struct.split_off(2);

        // Verify the split.
        assert_eq!(outer_struct.len(), 2);
        assert_eq!(second_half.len(), 2);
        assert_eq!(outer_struct.fields.len(), 2);
        assert_eq!(second_half.fields.len(), 2);

        // Verify nested structure is preserved.
        assert!(matches!(outer_struct.fields[0], VectorMut::Struct(_)));
        assert!(matches!(outer_struct.fields[1], VectorMut::Struct(_)));
        assert!(matches!(second_half.fields[0], VectorMut::Struct(_)));
        assert!(matches!(second_half.fields[1], VectorMut::Struct(_)));

        // Unsplit to rejoin.
        outer_struct.unsplit(second_half);

        // Verify the result.
        assert_eq!(outer_struct.len(), 4);
        assert_eq!(outer_struct.fields.len(), 2);
        assert!(matches!(outer_struct.fields[0], VectorMut::Struct(_)));
        assert!(matches!(outer_struct.fields[1], VectorMut::Struct(_)));
    }

    #[test]
    fn test_split_off_empty_fields() {
        // Create a `StructVectorMut` with no fields.
        let mut struct_vec = StructVectorMut::try_new(vec![], 10, MaskMut::new_true(10)).unwrap();

        // Split at position 6.
        let second_half = struct_vec.split_off(6);

        // Verify the split.
        assert_eq!(struct_vec.len(), 6);
        assert_eq!(second_half.len(), 4);
        assert_eq!(struct_vec.fields.len(), 0);
        assert_eq!(second_half.fields.len(), 0);

        // Verify capacity handling for empty fields case.
        assert_eq!(struct_vec.capacity(), 6);
        assert_eq!(second_half.capacity(), 4);
    }
}
