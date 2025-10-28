// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVectorMut`].

use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_panic};
use vortex_mask::MaskMut;

use crate::{StructVector, Vector, VectorMut, VectorMutOps, VectorOps};

/// A mutable vector of struct values (values with named fields).
///
/// Struct values are stored column-wise in the vector, so values in the same field are stored next
/// to each other (rather than values in the same struct stored next to each other).
///
/// # Examples
///
/// ## Creating a [`StructVector`] and [`StructVectorMut`]
///
/// ```
/// use vortex_vector::{
///     BoolVectorMut, NullVectorMut, PVectorMut, StructVectorMut, VectorMut, VectorMutOps,
/// };
/// use vortex_mask::MaskMut;
///
/// // Create a struct with three fields: nulls, booleans, and integers.
/// let fields = vec![
///     NullVectorMut::new(3).into(),
///     BoolVectorMut::from_iter([true, false, true]).into(),
///     PVectorMut::<i32>::from_iter([10, 20, 30]).into(),
/// ];
///
/// let mut struct_vec = StructVectorMut::new(fields, MaskMut::new_true(3));
/// assert_eq!(struct_vec.len(), 3);
/// ```
///
/// ## Working with [`split_off()`] and [`unsplit()`]
///
/// [`split_off()`]: VectorMutOps::split_off
/// [`unsplit()`]: VectorMutOps::unsplit
///
/// ```
/// use vortex_vector::{
///     BoolVectorMut, NullVectorMut, PVectorMut, StructVectorMut, VectorMut, VectorMutOps,
/// };
/// use vortex_mask::MaskMut;
///
/// let fields = vec![
///     NullVectorMut::new(6).into(),
///     PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6]).into(),
/// ];
///
/// let mut struct_vec = StructVectorMut::new(fields, MaskMut::new_true(6));
///
/// // Split at position 4.
/// let second_part = struct_vec.split_off(4);
///
/// assert_eq!(struct_vec.len(), 4);
/// assert_eq!(second_part.len(), 2);
///
/// // Rejoin the parts.
/// struct_vec.unsplit(second_part);
/// assert_eq!(struct_vec.len(), 6);
/// ```
///
/// ## Accessing field values
///
/// ```
/// use vortex_vector::{
///     BoolVectorMut, NullVectorMut, PVectorMut, StructVectorMut, VectorMut, VectorMutOps,
/// };
/// use vortex_mask::MaskMut;
/// use vortex_dtype::PTypeDowncast;
///
/// let fields = vec![
///     NullVectorMut::new(3).into(),
///     BoolVectorMut::from_iter([true, false, true]).into(),
///     PVectorMut::<i32>::from_iter([10, 20, 30]).into(),
/// ];
///
/// let struct_vec = StructVectorMut::new(fields, MaskMut::new_true(3));
///
/// // Access the boolean field vector (field index 1).
/// if let VectorMut::Bool(bool_vec) = struct_vec.fields()[1].clone() {
///     let values: Vec<_> = bool_vec.into_iter().map(|v| v.unwrap()).collect();
///     assert_eq!(values, vec![true, false, true]);
/// }
///
/// // Access the integer field column (field index 2).
/// if let VectorMut::Primitive(prim_vec) = struct_vec.fields()[2].clone() {
///     let values: Vec<_> = prim_vec.into_i32().into_iter().map(|v| v.unwrap()).collect();
///     assert_eq!(values, vec![10, 20, 30]);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct StructVectorMut {
    /// The fields of the `StructVectorMut`, each stored column-wise as a [`VectorMut`].
    ///
    /// We store this as a mutable vector instead of a fixed-sized type since vectors do not have an
    /// associated [`DType`](vortex_dtype::DType), thus users can add field columns if they need.
    pub(super) fields: Vec<VectorMut>,

    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: MaskMut,

    /// The length of the vector (which is the same as all field vectors).
    ///
    /// This is stored here as a convenience, and also helps in the case that the `StructVector` has
    /// no fields.
    pub(super) len: usize,
}

impl StructVectorMut {
    /// Creates a new [`StructVectorMut`] with the given fields and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - Any field vector has a length that does not match the length of other fields.
    /// - The validity mask length does not match the field length.
    pub fn new(fields: Vec<VectorMut>, validity: MaskMut) -> Self {
        Self::try_new(fields, validity).vortex_expect("Failed to create `StructVectorMut`")
    }

    /// Tries to create a new [`StructVectorMut`] with the given fields and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - Any field vector has a length that does not match the length of other fields.
    /// - The validity mask length does not match the field length.
    pub fn try_new(fields: Vec<VectorMut>, validity: MaskMut) -> VortexResult<Self> {
        let len = if fields.is_empty() {
            validity.len()
        } else {
            fields[0].len()
        };

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

        Ok(Self {
            fields,
            validity,
            len,
        })
    }

    /// Creates a new [`StructVectorMut`] with the given fields and validity mask without
    /// validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - All field vectors have the same length.
    /// - The validity mask has a length equal to the field length.
    pub unsafe fn new_unchecked(fields: Vec<VectorMut>, validity: MaskMut) -> Self {
        let len = if fields.is_empty() {
            validity.len()
        } else {
            fields[0].len()
        };

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

    /// Decomposes the struct vector into its constituent parts (fields, validity, and length).
    pub fn into_parts(self) -> (Vec<VectorMut>, MaskMut, usize) {
        (self.fields, self.validity, self.len)
    }

    /// Returns the fields of the `StructVectorMut`, each stored column-wise as a [`VectorMut`].
    pub fn fields(&self) -> &[VectorMut] {
        self.fields.as_slice()
    }

    /// Finds the minimum capacity of all field vectors.
    ///
    /// This is equal to the maximum amount of scalars we can add before we need to reallocate at
    /// least one of the child field vectors.
    ///
    /// If there are no fields, this returns the length of the vector.
    ///
    /// Note that this takes time in `O(f)`, where `f` is the number of fields.
    pub fn minimum_capacity(&self) -> usize {
        self.fields
            .iter()
            .map(|field| field.capacity())
            .min()
            .unwrap_or(self.len)
    }
}

impl VectorMutOps for StructVectorMut {
    type Immutable = StructVector;

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.minimum_capacity()
    }

    fn reserve(&mut self, additional: usize) {
        // Reserve the additional capacity in each field vector.
        for field in &mut self.fields {
            field.reserve(additional);

            debug_assert_eq!(
                field.len(),
                self.len,
                "Field length must match `StructVectorMut` length"
            );
        }

        self.validity.reserve(additional);
    }

    fn extend_from_vector(&mut self, other: &StructVector) {
        assert_eq!(
            self.fields.len(),
            other.fields().len(),
            "Cannot extend StructVectorMut: field count mismatch (self had {} but other had {})",
            self.fields.len(),
            other.fields().len()
        );

        // Extend each field vector.
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
    }

    fn append_nulls(&mut self, n: usize) {
        for field in &mut self.fields {
            field.append_nulls(n); // Note that the value we push to each doesn't actually matter.
        }
        self.validity.append_n(false, n);

        self.len += n;
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

        let split_fields: Vec<VectorMut> = self
            .fields
            .iter_mut()
            .map(|field| field.split_off(at))
            .collect();

        let split_validity = self.validity.split_off(at);

        // Update self's state.
        let split_len = self.len.saturating_sub(at);
        self.len = at;

        Self {
            fields: split_fields,
            len: split_len,
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

        // Unsplit each field vector.
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

        self.validity.unsplit(other.validity);

        self.len += other.len;
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PTypeDowncast;
    use vortex_mask::{Mask, MaskMut};

    use super::*;
    use crate::{BoolVectorMut, NullVector, NullVectorMut, PVectorMut, VectorMut};

    #[test]
    fn test_empty_fields() {
        let mut struct_vec = StructVectorMut::try_new(vec![], MaskMut::new_true(10)).unwrap();
        let second_half = struct_vec.split_off(6);
        assert_eq!(struct_vec.len(), 6);
        assert_eq!(second_half.len(), 4);
    }

    #[test]
    fn test_try_into_mut_and_values() {
        let struct_vec = StructVector {
            fields: vec![
                NullVector::new(5).into(),
                BoolVectorMut::from_iter([true, false, true, false, true])
                    .freeze()
                    .into(),
                PVectorMut::<i32>::from_iter([10, 20, 30, 40, 50])
                    .freeze()
                    .into(),
            ]
            .into_boxed_slice(),
            len: 5,
            validity: Mask::AllTrue(5),
        };

        let mut_struct = struct_vec.try_into_mut().unwrap();
        assert_eq!(mut_struct.len(), 5);

        // Verify values are preserved.
        if let VectorMut::Bool(bool_vec) = mut_struct.fields[1].clone() {
            let values: Vec<_> = bool_vec.into_iter().map(|v| v.unwrap()).collect();
            assert_eq!(values, vec![true, false, true, false, true]);
        }

        if let VectorMut::Primitive(prim_vec) = mut_struct.fields[2].clone() {
            let values: Vec<_> = prim_vec
                .into_i32()
                .into_iter()
                .map(|v| v.unwrap())
                .collect();
            assert_eq!(values, vec![10, 20, 30, 40, 50]);
        }
    }

    #[test]
    fn test_try_into_mut_shared_ownership() {
        // Test that conversion fails when a field has shared ownership.
        let bool_field: Vector = BoolVectorMut::from_iter([true, false, true])
            .freeze()
            .into();
        let bool_field_clone = bool_field.clone();

        let struct_vec = StructVector {
            fields: vec![
                NullVector::new(3).into(),
                bool_field_clone,
                PVectorMut::<i32>::from_iter([1, 2, 3]).freeze().into(),
            ]
            .into_boxed_slice(),
            len: 3,
            validity: Mask::AllTrue(3),
        };

        assert!(struct_vec.try_into_mut().is_err());
        drop(bool_field); // Keep original alive to maintain shared ownership
    }

    #[test]
    fn test_split_unsplit_values() {
        let mut struct_vec = StructVectorMut::try_new(
            vec![
                NullVector::new(8).try_into_mut().unwrap().into(),
                BoolVectorMut::from_iter([true, false, true, false, true, false, true, false])
                    .into(),
                PVectorMut::<i32>::from_iter([10, 20, 30, 40, 50, 60, 70, 80]).into(),
            ],
            MaskMut::new_true(8),
        )
        .unwrap();

        let second_half = struct_vec.split_off(5);
        assert_eq!(struct_vec.len(), 5);
        assert_eq!(second_half.len(), 3);

        // Verify values after split.
        if let VectorMut::Bool(bool_vec) = struct_vec.fields[1].clone() {
            let values: Vec<_> = bool_vec.into_iter().take(5).map(|v| v.unwrap()).collect();
            assert_eq!(values, vec![true, false, true, false, true]);
        }

        if let VectorMut::Bool(bool_vec) = second_half.fields[1].clone() {
            let values: Vec<_> = bool_vec.into_iter().map(|v| v.unwrap()).collect();
            assert_eq!(values, vec![false, true, false]);
        }

        // Unsplit and verify.
        struct_vec.unsplit(second_half);
        assert_eq!(struct_vec.len(), 8);

        if let VectorMut::Bool(bool_vec) = struct_vec.fields[1].clone() {
            let values: Vec<_> = bool_vec.into_iter().map(|v| v.unwrap()).collect();
            assert_eq!(
                values,
                vec![true, false, true, false, true, false, true, false]
            );
        }
    }

    #[test]
    fn test_extend_and_append_nulls() {
        let mut struct_vec = StructVectorMut::try_new(
            vec![
                NullVector::new(3).try_into_mut().unwrap().into(),
                BoolVectorMut::from_iter([true, false, true]).into(),
                PVectorMut::<i32>::from_iter([10, 20, 30]).into(),
            ],
            MaskMut::new_true(3),
        )
        .unwrap();

        // Test extend.
        let to_extend = StructVector {
            fields: vec![
                NullVector::new(2).into(),
                BoolVectorMut::from_iter([false, true]).freeze().into(),
                PVectorMut::<i32>::from_iter([40, 50]).freeze().into(),
            ]
            .into_boxed_slice(),
            len: 2,
            validity: Mask::AllTrue(2),
        };

        struct_vec.extend_from_vector(&to_extend);
        assert_eq!(struct_vec.len(), 5);

        // Test append_nulls.
        struct_vec.append_nulls(2);
        assert_eq!(struct_vec.len(), 7);

        // Verify final values include nulls.
        if let VectorMut::Bool(bool_vec) = struct_vec.fields[1].clone() {
            let values: Vec<_> = bool_vec.into_iter().collect();
            assert_eq!(
                values,
                vec![
                    Some(true),
                    Some(false),
                    Some(true),
                    Some(false),
                    Some(true),
                    None,
                    None
                ]
            );
        }
    }

    #[test]
    fn test_roundtrip() {
        let original_bool = vec![Some(true), None, Some(false), Some(true)];
        let original_int = vec![Some(100i32), None, Some(200), Some(300)];

        let struct_vec = StructVectorMut::try_new(
            vec![
                NullVector::new(4).try_into_mut().unwrap().into(),
                BoolVectorMut::from_iter(original_bool.clone()).into(),
                PVectorMut::<i32>::from_iter(original_int.clone()).into(),
            ],
            MaskMut::new_true(4),
        )
        .unwrap();

        // Verify roundtrip preserves nulls.
        if let VectorMut::Bool(bool_vec) = struct_vec.fields[1].clone() {
            let roundtrip: Vec<_> = bool_vec.into_iter().collect();
            assert_eq!(roundtrip, original_bool);
        }

        if let VectorMut::Primitive(prim_vec) = struct_vec.fields[2].clone() {
            let roundtrip: Vec<_> = prim_vec.into_i32().into_iter().collect();
            assert_eq!(roundtrip, original_int);
        }
    }

    #[test]
    fn test_nested_struct() {
        let inner1 = StructVectorMut::try_new(
            vec![
                NullVector::new(4).try_into_mut().unwrap().into(),
                BoolVectorMut::from_iter([true, false, true, false]).into(),
            ],
            MaskMut::new_true(4),
        )
        .unwrap()
        .into();

        let inner2 = StructVectorMut::try_new(
            vec![PVectorMut::<u32>::from_iter([100, 200, 300, 400]).into()],
            MaskMut::new_true(4),
        )
        .unwrap()
        .into();

        let mut outer =
            StructVectorMut::try_new(vec![inner1, inner2], MaskMut::new_true(4)).unwrap();

        let second = outer.split_off(2);
        assert_eq!(outer.len(), 2);
        assert_eq!(second.len(), 2);

        outer.unsplit(second);
        assert_eq!(outer.len(), 4);
        assert!(matches!(outer.fields[0], VectorMut::Struct(_)));
    }

    #[test]
    fn test_reserve() {
        // Test that reserve increases capacity for all fields correctly.
        let mut struct_vec = StructVectorMut::try_new(
            vec![
                NullVectorMut::new(3).into(),
                BoolVectorMut::from_iter([true, false, true]).into(),
                PVectorMut::<i32>::from_iter([10, 20, 30]).into(),
            ],
            MaskMut::new_true(3),
        )
        .unwrap();

        let initial_capacity = struct_vec.capacity();
        assert_eq!(struct_vec.len(), 3);

        // Reserve additional capacity.
        struct_vec.reserve(50);

        // Capacity should now be at least len + 50.
        assert!(struct_vec.capacity() >= 3 + 50);
        assert!(struct_vec.capacity() >= initial_capacity + 50);

        // Verify minimum_capacity returns the smallest field capacity.
        let min_cap = struct_vec.minimum_capacity();
        for field in struct_vec.fields() {
            assert!(field.capacity() >= min_cap);
        }

        // Test reserve on an empty struct.
        let mut empty_struct = StructVectorMut::try_new(
            vec![
                NullVectorMut::new(0).into(),
                BoolVectorMut::with_capacity(0).into(),
            ],
            MaskMut::new_true(0),
        )
        .unwrap();

        empty_struct.reserve(100);
        assert!(empty_struct.capacity() >= 100);
    }

    #[test]
    fn test_freeze_and_new_unchecked() {
        // Test new_unchecked creates a valid struct, and freeze preserves data correctly.
        let fields = vec![
            NullVectorMut::new(4).into(),
            BoolVectorMut::from_iter([Some(true), None, Some(false), Some(true)]).into(),
            PVectorMut::<i32>::from_iter([Some(100), Some(200), None, Some(400)]).into(),
        ];

        let validity = Mask::from_iter([true, false, true, true])
            .try_into_mut()
            .unwrap();

        // Use new_unchecked to create the struct.
        // SAFETY: All fields have length 4 and validity has length 4.
        let struct_vec = unsafe { StructVectorMut::new_unchecked(fields, validity) };

        assert_eq!(struct_vec.len(), 4);
        assert_eq!(struct_vec.fields().len(), 3);

        // Freeze the struct and verify data preservation.
        let frozen = struct_vec.freeze();

        assert_eq!(frozen.len(), 4);
        assert_eq!(frozen.fields().len(), 3);

        // Verify validity is preserved (only indices 0, 2, 3 are valid at the struct level).
        assert_eq!(frozen.validity().true_count(), 3);

        // Verify that `try_into_mut` fails when data isn't owned.
        {
            let cloned_vector = frozen.fields()[1].clone();
            cloned_vector.try_into_mut().unwrap_err();
        }

        // Verify field data is preserved.
        let mut fields = frozen.into_parts().0.into_vec();

        if let Vector::Primitive(prim_vec) = fields.pop().unwrap() {
            let prim_vec_mut = prim_vec.try_into_mut().unwrap();
            let values: Vec<_> = prim_vec_mut.into_i32().into_iter().collect();
            assert_eq!(values, vec![Some(100), Some(200), None, Some(400)]);
        } else {
            panic!("Expected primitive vector");
        }

        if let Vector::Bool(bool_vec) = fields.pop().unwrap() {
            let bool_vec_mut = bool_vec.try_into_mut().unwrap();
            let values: Vec<_> = bool_vec_mut.into_iter().collect();
            // Note: struct-level validity doesn't affect field-level data, only the interpretation.
            assert_eq!(values, vec![Some(true), None, Some(false), Some(true)]);
        } else {
            panic!("Expected bool vector");
        }
    }
}
