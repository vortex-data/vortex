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
/// use vortex_vector::{StructVectorMut, VectorMut, BoolVectorMut, PVectorMut, NullVector};
/// use vortex_mask::MaskMut;
///
/// // Create a struct with three fields: nulls, booleans, and integers.
/// let fields = vec![
///     NullVector::new(3).try_into_mut().unwrap().into(),
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
/// use vortex_vector::{StructVectorMut, VectorMut, BoolVectorMut, PVectorMut, NullVector};
/// use vortex_mask::MaskMut;
///
/// let fields = vec![
///     NullVector::new(6).try_into_mut().unwrap().into(),
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
/// use vortex_vector::{StructVectorMut, VectorMut, BoolVectorMut, PVectorMut, NullVector};
/// use vortex_mask::MaskMut;
/// use vortex_dtype::PTypeDowncast;
///
/// let fields = vec![
///     NullVector::new(3).try_into_mut().unwrap().into(),
///     BoolVectorMut::from_iter([true, false, true]).into(),
///     PVectorMut::<i32>::from_iter([10, 20, 30]).into(),
/// ];
///
/// let struct_vec = StructVectorMut::new(fields, MaskMut::new_true(3));
///
/// // Access the boolean field vector (field index 1).
/// if let VectorMut::Bool(bool_vec) = struct_vec.fields[1].clone() {
///     let values: Vec<_> = bool_vec.into_iter().map(|v| v.unwrap()).collect();
///     assert_eq!(values, vec![true, false, true]);
/// }
///
/// // Access the integer field column (field index 2).
/// if let VectorMut::Primitive(prim_vec) = struct_vec.fields[2].clone() {
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
        Self::try_new(fields, validity).vortex_expect(
            "`StructVectorMut` fields must have matching length and validity constraints",
        )
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

        Self::validate(&fields, len, &validity)?;

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

        debug_assert!(
            Self::validate(&fields, len, &validity).is_ok(),
            "`StructVectorMut` fields must have matching length and validity constraints"
        );

        Self {
            fields,
            validity,
            len,
        }
    }

    /// Validates the fields and validity mask for a [`StructVectorMut`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - Any field vector has a length that does not match `len`.
    /// - The validity mask length does not match `len`.
    fn validate(fields: &[VectorMut], len: usize, validity: &MaskMut) -> VortexResult<()> {
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
    pub fn into_parts(self) -> (Vec<VectorMut>, MaskMut, usize) {
        (self.fields, self.validity, self.len)
    }

    /// Returns the fields of the `StructVectorMut`, each stored column-wise as a [`VectorMut`].
    pub fn fields(&mut self) -> &[VectorMut] {
        self.fields.as_mut_slice()
    }

    /// Finds the minimum capacity of all field vectors.
    ///
    /// This is equal to the maximum amount of scalars we can add before we need to reallocate at
    /// least one of the child field vectors.
    ///
    /// If there are no fields, this returns the length of the vector.
    pub fn minimum_capacity(&self) -> usize {
        if self.fields.is_empty() {
            return self.len;
        }

        let mut minimum_capacity = usize::MAX;
        for field in &self.fields {
            minimum_capacity = minimum_capacity.min(field.capacity());
        }

        minimum_capacity
    }
}

impl VectorMutOps for StructVectorMut {
    type Immutable = StructVector;

    fn len(&self) -> usize {
        self.len
    }

    /// Note that this returns the length of the [`StructVectorMut`].
    ///
    /// If you want the actual capacity of the struct vector, use the [`minimum_capacity()`] method.
    ///
    /// [`minimum_capacity()`]: Self::minimum_capacity
    fn capacity(&self) -> usize {
        self.len
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
            "Cannot extend StructVectorMut: field count mismatch ({} vs {})",
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
    use crate::{BoolVectorMut, NullVector, PVectorMut, VectorMut};

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
}
