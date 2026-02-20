// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::arbitrary::ArbitraryConstrained;
use vortex_array::arrays::arbitrary::ArrayConstraints;
use vortex_array::arrays::arbitrary::BoundConstraint;
use vortex_array::arrays::arbitrary::ConstraintKind;
use vortex_array::arrays::arbitrary::OrderingConstraint;
use vortex_array::arrays::arbitrary::arbitrary_constrained_array;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::arbitrary::random_scalar;
use vortex_error::VortexExpect;

use crate::SparseArray;

/// SparseArray has strictly sorted indices. It doesn't preserve order of values in general,
/// but can use constrained generation for its indices.
pub const SPARSE_CAN_GENERATE: &[ConstraintKind] = &[ConstraintKind::NonNullable];

/// A wrapper type to implement `Arbitrary` for `SparseArray`.
#[derive(Clone, Debug)]
pub struct ArbitrarySparseArray(pub SparseArray);

impl<'a> Arbitrary<'a> for ArbitrarySparseArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype: DType = u.arbitrary()?;
        Self::with_dtype(u, &dtype, None)
    }
}

impl ArbitrarySparseArray {
    /// Generate an arbitrary SparseArray with the given dtype.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Generate number of sparse values (patches)
        let num_patches = if len == 0 {
            0
        } else {
            u.int_in_range(0..=len.min(50))?
        };

        // Generate strictly sorted indices within [0, len)
        let indices_constraints = ArrayConstraints {
            ordering: OrderingConstraint {
                strictly_sorted: true,
                sorted: true,
                starts_at: Some(0),
            },
            bounds: BoundConstraint {
                upper_bound: Some(len as u64),
                lower_bound: Some(0),
                ..Default::default()
            },
            non_nullable: true,
            ..Default::default()
        };

        let indices_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let indices = arbitrary_constrained_array(
            u,
            Some(num_patches),
            &indices_dtype,
            &indices_constraints,
        )?;

        // Generate arbitrary values
        let values =
            arbitrary_constrained_array(u, Some(num_patches), dtype, &ArrayConstraints::default())?;

        // Generate fill value
        let fill_value = random_scalar(u, dtype)?;

        let array = SparseArray::try_new(indices, values, len, fill_value)
            .vortex_expect("generated valid sparse array");

        Ok(ArbitrarySparseArray(array))
    }

    /// Generate an arbitrary SparseArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Generate number of sparse values (patches)
        let num_patches = if len == 0 {
            0
        } else {
            u.int_in_range(0..=len.min(50))?
        };

        // Generate strictly sorted indices within [0, len)
        let indices_constraints = ArrayConstraints {
            ordering: OrderingConstraint {
                strictly_sorted: true,
                sorted: true,
                starts_at: Some(0),
            },
            bounds: BoundConstraint {
                upper_bound: Some(len as u64),
                lower_bound: Some(0),
                ..Default::default()
            },
            non_nullable: true,
            ..Default::default()
        };

        let indices_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let indices = arbitrary_constrained_array(
            u,
            Some(num_patches),
            &indices_dtype,
            &indices_constraints,
        )?;

        // Generate values that satisfy the constraints (for values themselves)
        let values = arbitrary_constrained_array(u, Some(num_patches), dtype, constraints)?;

        // Generate fill value - for non-nullable constraint, make it non-null
        let fill_dtype = if constraints.non_nullable {
            dtype.as_nonnullable()
        } else {
            dtype.clone()
        };
        let fill_value = random_scalar(u, &fill_dtype)?;

        let array = SparseArray::try_new(indices, values, len, fill_value)
            .vortex_expect("generated valid sparse array");

        Ok(ArbitrarySparseArray(array))
    }
}

impl ArbitraryConstrained for SparseArray {
    fn can_generate() -> &'static [ConstraintKind] {
        SPARSE_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitrarySparseArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
