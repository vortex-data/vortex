// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;

use super::ConstantArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::arbitrary::ArbitraryConstrained;
use crate::arrays::arbitrary::ArrayConstraints;
use crate::arrays::arbitrary::ConstraintKind;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar::arbitrary::random_scalar;

/// ConstantArray can generate all constraint kinds - a single repeated value trivially satisfies
/// all ordering constraints (sorted, strictly sorted for len <= 1, bounded, etc.)
pub const CONSTANT_CAN_GENERATE: &[ConstraintKind] = &[
    ConstraintKind::Sorted,
    ConstraintKind::StartsAtZero,
    ConstraintKind::BoundedAbove,
    ConstraintKind::BoundedBelow,
    ConstraintKind::BitWidthBounded,
    ConstraintKind::NonNullable,
    ConstraintKind::Unsigned,
    ConstraintKind::IntegerOnly,
    // Note: StrictlySorted only works for len <= 1, so we don't include it
];

/// A wrapper type to implement `Arbitrary` for `ConstantArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryConstantArray(pub ConstantArray);

impl<'a> Arbitrary<'a> for ArbitraryConstantArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype: DType = u.arbitrary()?;
        Self::with_dtype(u, &dtype, None)
    }
}

impl ArbitraryConstantArray {
    /// Generate an arbitrary ConstantArray with the given dtype.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<Self> {
        let scalar = random_scalar(u, dtype)?;
        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        Ok(ArbitraryConstantArray(ConstantArray::new(scalar, len)))
    }

    /// Generate an arbitrary ConstantArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);

        // For strictly sorted with len > 1, constant can't satisfy it
        if constraints.ordering.strictly_sorted && len > 1 {
            // Fall back to a len of 0 or 1
            let len = u.int_in_range(0..=1)?;
            let scalar = random_constrained_scalar(u, dtype, constraints)?;
            return Ok(ArbitraryConstantArray(ConstantArray::new(scalar, len)));
        }

        let scalar = random_constrained_scalar(u, dtype, constraints)?;
        Ok(ArbitraryConstantArray(ConstantArray::new(scalar, len)))
    }
}

impl ArbitraryConstrained for ConstantArray {
    fn can_generate() -> &'static [ConstraintKind] {
        CONSTANT_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryConstantArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}

/// Generate a scalar that satisfies the given constraints.
fn random_constrained_scalar(
    u: &mut Unstructured,
    dtype: &DType,
    constraints: &ArrayConstraints,
) -> Result<Scalar> {
    // For primitive types with bounds, generate a value within bounds
    if let DType::Primitive(ptype, nullability) = dtype {
        let nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            *nullability
        };

        // Handle starts_at constraint
        if let Some(start) = constraints.ordering.starts_at {
            return Ok(scalar_from_u64(start, *ptype, nullability));
        }

        // Handle bounded constraints
        if constraints.bounds.upper_bound.is_some() || constraints.bounds.lower_bound.is_some() {
            let lower = constraints.bounds.lower_bound.unwrap_or(0);
            let upper = constraints
                .bounds
                .upper_bound
                .unwrap_or_else(|| ptype_max(*ptype))
                .saturating_sub(1)
                .max(lower);
            let value = u.int_in_range(lower..=upper)?;
            return Ok(scalar_from_u64(value, *ptype, nullability));
        }
    }

    // Fall back to random scalar
    random_scalar(u, dtype)
}

#[allow(clippy::cast_possible_truncation)]
fn scalar_from_u64(value: u64, ptype: PType, nullability: Nullability) -> Scalar {
    match ptype {
        PType::U8 => Scalar::primitive(value as u8, nullability),
        PType::U16 => Scalar::primitive(value as u16, nullability),
        PType::U32 => Scalar::primitive(value as u32, nullability),
        PType::U64 => Scalar::primitive(value, nullability),
        PType::I8 => Scalar::primitive(value as i8, nullability),
        PType::I16 => Scalar::primitive(value as i16, nullability),
        PType::I32 => Scalar::primitive(value as i32, nullability),
        PType::I64 => Scalar::primitive(value as i64, nullability),
        PType::F16 => {
            let f = half::f16::from_f64(value as f64);
            Scalar::primitive(f, nullability)
        }
        PType::F32 => Scalar::primitive(value as f32, nullability),
        PType::F64 => Scalar::primitive(value as f64, nullability),
    }
}

fn ptype_max(ptype: PType) -> u64 {
    match ptype {
        PType::U8 => u8::MAX as u64,
        PType::U16 => u16::MAX as u64,
        PType::U32 => u32::MAX as u64,
        PType::U64 => u64::MAX,
        PType::I8 => i8::MAX as u64,
        PType::I16 => i16::MAX as u64,
        PType::I32 => i32::MAX as u64,
        PType::I64 => i64::MAX as u64,
        PType::F16 => u16::MAX as u64,
        PType::F32 => u32::MAX as u64,
        PType::F64 => u64::MAX,
    }
}
