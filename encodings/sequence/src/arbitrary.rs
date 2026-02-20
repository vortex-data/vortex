// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::arbitrary::ArbitraryConstrained;
use vortex_array::arrays::arbitrary::ArrayConstraints;
use vortex_array::arrays::arbitrary::ConstraintKind;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::PValue;
use vortex_error::VortexExpect;

use crate::SequenceArray;

/// SequenceArray can generate sorted and strictly sorted arrays by choosing appropriate
/// base and multiplier values. For strictly sorted, multiplier > 0. For sorted, multiplier >= 0.
pub const SEQUENCE_CAN_GENERATE: &[ConstraintKind] = &[
    ConstraintKind::StrictlySorted,
    ConstraintKind::Sorted,
    ConstraintKind::StartsAtZero,
    ConstraintKind::BoundedAbove,
    ConstraintKind::BoundedBelow,
    ConstraintKind::NonNullable,
    ConstraintKind::Unsigned,
    ConstraintKind::IntegerOnly,
];

/// A wrapper type to implement `Arbitrary` for `SequenceArray`.
#[derive(Clone, Debug)]
pub struct ArbitrarySequenceArray(pub SequenceArray);

impl<'a> Arbitrary<'a> for ArbitrarySequenceArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let ptype = *u.choose(&[
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
        ])?;
        let nullability = u.arbitrary()?;
        Self::with_ptype(u, ptype, nullability, None)
    }
}

impl ArbitrarySequenceArray {
    /// Generate an arbitrary SequenceArray with the given ptype and nullability.
    pub fn with_ptype(
        u: &mut Unstructured,
        ptype: PType,
        nullability: Nullability,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Generate base and multiplier that won't overflow
        let (base, multiplier) = generate_safe_base_multiplier(u, ptype, len)?;

        let array = SequenceArray::new(base, multiplier, ptype, nullability, len)
            .vortex_expect("generated valid sequence parameters");

        Ok(ArbitrarySequenceArray(array))
    }

    /// Generate an arbitrary SequenceArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let DType::Primitive(ptype, nullability) = dtype else {
            // Sequence only supports integer types
            return Self::with_ptype(
                u,
                PType::I64,
                if constraints.non_nullable {
                    Nullability::NonNullable
                } else {
                    Nullability::Nullable
                },
                len,
            );
        };

        // SequenceArray only supports integers
        if !ptype.is_int() {
            return Self::with_ptype(
                u,
                if ptype.is_unsigned_int() {
                    PType::U64
                } else {
                    PType::I64
                },
                if constraints.non_nullable {
                    Nullability::NonNullable
                } else {
                    *nullability
                },
                len,
            );
        }

        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            *nullability
        };

        // Generate base and multiplier based on constraints
        let (base, multiplier) = generate_constrained_base_multiplier(u, *ptype, len, constraints)?;

        let array = SequenceArray::new(base, multiplier, *ptype, nullability, len)
            .vortex_expect("generated valid sequence parameters");

        Ok(ArbitrarySequenceArray(array))
    }
}

impl ArbitraryConstrained for SequenceArray {
    fn can_generate() -> &'static [ConstraintKind] {
        SEQUENCE_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitrarySequenceArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}

/// Generate base and multiplier that won't overflow for the given ptype and length.
fn generate_safe_base_multiplier(
    u: &mut Unstructured,
    ptype: PType,
    len: usize,
) -> Result<(PValue, PValue)> {
    if len == 0 {
        return Ok((PValue::I64(0), PValue::I64(1)));
    }

    // For safety, use small values that won't overflow
    let base: i64 = u.int_in_range(0..=100)?;
    let multiplier: i64 = u.int_in_range(1..=10)?;

    // Check that last value won't overflow
    let last_idx = (len - 1) as i64;
    if base
        .checked_add(last_idx.checked_mul(multiplier).unwrap_or(i64::MAX))
        .is_none()
    {
        // Fall back to smaller multiplier
        return Ok((pvalue_from_i64(0, ptype), pvalue_from_i64(1, ptype)));
    }

    Ok((
        pvalue_from_i64(base, ptype),
        pvalue_from_i64(multiplier, ptype),
    ))
}

/// Generate base and multiplier satisfying the given constraints.
fn generate_constrained_base_multiplier(
    u: &mut Unstructured,
    ptype: PType,
    len: usize,
    constraints: &ArrayConstraints,
) -> Result<(PValue, PValue)> {
    if len == 0 {
        let base = constraints.ordering.starts_at.unwrap_or(0) as i64;
        return Ok((pvalue_from_i64(base, ptype), pvalue_from_i64(1, ptype)));
    }

    // Determine base
    let base: i64 = if let Some(start) = constraints.ordering.starts_at {
        start as i64
    } else if let Some(lower) = constraints.bounds.lower_bound {
        lower as i64
    } else {
        u.int_in_range(0..=100)?
    };

    // Determine multiplier based on sorting constraints
    let multiplier: i64 = if constraints.ordering.strictly_sorted {
        // Must be positive for strictly sorted
        u.int_in_range(1..=10)?
    } else if constraints.ordering.sorted {
        // Can be zero or positive for sorted
        u.int_in_range(0..=10)?
    } else {
        // No sorting constraint, can be any value
        u.int_in_range(-10..=10)?
    };

    // Ensure we don't overflow and stay within bounds
    let last_idx = (len - 1) as i64;
    let last_value = base.saturating_add(last_idx.saturating_mul(multiplier));

    // Check upper bound
    if let Some(upper) = constraints.bounds.upper_bound
        && last_value >= upper as i64
    {
        // Adjust multiplier to fit within bounds
        let available = (upper as i64).saturating_sub(base).saturating_sub(1);
        let safe_multiplier = if len > 1 {
            (available / last_idx).max(if constraints.ordering.strictly_sorted {
                1
            } else {
                0
            })
        } else {
            1
        };
        return Ok((
            pvalue_from_i64(base, ptype),
            pvalue_from_i64(safe_multiplier, ptype),
        ));
    }

    // Check target_max
    if let Some(target) = constraints.bounds.target_max
        && last_value > target as i64
    {
        let available = (target as i64).saturating_sub(base);
        let safe_multiplier = if len > 1 {
            (available / last_idx).max(if constraints.ordering.strictly_sorted {
                1
            } else {
                0
            })
        } else {
            1
        };
        return Ok((
            pvalue_from_i64(base, ptype),
            pvalue_from_i64(safe_multiplier, ptype),
        ));
    }

    Ok((
        pvalue_from_i64(base, ptype),
        pvalue_from_i64(multiplier, ptype),
    ))
}

#[allow(clippy::cast_possible_truncation)]
fn pvalue_from_i64(value: i64, ptype: PType) -> PValue {
    match ptype {
        PType::I8 => PValue::I8(value as i8),
        PType::I16 => PValue::I16(value as i16),
        PType::I32 => PValue::I32(value as i32),
        PType::I64 => PValue::I64(value),
        PType::U8 => PValue::U8(value as u8),
        PType::U16 => PValue::U16(value as u16),
        PType::U32 => PValue::U32(value as u32),
        PType::U64 => PValue::U64(value as u64),
        PType::F16 | PType::F32 | PType::F64 => {
            // Sequence doesn't support floats, but provide a fallback
            PValue::I64(value)
        }
    }
}
