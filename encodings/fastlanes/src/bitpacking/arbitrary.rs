// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::arbitrary::ArbitraryConstrained;
use vortex_array::arrays::arbitrary::ArrayConstraints;
use vortex_array::arrays::arbitrary::ConstraintKind;
use vortex_array::arrays::arbitrary::arbitrary_constrained_array;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;

use crate::BitPackedArray;

/// BitPackedArray preserves sorted order and can enforce bit-width bounds.
pub const BITPACKED_CAN_GENERATE: &[ConstraintKind] = &[
    ConstraintKind::StrictlySorted,
    ConstraintKind::Sorted,
    ConstraintKind::BoundedAbove,
    ConstraintKind::BitWidthBounded,
    ConstraintKind::NonNullable,
    ConstraintKind::Unsigned,
    ConstraintKind::IntegerOnly,
];

/// A wrapper type to implement `Arbitrary` for `BitPackedArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryBitPackedArray(pub BitPackedArray);

impl<'a> Arbitrary<'a> for ArbitraryBitPackedArray {
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

impl ArbitraryBitPackedArray {
    /// Generate an arbitrary BitPackedArray with the given ptype and nullability.
    pub fn with_ptype(
        u: &mut Unstructured,
        ptype: PType,
        nullability: Nullability,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let dtype = DType::Primitive(ptype, nullability);

        // Generate small positive values that can be bit-packed efficiently
        let mut constraints = ArrayConstraints::default();
        // Use a small bit width to ensure values fit
        let max_bits = ptype.byte_width() as u8 * 8;
        let bit_width = u.int_in_range(1..=max_bits.min(16))?;
        constraints.bounds.upper_bound = Some(1u64 << bit_width);

        let values = arbitrary_constrained_array(u, Some(len), &dtype, &constraints)?;
        let primitive = values.to_primitive();

        let array = BitPackedArray::encode(primitive.as_ref(), bit_width)
            .vortex_expect("bitpacking should succeed");

        Ok(ArbitraryBitPackedArray(array))
    }

    /// Generate an arbitrary BitPackedArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let DType::Primitive(ptype, nullability) = dtype else {
            return Self::with_ptype(u, PType::U64, Nullability::NonNullable, len);
        };

        if !ptype.is_int() {
            return Self::with_ptype(u, PType::U64, *nullability, len);
        }

        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            *nullability
        };

        // Determine bit width from constraints
        let max_type_bits = ptype.byte_width() as u8 * 8;
        let bit_width = if let Some(bits) = constraints.bounds.bit_width {
            bits.min(max_type_bits)
        } else if let Some(upper) = constraints.bounds.upper_bound {
            // Calculate minimum bits needed for upper bound
            (64 - upper.leading_zeros()).max(1) as u8
        } else {
            // Default to a reasonable bit width
            u.int_in_range(1..=max_type_bits.min(16))?
        };

        // Build constraints ensuring values fit in bit_width
        let mut adjusted_constraints = constraints.clone();
        let max_value = if bit_width >= 64 {
            u64::MAX
        } else {
            (1u64 << bit_width).saturating_sub(1)
        };

        // Ensure upper bound is within bit width
        if let Some(upper) = adjusted_constraints.bounds.upper_bound {
            adjusted_constraints.bounds.upper_bound = Some(upper.min(max_value + 1));
        } else {
            adjusted_constraints.bounds.upper_bound = Some(max_value + 1);
        }

        // Generate constrained values
        let values_dtype = DType::Primitive(*ptype, nullability);
        let values =
            arbitrary_constrained_array(u, Some(len), &values_dtype, &adjusted_constraints)?;
        let primitive = values.to_primitive();

        let array = BitPackedArray::encode(primitive.as_ref(), bit_width)
            .vortex_expect("bitpacking should succeed");

        Ok(ArbitraryBitPackedArray(array))
    }
}

impl ArbitraryConstrained for BitPackedArray {
    fn can_generate() -> &'static [ConstraintKind] {
        BITPACKED_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryBitPackedArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
