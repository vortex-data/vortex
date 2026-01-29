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
use vortex_array::arrays::arbitrary::arbitrary_constrained_array;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;

use crate::ZigZagArray;

/// ZigZagArray can preserve sorted order if the encoded values are sorted.
pub const ZIGZAG_CAN_GENERATE: &[ConstraintKind] = &[ConstraintKind::NonNullable];

/// A wrapper type to implement `Arbitrary` for `ZigZagArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryZigZagArray(pub ZigZagArray);

impl<'a> Arbitrary<'a> for ArbitraryZigZagArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let ptype = *u.choose(&[PType::U8, PType::U16, PType::U32, PType::U64])?;
        let nullability: Nullability = u.arbitrary()?;
        Self::with_ptype(u, ptype, nullability, None)
    }
}

impl ArbitraryZigZagArray {
    /// Generate an arbitrary ZigZagArray with the given ptype and nullability.
    pub fn with_ptype(
        u: &mut Unstructured,
        ptype: PType,
        nullability: Nullability,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let encoded_dtype = DType::Primitive(ptype, nullability);
        let encoded = arbitrary_constrained_array(
            u,
            Some(len),
            &encoded_dtype,
            &ArrayConstraints::default(),
        )?;
        let array =
            ZigZagArray::try_new(encoded).vortex_expect("ZigZagArray creation should succeed");
        Ok(ArbitraryZigZagArray(array))
    }

    /// Generate an arbitrary ZigZagArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let ptype = match dtype {
            DType::Primitive(p, _) if p.is_unsigned_int() => *p,
            _ => PType::U64,
        };
        let nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            dtype.nullability()
        };
        Self::with_ptype(u, ptype, nullability, len)
    }
}

impl ArbitraryConstrained for ZigZagArray {
    fn can_generate() -> &'static [ConstraintKind] {
        ZIGZAG_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryZigZagArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
