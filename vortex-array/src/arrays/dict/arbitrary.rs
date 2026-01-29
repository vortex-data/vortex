// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_error::VortexExpect;

use super::DictArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::arbitrary::ArbitraryArray;
use crate::arrays::arbitrary::ArbitraryConstrained;
use crate::arrays::arbitrary::ArrayConstraints;
use crate::arrays::arbitrary::BoundConstraint;
use crate::arrays::arbitrary::ConstraintKind;
use crate::arrays::arbitrary::arbitrary_constrained_array;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

/// DictArray doesn't preserve ordering (codes are indices), but codes are bounded.
pub const DICT_CAN_GENERATE: &[ConstraintKind] = &[ConstraintKind::NonNullable];

/// A wrapper type to implement `Arbitrary` for `DictArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryDictArray(pub DictArray);

impl<'a> Arbitrary<'a> for ArbitraryDictArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype: DType = u.arbitrary()?;
        Self::with_dtype(u, &dtype, None)
    }
}

impl ArbitraryDictArray {
    /// Generate an arbitrary DictArray with the given dtype for values.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<Self> {
        Self::with_constraints(u, len, dtype, &ArrayConstraints::default())
    }

    /// Generate an arbitrary DictArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        // Generate the number of unique values (dictionary size)
        let values_len = u.int_in_range(1..=20)?;

        // Generate values array with the given dtype
        let values = ArbitraryArray::arbitrary_with(u, Some(values_len), dtype)?.0;

        // Generate codes that index into the values
        let codes_len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Determine the minimum PType that can represent all indices
        let min_codes_ptype = PType::min_unsigned_ptype_for_value((values_len - 1) as u64);

        // Choose a random PType at least as wide as the minimum
        let valid_ptypes: &[PType] = match min_codes_ptype {
            PType::U8 => &[
                PType::U8,
                PType::U16,
                PType::U32,
                PType::U64,
                PType::I8,
                PType::I16,
                PType::I32,
                PType::I64,
            ],
            PType::U16 => &[
                PType::U16,
                PType::U32,
                PType::U64,
                PType::I16,
                PType::I32,
                PType::I64,
            ],
            PType::U32 => &[PType::U32, PType::U64, PType::I32, PType::I64],
            PType::U64 => &[PType::U64, PType::I64],
            _ => unreachable!(),
        };
        let codes_ptype = *u.choose(valid_ptypes)?;

        // Generate codes with optional nullability using constrained generation
        let codes_nullable: Nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            u.arbitrary()?
        };

        // Codes must be bounded within [0, values_len)
        let codes_constraints = ArrayConstraints {
            bounds: BoundConstraint {
                lower_bound: Some(0),
                upper_bound: Some(values_len as u64),
                ..Default::default()
            },
            non_nullable: codes_nullable == Nullability::NonNullable,
            ..Default::default()
        };

        let codes_dtype = DType::Primitive(codes_ptype, codes_nullable);
        let codes =
            arbitrary_constrained_array(u, Some(codes_len), &codes_dtype, &codes_constraints)?;

        Ok(ArbitraryDictArray(
            DictArray::try_new(codes, values)
                .vortex_expect("DictArray creation should succeed in arbitrary impl"),
        ))
    }
}

impl ArbitraryConstrained for DictArray {
    fn can_generate() -> &'static [ConstraintKind] {
        DICT_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryDictArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
