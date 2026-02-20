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

use crate::FoRArray;

/// FoRArray (Frame of Reference) preserves sorted order since it just adds a constant offset.
pub const FOR_CAN_GENERATE: &[ConstraintKind] = &[
    ConstraintKind::StrictlySorted,
    ConstraintKind::Sorted,
    ConstraintKind::BoundedAbove,
    ConstraintKind::NonNullable,
    ConstraintKind::Unsigned,
    ConstraintKind::IntegerOnly,
];

/// A wrapper type to implement `Arbitrary` for `FoRArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryFoRArray(pub FoRArray);

impl<'a> Arbitrary<'a> for ArbitraryFoRArray {
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

impl ArbitraryFoRArray {
    /// Generate an arbitrary FoRArray with the given ptype and nullability.
    pub fn with_ptype(
        u: &mut Unstructured,
        ptype: PType,
        nullability: Nullability,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let dtype = DType::Primitive(ptype, nullability);

        // Generate values and FoR-compress them
        let values =
            arbitrary_constrained_array(u, Some(len), &dtype, &ArrayConstraints::default())?;
        let primitive = values.to_primitive();
        let array = FoRArray::encode(primitive).vortex_expect("FoR compression should succeed");

        Ok(ArbitraryFoRArray(array))
    }

    /// Generate an arbitrary FoRArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let DType::Primitive(ptype, nullability) = dtype else {
            return Self::with_ptype(u, PType::I64, Nullability::NonNullable, len);
        };

        if !ptype.is_int() {
            return Self::with_ptype(u, PType::I64, *nullability, len);
        }

        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            *nullability
        };

        // Generate sorted values using the constrained primitive generation
        let values_dtype = DType::Primitive(*ptype, nullability);
        let values = arbitrary_constrained_array(u, Some(len), &values_dtype, constraints)?;

        // FoR compress the values
        let primitive = values.to_primitive();
        let array = FoRArray::encode(primitive).vortex_expect("FoR compression should succeed");

        Ok(ArbitraryFoRArray(array))
    }
}

impl ArbitraryConstrained for FoRArray {
    fn can_generate() -> &'static [ConstraintKind] {
        FOR_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryFoRArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
