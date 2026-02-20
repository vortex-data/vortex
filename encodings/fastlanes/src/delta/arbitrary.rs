// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryConstrained;
use vortex_array::arrays::arbitrary::ArrayConstraints;
use vortex_array::arrays::arbitrary::ConstraintKind;
use vortex_array::arrays::arbitrary::arbitrary_constrained_array;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

use crate::DeltaArray;

/// DeltaArray can generate sorted arrays by wrapping non-negative deltas.
/// For strictly sorted, deltas are positive (>= 1).
/// For sorted, deltas are non-negative (>= 0).
pub const DELTA_CAN_GENERATE: &[ConstraintKind] = &[
    ConstraintKind::StrictlySorted,
    ConstraintKind::Sorted,
    ConstraintKind::BoundedAbove,
    ConstraintKind::NonNullable,
    ConstraintKind::Unsigned,
    ConstraintKind::IntegerOnly,
];

/// A wrapper type to implement `Arbitrary` for `DeltaArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryDeltaArray(pub DeltaArray);

impl<'a> Arbitrary<'a> for ArbitraryDeltaArray {
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

impl ArbitraryDeltaArray {
    /// Generate an arbitrary DeltaArray with the given ptype and nullability.
    pub fn with_ptype(
        u: &mut Unstructured,
        ptype: PType,
        nullability: Nullability,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Generate values and delta-compress them
        let values = generate_random_values(u, ptype, nullability, len)?;
        let array = DeltaArray::try_from_primitive_array(&values)
            .vortex_expect("delta compression should succeed");

        Ok(ArbitraryDeltaArray(array))
    }

    /// Generate an arbitrary DeltaArray satisfying the given constraints.
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

        // Delta compress the sorted values
        let primitive = values.to_primitive();
        let array = DeltaArray::try_from_primitive_array(&primitive)
            .vortex_expect("delta compression should succeed");

        Ok(ArbitraryDeltaArray(array))
    }
}

impl ArbitraryConstrained for DeltaArray {
    fn can_generate() -> &'static [ConstraintKind] {
        DELTA_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryDeltaArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}

/// Generate random primitive values.
fn generate_random_values(
    u: &mut Unstructured,
    ptype: PType,
    nullability: Nullability,
    len: usize,
) -> Result<PrimitiveArray> {
    let validity = if nullability == Nullability::NonNullable {
        vortex_array::validity::Validity::NonNullable
    } else {
        vortex_array::validity::Validity::AllValid
    };

    match ptype {
        PType::I8 => {
            let values: Vec<i8> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        PType::I16 => {
            let values: Vec<i16> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        PType::I32 => {
            let values: Vec<i32> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        PType::I64 => {
            let values: Vec<i64> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        PType::U8 => {
            let values: Vec<u8> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        PType::U16 => {
            let values: Vec<u16> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        PType::U32 => {
            let values: Vec<u32> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        PType::U64 => {
            let values: Vec<u64> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
        _ => {
            // Fallback for floats (which Delta doesn't support)
            let values: Vec<i64> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(values), validity))
        }
    }
}
