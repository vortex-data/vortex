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
use vortex_array::buffer::BufferHandle;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability;

use crate::ByteBoolArray;

/// ByteBoolArray can generate non-nullable arrays.
pub const BYTEBOOL_CAN_GENERATE: &[ConstraintKind] = &[ConstraintKind::NonNullable];

/// A wrapper type to implement `Arbitrary` for `ByteBoolArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryByteBoolArray(pub ByteBoolArray);

impl<'a> Arbitrary<'a> for ArbitraryByteBoolArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let nullability: Nullability = u.arbitrary()?;
        Self::with_nullability(u, nullability, None)
    }
}

impl ArbitraryByteBoolArray {
    /// Generate an arbitrary ByteBoolArray with the given nullability.
    pub fn with_nullability(
        u: &mut Unstructured,
        nullability: Nullability,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Generate random bytes for bool values (0 or 1)
        let data: Vec<u8> = (0..len)
            .map(|_| u.arbitrary::<bool>().map(|b| b as u8))
            .collect::<Result<_>>()?;

        let buffer = BufferHandle::new_host(ByteBuffer::from(data));

        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => {
                // Generate random validity
                let validity_bools: Vec<bool> =
                    (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
                Validity::from_iter(validity_bools)
            }
        };

        Ok(ArbitraryByteBoolArray(ByteBoolArray::new(buffer, validity)))
    }

    /// Generate an arbitrary ByteBoolArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        _dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            Nullability::Nullable
        };
        Self::with_nullability(u, nullability, len)
    }
}

impl ArbitraryConstrained for ByteBoolArray {
    fn can_generate() -> &'static [ConstraintKind] {
        BYTEBOOL_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryByteBoolArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
