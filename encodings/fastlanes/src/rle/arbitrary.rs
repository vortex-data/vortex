// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryConstrained;
use vortex_array::arrays::arbitrary::ArrayConstraints;
use vortex_array::arrays::arbitrary::ConstraintKind;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

use crate::FL_CHUNK_SIZE;
use crate::RLEArray;

/// RLEArray (FastLanes) has bounded indices. It's a chunk-based encoding.
pub const RLE_CAN_GENERATE: &[ConstraintKind] = &[ConstraintKind::NonNullable];

/// A wrapper type to implement `Arbitrary` for `RLEArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryRLEArray(pub RLEArray);

impl<'a> Arbitrary<'a> for ArbitraryRLEArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let ptype: PType = *u.choose(&[
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
        ])?;
        let nullability: Nullability = u.arbitrary()?;
        let dtype = DType::Primitive(ptype, nullability);
        Self::with_dtype(u, &dtype, None)
    }
}

impl ArbitraryRLEArray {
    /// Generate an arbitrary RLEArray with the given dtype for values.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<Self> {
        Self::with_constraints(u, len, dtype, &ArrayConstraints::default())
    }

    /// Generate an arbitrary RLEArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        let DType::Primitive(ptype, _nullability) = dtype else {
            // Fallback to i32
            return Self::with_dtype(
                u,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
                len,
            );
        };

        // Number of chunks (1-3 for testing)
        let num_chunks = u.int_in_range(1..=3)?;
        let total_indices = num_chunks * FL_CHUNK_SIZE;

        // Actual array length (may be smaller due to offset)
        let offset = u.int_in_range(0..=FL_CHUNK_SIZE.min(100) - 1)?;
        let length = len.unwrap_or_else(|| {
            let max_len = total_indices.saturating_sub(offset);
            if max_len == 0 {
                0
            } else {
                u.int_in_range(1..=max_len).unwrap_or(1)
            }
        });

        // Generate unique values per chunk (small number for RLE efficiency)
        let values_per_chunk = u.int_in_range(1..=10)?;
        let total_values = values_per_chunk * num_chunks;

        // Generate values array
        let values = generate_values(u, *ptype, total_values)?;

        // Generate indices (u8 or u16, bounded by values_per_chunk)
        let indices_ptype = if values_per_chunk <= 256 {
            PType::U8
        } else {
            PType::U16
        };

        let indices_nullability = if constraints.non_nullable {
            Nullability::NonNullable
        } else {
            u.arbitrary()?
        };

        let indices = generate_indices(
            u,
            indices_ptype,
            indices_nullability,
            total_indices,
            values_per_chunk,
        )?;

        // Generate values_idx_offsets (one per chunk)
        let values_idx_offsets: Vec<u64> = (0..num_chunks)
            .map(|i| (i * values_per_chunk) as u64)
            .collect();
        let values_idx_offsets = PrimitiveArray::new(
            Buffer::copy_from(values_idx_offsets),
            vortex_array::validity::Validity::NonNullable,
        )
        .into_array();

        let array = RLEArray::try_new(values, indices, values_idx_offsets, offset, length)
            .vortex_expect("RLEArray creation should succeed");

        Ok(ArbitraryRLEArray(array))
    }
}

fn generate_values(u: &mut Unstructured, ptype: PType, len: usize) -> Result<ArrayRef> {
    macro_rules! gen_values {
        ($t:ty) => {{
            let values: Vec<$t> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(
                Buffer::copy_from(values),
                vortex_array::validity::Validity::NonNullable,
            )
            .into_array())
        }};
    }

    match ptype {
        PType::I8 => gen_values!(i8),
        PType::I16 => gen_values!(i16),
        PType::I32 => gen_values!(i32),
        PType::I64 => gen_values!(i64),
        PType::U8 => gen_values!(u8),
        PType::U16 => gen_values!(u16),
        PType::U32 => gen_values!(u32),
        PType::U64 => gen_values!(u64),
        _ => gen_values!(i32),
    }
}

#[allow(clippy::cast_possible_truncation)]
fn generate_indices(
    u: &mut Unstructured,
    ptype: PType,
    nullability: Nullability,
    len: usize,
    max_value: usize,
) -> Result<ArrayRef> {
    let validity = if nullability == Nullability::NonNullable {
        vortex_array::validity::Validity::NonNullable
    } else {
        vortex_array::validity::Validity::from_iter((0..len).map(|_| u.arbitrary().unwrap_or(true)))
    };

    match ptype {
        PType::U8 => {
            let indices: Vec<u8> = (0..len)
                .map(|_| Ok(u.int_in_range(0..=max_value - 1)? as u8))
                .collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(indices), validity).into_array())
        }
        PType::U16 => {
            let indices: Vec<u16> = (0..len)
                .map(|_| Ok(u.int_in_range(0..=max_value - 1)? as u16))
                .collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(indices), validity).into_array())
        }
        _ => {
            let indices: Vec<u8> = (0..len)
                .map(|_| Ok(u.int_in_range(0..=max_value - 1)? as u8))
                .collect::<Result<_>>()?;
            Ok(PrimitiveArray::new(Buffer::copy_from(indices), validity).into_array())
        }
    }
}

impl ArbitraryConstrained for RLEArray {
    fn can_generate() -> &'static [ConstraintKind] {
        RLE_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryRLEArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
