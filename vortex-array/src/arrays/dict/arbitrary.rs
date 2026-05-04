// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use num_traits::NumCast;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

use super::DictArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::arbitrary::ArbitraryArray;
use crate::arrays::arbitrary::ArbitraryArrayConfig;
use crate::arrays::arbitrary::ArbitraryWith;
use crate::arrays::arbitrary::random_validity;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;

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
        // Generate the number of unique values (dictionary size)
        let values_len = u.int_in_range(1..=20)?;
        // Generate values array with the given dtype
        let values = ArbitraryArray::arbitrary_with_config(
            u,
            &ArbitraryArrayConfig {
                dtype: Some(dtype.clone()),
                len: values_len..=values_len,
            },
        )?
        .0;

        // Generate codes that index into the values
        let codes_len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Determine the minimum PType that can represent all indices (max index is values_len - 1)
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

        // Generate codes with optional nullability
        let codes_nullable: Nullability = u.arbitrary()?;
        let codes = match codes_ptype {
            PType::U8 => random_codes::<u8>(u, codes_len, values_len, codes_nullable)?,
            PType::U16 => random_codes::<u16>(u, codes_len, values_len, codes_nullable)?,
            PType::U32 => random_codes::<u32>(u, codes_len, values_len, codes_nullable)?,
            PType::U64 => random_codes::<u64>(u, codes_len, values_len, codes_nullable)?,
            PType::I8 => random_codes::<i8>(u, codes_len, values_len, codes_nullable)?,
            PType::I16 => random_codes::<i16>(u, codes_len, values_len, codes_nullable)?,
            PType::I32 => random_codes::<i32>(u, codes_len, values_len, codes_nullable)?,
            PType::I64 => random_codes::<i64>(u, codes_len, values_len, codes_nullable)?,
            _ => unreachable!(),
        };

        Ok(ArbitraryDictArray(
            DictArray::try_new(codes, values)
                .vortex_expect("DictArray creation should succeed in arbitrary impl"),
        ))
    }
}

/// Generate random codes for a DictArray with a specific integer type.
fn random_codes<T>(
    u: &mut Unstructured,
    len: usize,
    max_value: usize,
    nullability: Nullability,
) -> Result<ArrayRef>
where
    T: NativePType + NumCast,
{
    let codes: Vec<T> = (0..len)
        .map(|_| {
            let idx = u.int_in_range(0..=max_value - 1)?;
            // max_value is bounded by T::MAX in the caller, so conversion always succeeds
            Ok(T::from(idx).vortex_expect("value within type bounds"))
        })
        .collect::<Result<Vec<_>>>()?;
    let validity = random_validity(u, nullability, len)?;
    Ok(PrimitiveArray::new(Buffer::copy_from(codes), validity).into_array())
}
