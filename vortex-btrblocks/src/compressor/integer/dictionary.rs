// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary compressor that reuses the unique values in the `IntegerStats`.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for external compatibility.

use std::hash::Hash;

use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use super::IntegerStats;
use super::stats::ErasedStats;

/// Encode values using a HashMap built from sorted distinct values.
///
/// Sorting the distinct values means the DictArray's values child is sorted,
/// which can help downstream compression of the values array.
macro_rules! typed_encode_hashmap {
    ($stats:ident, $typed:ident, $validity:ident, $typ:ty) => {{
        let mut values: Vec<$typ> = $typed.distinct_values.keys().map(|x| x.0).collect();
        values.sort_unstable();
        let values_buf: Buffer<$typ> = values.into();

        let max_code = values_buf.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf =
                encode_hashmap::<$typ, u8>(values_buf.as_slice(), $stats.src.as_slice::<$typ>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else if max_code <= u16::MAX as usize {
            let buf =
                encode_hashmap::<$typ, u16>(values_buf.as_slice(), $stats.src.as_slice::<$typ>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else {
            let buf =
                encode_hashmap::<$typ, u32>(values_buf.as_slice(), $stats.src.as_slice::<$typ>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        };

        let values_validity = match $validity {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };

        let values = PrimitiveArray::new(values_buf, values_validity).into_array();
        // SAFETY: invariants enforced by encode_hashmap
        unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) }
    }};
}

/// Encode values into dictionary codes using FxHashMap for fast lookups.
#[allow(clippy::cast_possible_truncation)]
#[inline]
fn encode_hashmap<T: Copy + Eq + Hash, I>(
    sorted_distinct: &[T],
    values: &[T],
) -> Buffer<I>
where
    I: Copy + Default + TryFrom<usize>,
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut codes =
        vortex_utils::aliases::hash_map::HashMap::<T, I>::with_capacity(sorted_distinct.len());
    for (code, &value) in sorted_distinct.iter().enumerate() {
        codes.insert(value, I::try_from(code).unwrap_or_default());
    }

    let mut output = BufferMut::with_capacity(values.len());
    for &value in values {
        // Any code lookups which fail are for nulls, so their value does not matter.
        // SAFETY: we have exactly sized output to be as large as values.
        unsafe { output.push_unchecked(codes.get(&value).copied().unwrap_or_default()) };
    }
    output.freeze()
}

// ── u8 direct lookup table ──────────────────────────────────────────────────

/// Specialized encoding for u8 values using a 256-entry direct lookup table.
/// O(1) per value with no hashing overhead.
#[allow(clippy::cast_possible_truncation)]
fn encode_u8_direct<I: Copy + Default + TryFrom<usize>>(
    sorted_distinct: &[u8],
    values: &[u8],
) -> Buffer<I>
where
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut table = [I::default(); 256];
    for (code, &value) in sorted_distinct.iter().enumerate() {
        table[value as usize] = I::try_from(code).unwrap_or_default();
    }

    let mut output = BufferMut::with_capacity(values.len());
    for &value in values {
        // SAFETY: we have exactly sized output to be as large as values.
        unsafe { output.push_unchecked(table[value as usize]) };
    }
    output.freeze()
}

macro_rules! typed_encode_u8_direct {
    ($stats:ident, $typed:ident, $validity:ident) => {{
        let mut values: Vec<u8> = $typed.distinct_values.keys().map(|x| x.0).collect();
        values.sort_unstable();
        let values_buf: Buffer<u8> = values.into();

        let max_code = values_buf.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf = encode_u8_direct::<u8>(values_buf.as_slice(), $stats.src.as_slice::<u8>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else {
            let buf = encode_u8_direct::<u16>(values_buf.as_slice(), $stats.src.as_slice::<u8>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        };

        let values_validity = match $validity {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };

        let values = PrimitiveArray::new(values_buf, values_validity).into_array();
        unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) }
    }};
}

// ── i8 direct lookup table ──────────────────────────────────────────────────

/// Specialized encoding for i8 values using a 256-entry direct lookup table.
#[allow(clippy::cast_possible_truncation)]
fn encode_i8_direct<I: Copy + Default + TryFrom<usize>>(
    sorted_distinct: &[i8],
    values: &[i8],
) -> Buffer<I>
where
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut table = [I::default(); 256];
    for (code, &value) in sorted_distinct.iter().enumerate() {
        table[value as u8 as usize] = I::try_from(code).unwrap_or_default();
    }

    let mut output = BufferMut::with_capacity(values.len());
    for &value in values {
        // SAFETY: we have exactly sized output to be as large as values.
        unsafe { output.push_unchecked(table[value as u8 as usize]) };
    }
    output.freeze()
}

macro_rules! typed_encode_i8_direct {
    ($stats:ident, $typed:ident, $validity:ident) => {{
        let mut values: Vec<i8> = $typed.distinct_values.keys().map(|x| x.0).collect();
        values.sort_unstable();
        let values_buf: Buffer<i8> = values.into();

        let max_code = values_buf.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf = encode_i8_direct::<u8>(values_buf.as_slice(), $stats.src.as_slice::<i8>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else {
            let buf = encode_i8_direct::<u16>(values_buf.as_slice(), $stats.src.as_slice::<i8>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        };

        let values_validity = match $validity {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };

        let values = PrimitiveArray::new(values_buf, values_validity).into_array();
        unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) }
    }};
}

/// Compresses an integer array into a dictionary array according to attached stats.
///
/// Uses direct lookup tables for u8/i8 (O(1) per value, no hashing) and
/// FxHashMap for larger integer types. Values are sorted for better
/// downstream compression.
pub fn dictionary_encode(stats: &IntegerStats) -> DictArray {
    let src_validity = stats.src.validity();

    match &stats.typed {
        ErasedStats::U8(typed) => typed_encode_u8_direct!(stats, typed, src_validity),
        ErasedStats::I8(typed) => typed_encode_i8_direct!(stats, typed, src_validity),
        ErasedStats::U16(typed) => typed_encode_hashmap!(stats, typed, src_validity, u16),
        ErasedStats::U32(typed) => typed_encode_hashmap!(stats, typed, src_validity, u32),
        ErasedStats::U64(typed) => typed_encode_hashmap!(stats, typed, src_validity, u64),
        ErasedStats::I16(typed) => typed_encode_hashmap!(stats, typed, src_validity, i16),
        ErasedStats::I32(typed) => typed_encode_hashmap!(stats, typed, src_validity, i32),
        ErasedStats::I64(typed) => typed_encode_hashmap!(stats, typed, src_validity, i64),
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::IntegerStats;
    use super::dictionary_encode;
    use crate::CompressorStats;

    #[test]
    fn test_dict_encode_integer_stats() {
        // Create an array that has some nulls
        let data = buffer![100i32, 200, 100, 0, 100];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(data, validity);

        let stats = IntegerStats::generate(&array);
        let dict_array = dictionary_encode(&stats);
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let undict = dict_array;

        let expected = PrimitiveArray::new(
            buffer![100i32, 200, 100, 100, 100],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        assert_arrays_eq!(undict.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_dict_encode_u8() {
        let data = buffer![10u8, 20, 10, 30, 20];
        let array = PrimitiveArray::new(data, Validity::NonNullable);

        let stats = IntegerStats::generate(&array);
        let dict_array = dictionary_encode(&stats);
        assert_eq!(dict_array.values().len(), 3);
        assert_eq!(dict_array.codes().len(), 5);

        let expected =
            PrimitiveArray::new(buffer![10u8, 20, 10, 30, 20], Validity::NonNullable).into_array();
        assert_arrays_eq!(dict_array.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_dict_encode_i8() {
        let data = buffer![-5i8, 10, -5, 10, -5];
        let array = PrimitiveArray::new(data, Validity::NonNullable);

        let stats = IntegerStats::generate(&array);
        let dict_array = dictionary_encode(&stats);
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected =
            PrimitiveArray::new(buffer![-5i8, 10, -5, 10, -5], Validity::NonNullable).into_array();
        assert_arrays_eq!(dict_array.as_ref(), expected.as_ref());
    }
}
