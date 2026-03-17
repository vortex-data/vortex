// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for external compatibility.

use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::half::f16;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use super::stats::ErasedDistinctValues;
use super::stats::FloatStats;

/// Trait for converting float types to their bit representation for total ordering.
trait FloatBits: Copy {
    type Bits: Ord + Copy;
    fn to_sort_key(self) -> Self::Bits;
}

impl FloatBits for f16 {
    type Bits = u16;
    fn to_sort_key(self) -> u16 {
        let bits = self.to_bits();
        // Convert to a representation where total ordering works:
        // if sign bit is set, flip all bits; otherwise flip only sign bit.
        if bits & 0x8000 != 0 {
            !bits
        } else {
            bits ^ 0x8000
        }
    }
}

impl FloatBits for f32 {
    type Bits = u32;
    fn to_sort_key(self) -> u32 {
        let bits = self.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }
}

impl FloatBits for f64 {
    type Bits = u64;
    fn to_sort_key(self) -> u64 {
        let bits = self.to_bits();
        if bits & 0x8000_0000_0000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000_0000_0000
        }
    }
}

macro_rules! typed_encode {
    ($stats:ident, $typed:ident, $validity:ident, $typ:ty, $utyp:ty) => {{
        // Collect and sort distinct values using total ordering on bit patterns
        let mut values: Vec<$typ> = $typed.values.iter().map(|x| x.0).collect();
        values.sort_unstable_by_key(|v| v.to_sort_key());
        let values_buf: Buffer<$typ> = values.into();

        let max_code = values_buf.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf = encode_float_sorted::<$typ, $utyp, u8>(
                values_buf.as_slice(),
                $stats.src.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else if max_code <= u16::MAX as usize {
            let buf = encode_float_sorted::<$typ, $utyp, u16>(
                values_buf.as_slice(),
                $stats.src.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else {
            let buf = encode_float_sorted::<$typ, $utyp, u32>(
                values_buf.as_slice(),
                $stats.src.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        };

        let values_validity = match $validity {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };
        let values = PrimitiveArray::new(values_buf, values_validity).into_array();

        // SAFETY: enforced by the encode function
        unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) }
    }};
}

/// Compresses a floating-point array into a dictionary arrays according to attached stats.
pub fn dictionary_encode(stats: &FloatStats) -> DictArray {
    let validity = stats.src.validity();
    match &stats.distinct_values {
        ErasedDistinctValues::F16(typed) => typed_encode!(stats, typed, validity, f16, u16),
        ErasedDistinctValues::F32(typed) => typed_encode!(stats, typed, validity, f32, u32),
        ErasedDistinctValues::F64(typed) => typed_encode!(stats, typed, validity, f64, u64),
    }
}

/// Encode float values into dictionary codes using sorted distinct values with binary search
/// on bit representations for total ordering.
#[allow(clippy::cast_possible_truncation)]
#[inline]
fn encode_float_sorted<T, U, I>(sorted_distinct: &[T], values: &[T]) -> Buffer<I>
where
    T: FloatBits<Bits = U> + Copy,
    U: Ord + Copy,
    I: Copy + Default + TryFrom<usize>,
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    // Pre-compute sort keys for the distinct values
    let distinct_keys: Vec<U> = sorted_distinct.iter().map(|v| v.to_sort_key()).collect();

    let mut output = BufferMut::with_capacity(values.len());

    if distinct_keys.len() <= 16 {
        for &value in values {
            let key = value.to_sort_key();
            let code = distinct_keys
                .iter()
                .position(|&d| d == key)
                .map(|idx| I::try_from(idx).unwrap_or_default())
                .unwrap_or_default();
            // SAFETY: we have exactly sized output to be as large as values.
            unsafe { output.push_unchecked(code) };
        }
    } else {
        for &value in values {
            let key = value.to_sort_key();
            let code = distinct_keys
                .binary_search(&key)
                .map(|idx| I::try_from(idx).unwrap_or_default())
                .unwrap_or_default();
            // SAFETY: we have exactly sized output to be as large as values.
            unsafe { output.push_unchecked(code) };
        }
    }

    output.freeze()
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

    use super::super::FloatStats;
    use crate::CompressorStats;
    use crate::compressor::float::dictionary::dictionary_encode;

    #[test]
    fn test_float_dict_encode() {
        // Create an array that has some nulls
        let values = buffer![1f32, 2f32, 2f32, 0f32, 1f32];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(values, validity);

        let stats = FloatStats::generate(&array);
        let dict_array = dictionary_encode(&stats);
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let undict = dict_array;

        // We just use code zero but it doesn't really matter.
        // We can just shove a whole validity buffer in there instead.
        let expected = PrimitiveArray::new(
            buffer![1f32, 2f32, 2f32, 1f32, 1f32],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        assert_arrays_eq!(undict.as_ref(), expected.as_ref());
    }
}
