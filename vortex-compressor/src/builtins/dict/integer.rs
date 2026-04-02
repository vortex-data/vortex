// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary compressor that reuses the unique values in the [`IntegerStats`].
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

use crate::stats::IntegerErasedStats;
use crate::stats::IntegerStats;

/// Encodes a typed integer array into a [`DictArray`] using the pre-computed distinct values.
macro_rules! typed_encode {
    ($stats:ident, $typed:ident, $validity:ident, $typ:ty) => {{
        let distinct = $typed.distinct().vortex_expect(
            "this must be present since `DictScheme` declared that we need distinct values",
        );

        let values: Buffer<$typ> = distinct.distinct_values().keys().map(|x| x.0).collect();

        let max_code = values.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u8>>::encode(
                &values,
                $stats.source().as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else if max_code <= u16::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u16>>::encode(
                &values,
                $stats.source().as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else {
            let buf = <DictEncoder as Encode<$typ, u32>>::encode(
                &values,
                $stats.source().as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        };

        let values_validity = match $validity {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };

        let values = PrimitiveArray::new(values, values_validity).into_array();
        // SAFETY: invariants enforced in DictEncoder.
        unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) }
    }};
}

/// Compresses an integer array into a dictionary array according to attached stats.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity from match on all integer types"
)]
pub fn dictionary_encode(stats: &IntegerStats) -> DictArray {
    let src_validity = stats.source().validity();

    match stats.erased() {
        IntegerErasedStats::U8(typed) => typed_encode!(stats, typed, src_validity, u8),
        IntegerErasedStats::U16(typed) => typed_encode!(stats, typed, src_validity, u16),
        IntegerErasedStats::U32(typed) => typed_encode!(stats, typed, src_validity, u32),
        IntegerErasedStats::U64(typed) => typed_encode!(stats, typed, src_validity, u64),
        IntegerErasedStats::I8(typed) => typed_encode!(stats, typed, src_validity, i8),
        IntegerErasedStats::I16(typed) => typed_encode!(stats, typed, src_validity, i16),
        IntegerErasedStats::I32(typed) => typed_encode!(stats, typed, src_validity, i32),
        IntegerErasedStats::I64(typed) => typed_encode!(stats, typed, src_validity, i64),
    }
}

/// Stateless encoder that maps values to dictionary codes via a `HashMap`.
struct DictEncoder;

/// Trait for encoding values of type `T` into codes of type `I`.
trait Encode<T, I> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<I>;
}

/// Implements [`Encode`] for an integer type with all code width variants (u8, u16, u32).
macro_rules! impl_encode {
    ($typ:ty) => { impl_encode!($typ, u8, u16, u32); };
    ($typ:ty, $($ityp:ty),+) => {
        $(
        impl Encode<$typ, $ityp> for DictEncoder {
            #[allow(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<$ityp> {
                let mut codes =
                    vortex_utils::aliases::hash_map::HashMap::<$typ, $ityp>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value, code as $ityp);
                }

                let mut output = vortex_buffer::BufferMut::with_capacity(values.len());
                for value in values {
                    // Any code lookups which fail are for nulls, so their value does not matter.
                    // SAFETY: we have exactly sized output to be as large as values.
                    unsafe { output.push_unchecked(codes.get(value).copied().unwrap_or_default()) };
                }

                output.freeze()
            }
        }
        )*
    };
}

impl_encode!(u8);
impl_encode!(u16);
impl_encode!(u32);
impl_encode!(u64);
impl_encode!(i8);
impl_encode!(i16);
impl_encode!(i32);
impl_encode!(i64);

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::dictionary_encode;
    use crate::stats::IntegerStats;

    #[test]
    fn test_dict_encode_integer_stats() {
        let data = buffer![100i32, 200, 100, 0, 100];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(data, validity);

        let stats = IntegerStats::generate_opts(
            &array,
            crate::stats::GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let dict_array = dictionary_encode(&stats);
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected = PrimitiveArray::new(
            buffer![100i32, 200, 100, 100, 100],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        let undict = dict_array.as_array().to_primitive().into_array();
        assert_arrays_eq!(undict, expected);
    }
}
