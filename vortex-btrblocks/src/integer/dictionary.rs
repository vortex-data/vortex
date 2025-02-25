//! Dictionary compressor that reuses the unique values in the `IntegerStats`.

use vortex_array::Array;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dict::DictArray;
use vortex_error::VortexResult;

use crate::integer::IntegerStats;
use crate::integer::stats::ErasedStats;

macro_rules! typed_encode {
    ($stats:ident, $typed:ident, $validity:ident, $typ:ty) => {{
        let values: Buffer<$typ> = $typed.distinct_values.keys().copied().collect();

        let max_code = values.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf =
                <DictEncoder as Encode<$typ, u8>>::encode(&values, $stats.src.as_slice::<$typ>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else if max_code <= u16::MAX as usize {
            let buf =
                <DictEncoder as Encode<$typ, u16>>::encode(&values, $stats.src.as_slice::<$typ>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        } else {
            let buf =
                <DictEncoder as Encode<$typ, u32>>::encode(&values, $stats.src.as_slice::<$typ>());
            PrimitiveArray::new(buf, $validity.clone()).into_array()
        };

        let values_validity = match $validity {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };

        let values = PrimitiveArray::new(values, values_validity).into_array();
        DictArray::try_new(codes, values)
    }};
}

#[allow(clippy::cognitive_complexity)]
pub fn dictionary_encode(stats: &IntegerStats) -> VortexResult<DictArray> {
    // We need to preserve the nullability somehow from the original
    let src_validity = stats.src.validity();

    match &stats.typed {
        ErasedStats::U8(typed) => typed_encode!(stats, typed, src_validity, u8),
        ErasedStats::U16(typed) => typed_encode!(stats, typed, src_validity, u16),
        ErasedStats::U32(typed) => typed_encode!(stats, typed, src_validity, u32),
        ErasedStats::U64(typed) => typed_encode!(stats, typed, src_validity, u64),
        ErasedStats::I8(typed) => typed_encode!(stats, typed, src_validity, i8),
        ErasedStats::I16(typed) => typed_encode!(stats, typed, src_validity, i16),
        ErasedStats::I32(typed) => typed_encode!(stats, typed, src_validity, i32),
        ErasedStats::I64(typed) => typed_encode!(stats, typed, src_validity, i64),
    }
}

struct DictEncoder;

trait Encode<T, I> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<I>;
}

macro_rules! impl_encode {
    ($typ:ty) => { impl_encode!($typ, u8, u16, u32); };
    ($typ:ty, $($ityp:ty),+) => {
        $(
        impl Encode<$typ, $ityp> for DictEncoder {
            #[allow(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<$ityp> {
                let mut codes =
                    vortex_array::aliases::hash_map::HashMap::<$typ, $ityp>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value, code as $ityp);
                }

                let mut output = vortex_buffer::BufferMut::with_capacity(values.len());
                for value in values {
                    // Any code lookups which fail are for nulls, so their value
                    // does not matter.
                    // SAFETY: we have exactly sized output to be as large as values.
                    unsafe { output.push_unchecked(codes.get(value).copied().unwrap_or_default()) };
                }

                return output.freeze();
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
    use vortex_array::Array;
    use vortex_array::arrays::{BoolArray, PrimitiveArray};
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::CompressorStats;
    use crate::integer::IntegerStats;
    use crate::integer::dictionary::dictionary_encode;

    #[test]
    fn test_dict_encode_integer_stats() {
        // Create an array that has some nulls
        let data = buffer![100i32, 200, 100, 0, 100];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(data, validity);

        let stats = IntegerStats::generate(&array);
        let dict_array = dictionary_encode(&stats).unwrap();
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let undict = dict_array.to_canonical().unwrap().into_primitive().unwrap();

        // We just use code zero but it doesn't really matter.
        // We can just shove a whole validity buffer in there instead.
        assert_eq!(undict.as_slice::<i32>(), &[100i32, 200, 100, 100, 100]);
    }
}
