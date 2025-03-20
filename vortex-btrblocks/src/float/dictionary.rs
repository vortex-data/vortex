//! Float-specific dictionary encoding implementation.

use vortex_array::Array;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dict::DictArray;
use vortex_dtype::half::f16;
use vortex_error::VortexResult;

use crate::float::stats::{ErasedDistinctValues, FloatStats};

macro_rules! typed_encode {
    ($stats:ident, $typed:ident, $validity:ident, $typ:ty) => {{
        let values: Buffer<$typ> = $typed.values.keys().map(|x| x.0).collect();

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

pub fn dictionary_encode(stats: &FloatStats) -> VortexResult<DictArray> {
    let validity = stats.src.validity();
    match &stats.distinct_values {
        ErasedDistinctValues::F16(typed) => typed_encode!(stats, typed, validity, f16),
        ErasedDistinctValues::F32(typed) => typed_encode!(stats, typed, validity, f32),
        ErasedDistinctValues::F64(typed) => typed_encode!(stats, typed, validity, f64),
    }
}

struct DictEncoder;

trait Encode<T, I> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<I>;
}

macro_rules! impl_encode {
    ($typ:ty, $utyp:ty) => { impl_encode!($typ, $utyp, u8, u16, u32); };
    ($typ:ty, $utyp:ty, $($ityp:ty),+) => {
        $(
        impl Encode<$typ, $ityp> for DictEncoder {
            #[allow(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<$ityp> {
                let mut codes =
                    vortex_array::aliases::hash_map::HashMap::<$utyp, $ityp>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value.to_bits(), code as $ityp);
                }

                let mut output = vortex_buffer::BufferMut::with_capacity(values.len());
                for value in values {
                    // Any code lookups which fail are for nulls, so their value
                    // does not matter.
                    output.push(codes.get(&value.to_bits()).copied().unwrap_or_default());
                }

                return output.freeze();
            }
        }
        )*
    };
}

impl_encode!(f16, u16);
impl_encode!(f32, u32);
impl_encode!(f64, u64);

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, PrimitiveArray};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ToCanonical};
    use vortex_buffer::buffer;

    use crate::CompressorStats;
    use crate::float::dictionary::dictionary_encode;
    use crate::float::stats::FloatStats;

    #[test]
    fn test_float_dict_encode() {
        // Create an array that has some nulls
        let values = buffer![1f32, 2f32, 2f32, 0f32, 1f32];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(values, validity);

        let stats = FloatStats::generate(&array);
        let dict_array = dictionary_encode(&stats).unwrap();
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let undict = dict_array.to_primitive().unwrap();

        // We just use code zero but it doesn't really matter.
        // We can just shove a whole validity buffer in there instead.
        assert_eq!(undict.as_slice::<f32>(), &[1f32, 2f32, 2f32, 1f32, 1f32]);
    }
}
