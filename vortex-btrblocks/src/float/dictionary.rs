//! Float-specific dictionary encoding implementation.

use vortex_array::array::PrimitiveArray;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_dict::DictArray;
use vortex_dtype::half::f16;
use vortex_error::VortexResult;

use crate::float::stats::{ErasedDistinctValues, FloatStats};

macro_rules! typed_encode {
    ($stats:ident, $typed:ident, $validity:ident, $typ:ty) => {{
        let values: Buffer<$typ> = $typed
            .values
            .keys()
            .copied()
            .map(|x| <$typ>::from_bits(x))
            .collect();
        let codes = <DictEncoder as Encode<$typ>>::encode(&values, $stats.src.as_slice::<$typ>());
        // preserve the original validity
        let codes = PrimitiveArray::new(codes, $validity).into_array();
        DictArray::try_new(codes, values.into_array())
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

trait Encode<T> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<u32>;
}

macro_rules! impl_encode {
    ($typ:ty, $utyp:ty) => {
        impl Encode<$typ> for DictEncoder {
            #[allow(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<u32> {
                let mut codes =
                    vortex_array::aliases::hash_map::HashMap::<$utyp, u32>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value.to_bits(), code as u32);
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
    };
}

impl_encode!(f16, u16);
impl_encode!(f32, u32);
impl_encode!(f64, u64);

#[cfg(test)]
mod tests {
    use vortex_array::array::{BoolArray, PrimitiveArray};
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, IntoCanonical};
    use vortex_buffer::buffer;

    use crate::float::dictionary::dictionary_encode;
    use crate::float::stats::FloatStats;
    use crate::CompressorStats;

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

        let undict = dict_array
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap();

        // We just use code zero but it doesn't really matter.
        // We can just shove a whole validity buffer in there instead.
        assert_eq!(undict.as_slice::<f32>(), &[1f32, 2f32, 2f32, 1f32, 1f32]);
    }
}
