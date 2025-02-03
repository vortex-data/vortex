use num_traits::{PrimInt, WrappingAdd, WrappingSub};
use vortex_array::array::PrimitiveArray;
use vortex_array::stats::Stat;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{IntoArray, IntoArrayVariant};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::Scalar;

use crate::FoRArray;

pub fn for_compress(array: PrimitiveArray) -> VortexResult<FoRArray> {
    let min = array
        .statistics()
        .compute(Stat::Min)
        .ok_or_else(|| vortex_err!("Min stat not found"))?;

    let dtype = array.dtype().clone();
    let encoded = match_each_integer_ptype!(array.ptype(), |$T| {
        let unsigned_ptype = array.ptype().to_unsigned();
        compress_primitive::<$T>(array, $T::try_from(&min)?)
            .reinterpret_cast(unsigned_ptype)
            .into_array()
    });
    FoRArray::try_new(encoded, Scalar::new(dtype, min))
}

#[allow(clippy::cast_possible_truncation)]
fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: PrimitiveArray,
    min: T,
) -> PrimitiveArray {
    parray.map_each::<T, _, _>(|v| v.wrapping_sub(&min))
}

pub fn decompress(array: FoRArray) -> VortexResult<PrimitiveArray> {
    let ptype = array.ptype();

    // TODO(ngates): do we need this to be into_encoded() somehow?
    let encoded = array.encoded().into_primitive()?.reinterpret_cast(ptype);
    let validity = encoded.validity();

    Ok(match_each_integer_ptype!(ptype, |$T| {
        let min = array.reference_scalar()
            .as_primitive()
            .typed_value::<$T>()
            .ok_or_else(|| vortex_err!("expected reference to be non-null"))?;
        if min == 0 {
            encoded
        } else {
            PrimitiveArray::new(
                decompress_primitive(encoded.into_buffer_mut::<$T>(), min),
                validity,
            )
        }
    }))
}

fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: BufferMut<T>,
    min: T,
) -> Buffer<T> {
    values.map_each(move |v| v.wrapping_add(&min)).freeze()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::compute::scalar_at;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn test_compress() {
        // Create a range offset by a million
        let array = PrimitiveArray::new(
            (0u32..10_000).map(|v| v + 1_000_000).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let compressed = for_compress(array).unwrap();
        assert_eq!(
            u32::try_from(compressed.reference_scalar()).unwrap(),
            1_000_000u32
        );
    }

    #[test]
    fn test_zeros() {
        let array = PrimitiveArray::new(buffer![0i32; 100], Validity::NonNullable);
        assert!(array.statistics().to_set().into_iter().next().is_none());

        let compressed = for_compress(array.clone()).unwrap();
        assert_eq!(compressed.dtype(), array.dtype());
        assert!(compressed.dtype().is_signed_int());
        assert!(compressed.encoded().dtype().is_unsigned_int());

        let constant = compressed.encoded().as_constant().unwrap();
        assert_eq!(constant, Scalar::from(0u32));
    }

    #[test]
    fn test_decompress() {
        // Create a range offset by a million
        let array = PrimitiveArray::from_iter((0u32..100_000).step_by(1024).map(|v| v + 1_000_000));
        let compressed = for_compress(array.clone()).unwrap();
        let decompressed = compressed.into_primitive().unwrap();
        assert_eq!(decompressed.as_slice::<u32>(), array.as_slice::<u32>());
    }

    #[test]
    fn test_overflow() {
        let array = PrimitiveArray::from_iter(i8::MIN..=i8::MAX);
        let compressed = for_compress(array.clone()).unwrap();
        assert_eq!(
            i8::MIN,
            compressed
                .reference_scalar()
                .as_primitive()
                .typed_value::<i8>()
                .unwrap()
        );

        let encoded = compressed.encoded().into_primitive().unwrap();
        let encoded_bytes: &[u8] = encoded.as_slice::<u8>();
        let unsigned: Vec<u8> = (0..=u8::MAX).collect_vec();
        assert_eq!(encoded_bytes, unsigned.as_slice());

        let decompressed = compressed.as_ref().clone().into_primitive().unwrap();
        assert_eq!(decompressed.as_slice::<i8>(), array.as_slice::<i8>());
        array
            .as_slice::<i8>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                assert_eq!(
                    *v,
                    i8::try_from(scalar_at(&compressed, i).unwrap().as_ref()).unwrap()
                );
            });
    }
}
