use vortex_array::array::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::IntoArrayData;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, PType};
use vortex_error::{vortex_bail, VortexResult};
use zigzag::ZigZag as ExternalZigZag;

use crate::ZigZagArray;

pub fn zigzag_encode(parray: PrimitiveArray) -> VortexResult<ZigZagArray> {
    let validity = parray.validity();
    let encoded = match parray.ptype() {
        PType::I8 => zigzag_encode_primitive::<i8>(parray.into_buffer_mut(), validity),
        PType::I16 => zigzag_encode_primitive::<i16>(parray.into_buffer_mut(), validity),
        PType::I32 => zigzag_encode_primitive::<i32>(parray.into_buffer_mut(), validity),
        PType::I64 => zigzag_encode_primitive::<i64>(parray.into_buffer_mut(), validity),
        _ => vortex_bail!(
            "ZigZag can only encode signed integers, got {}",
            parray.ptype()
        ),
    };
    ZigZagArray::try_new(encoded.into_array())
}

fn zigzag_encode_primitive<T: ExternalZigZag + NativePType>(
    values: BufferMut<T>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    PrimitiveArray::new(values.map_each(|v| T::encode(v)).freeze(), validity)
}

pub fn zigzag_decode(parray: PrimitiveArray) -> VortexResult<PrimitiveArray> {
    let validity = parray.validity();
    let decoded = match parray.ptype() {
        PType::U8 => zigzag_decode_primitive::<i8>(parray.into_buffer_mut(), validity),
        PType::U16 => zigzag_decode_primitive::<i16>(parray.into_buffer_mut(), validity),
        PType::U32 => zigzag_decode_primitive::<i32>(parray.into_buffer_mut(), validity),
        PType::U64 => zigzag_decode_primitive::<i64>(parray.into_buffer_mut(), validity),
        _ => vortex_bail!(
            "ZigZag can only decode unsigned integers, got {}",
            parray.ptype()
        ),
    };
    Ok(decoded)
}

fn zigzag_decode_primitive<T: ExternalZigZag + NativePType>(
    values: BufferMut<T::UInt>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    PrimitiveArray::new(values.map_each(|v| T::decode(v)).freeze(), validity)
}

#[cfg(test)]
mod test {
    use vortex_array::vtable::EncodingVTable;
    use vortex_array::IntoArrayVariant as _;

    use super::*;
    use crate::ZigZagEncoding;

    #[test]
    fn test_compress_i8() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i8..100)).unwrap();
        assert_eq!(compressed.as_ref().encoding().id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.into_primitive().unwrap().as_slice::<i8>(),
            (-100_i8..100).collect::<Vec<_>>()
        );
    }
    #[test]
    fn test_compress_i16() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i16..100)).unwrap();
        assert_eq!(compressed.as_ref().encoding().id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.into_primitive().unwrap().as_slice::<i16>(),
            (-100_i16..100).collect::<Vec<_>>()
        );
    }
    #[test]
    fn test_compress_i32() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i32..100)).unwrap();
        assert_eq!(compressed.as_ref().encoding().id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.into_primitive().unwrap().as_slice::<i32>(),
            (-100_i32..100).collect::<Vec<_>>()
        );
    }
    #[test]
    fn test_compress_i64() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i64..100)).unwrap();
        assert_eq!(compressed.as_ref().encoding().id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.into_primitive().unwrap().as_slice::<i64>(),
            (-100_i64..100).collect::<Vec<_>>()
        );
    }
}
