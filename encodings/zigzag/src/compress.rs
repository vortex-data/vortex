// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, PType};
use vortex_error::{VortexResult, vortex_bail};
use zigzag::ZigZag as ExternalZigZag;

use crate::ZigZagArray;

pub fn zigzag_encode(parray: PrimitiveArray) -> VortexResult<ZigZagArray> {
    use PType::*;

    let validity = parray.validity().clone();
    let encoded = match parray.ptype() {
        I8 => zigzag_encode_primitive::<i8>(parray.into_buffer_mut(), validity),
        I16 => zigzag_encode_primitive::<i16>(parray.into_buffer_mut(), validity),
        I32 => zigzag_encode_primitive::<i32>(parray.into_buffer_mut(), validity),
        I64 => zigzag_encode_primitive::<i64>(parray.into_buffer_mut(), validity),
        U8 | U16 | U32 | U64 | F16 | F32 | F64 => {
            vortex_bail!(
                "ZigZag can only encode signed integers, got {}",
                parray.ptype()
            )
        }
    };
    ZigZagArray::try_new(encoded.to_array())
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
    use PType::*;

    let validity = parray.validity().clone();
    let decoded = match parray.ptype() {
        U8 => zigzag_decode_primitive::<i8>(parray.into_buffer_mut(), validity),
        U16 => zigzag_decode_primitive::<i16>(parray.into_buffer_mut(), validity),
        U32 => zigzag_decode_primitive::<i32>(parray.into_buffer_mut(), validity),
        U64 => zigzag_decode_primitive::<i64>(parray.into_buffer_mut(), validity),
        I8 | I16 | I32 | I64 | F16 | F32 | F64 => {
            vortex_bail!(
                "ZigZag can only decode unsigned integers, got {}",
                parray.ptype()
            )
        }
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
    use vortex_array::ToCanonical;

    use super::*;
    use crate::ZigZagEncoding;

    #[test]
    fn test_compress_i8() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i8..100)).unwrap();
        assert_eq!(compressed.encoding_id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.to_primitive().unwrap().as_slice::<i8>(),
            (-100_i8..100).collect::<Vec<_>>()
        );
    }
    #[test]
    fn test_compress_i16() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i16..100)).unwrap();
        assert_eq!(compressed.encoding_id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.to_primitive().unwrap().as_slice::<i16>(),
            (-100_i16..100).collect::<Vec<_>>()
        );
    }
    #[test]
    fn test_compress_i32() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i32..100)).unwrap();
        assert_eq!(compressed.encoding_id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.to_primitive().unwrap().as_slice::<i32>(),
            (-100_i32..100).collect::<Vec<_>>()
        );
    }
    #[test]
    fn test_compress_i64() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i64..100)).unwrap();
        assert_eq!(compressed.encoding_id(), ZigZagEncoding.id());
        assert_eq!(
            compressed.to_primitive().unwrap().as_slice::<i64>(),
            (-100_i64..100).collect::<Vec<_>>()
        );
    }
}
