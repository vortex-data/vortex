// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use zigzag::ZigZag as ExternalZigZag;

use crate::ZigZagArray;
use crate::ZigZagData;
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
    ZigZagArray::try_from_data(ZigZagData::try_new(encoded.into_array())?)
}

fn zigzag_encode_primitive<T: ExternalZigZag + NativePType>(
    values: BufferMut<T>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    PrimitiveArray::new(
        values.map_each_in_place(|v| T::encode(v)).freeze(),
        validity,
    )
}

pub fn zigzag_decode(parray: PrimitiveArray) -> PrimitiveArray {
    let validity = parray.validity();
    match parray.ptype() {
        PType::U8 => zigzag_decode_primitive::<i8>(parray.into_buffer_mut(), validity),
        PType::U16 => zigzag_decode_primitive::<i16>(parray.into_buffer_mut(), validity),
        PType::U32 => zigzag_decode_primitive::<i32>(parray.into_buffer_mut(), validity),
        PType::U64 => zigzag_decode_primitive::<i64>(parray.into_buffer_mut(), validity),
        _ => vortex_panic!(
            "ZigZag can only decode unsigned integers, got {}",
            parray.ptype()
        ),
    }
}

fn zigzag_decode_primitive<T: ExternalZigZag + NativePType>(
    values: BufferMut<T::UInt>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    PrimitiveArray::new(
        values.map_each_in_place(|v| T::decode(v)).freeze(),
        validity,
    )
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::assert_arrays_eq;

    use super::*;
    use crate::ZigZag;

    #[test]
    fn test_compress_i8() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i8..100))
            .unwrap()
            .into_array();
        assert!(compressed.is::<ZigZag>());
        assert_arrays_eq!(
            compressed.to_primitive(),
            PrimitiveArray::from_iter(-100_i8..100)
        );
    }
    #[test]
    fn test_compress_i16() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i16..100))
            .unwrap()
            .into_array();
        assert!(compressed.is::<ZigZag>());
        assert_arrays_eq!(
            compressed.to_primitive(),
            PrimitiveArray::from_iter(-100_i16..100)
        );
    }
    #[test]
    fn test_compress_i32() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i32..100))
            .unwrap()
            .into_array();
        assert!(compressed.is::<ZigZag>());
        assert_arrays_eq!(
            compressed.to_primitive(),
            PrimitiveArray::from_iter(-100_i32..100)
        );
    }
    #[test]
    fn test_compress_i64() {
        let compressed = zigzag_encode(PrimitiveArray::from_iter(-100_i64..100))
            .unwrap()
            .into_array();
        assert!(compressed.is::<ZigZag>());
        assert_arrays_eq!(
            compressed.to_primitive(),
            PrimitiveArray::from_iter(-100_i64..100)
        );
    }
}
