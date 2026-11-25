// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use zigzag::ZigZag as ExternalZigZag;

use crate::ZigZag;
use crate::ZigZagArray;

pub fn zigzag_encode(parray: ArrayView<'_, Primitive>) -> VortexResult<ZigZagArray> {
    let parray = parray.into_owned();
    let validity = parray.validity()?;
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
    ZigZag::try_new(encoded.into_array())
}

fn zigzag_encode_primitive<T: ExternalZigZag + NativePType>(
    mut values: BufferMut<T>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    values
        .as_mut_slice()
        .map_into_in_place(|value| T::encode(value).as_());

    // SAFETY: `T` and `T::UInt` are same-width integers with identical alignment.
    // Every bit pattern is valid for both types.
    let encoded = unsafe { values.transmute::<T::UInt>() };

    PrimitiveArray::new(encoded.freeze(), validity)
}

pub fn zigzag_decode(parray: PrimitiveArray) -> PrimitiveArray {
    let validity = parray
        .validity()
        .vortex_expect("zigzag validity should be derivable");
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
    mut values: BufferMut<T::UInt>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    values
        .as_mut_slice()
        .map_into_in_place(|value| T::decode(value).as_());

    // SAFETY: `T` and `T::UInt` are same-width integers with identical alignment.
    // Every bit pattern is valid for both types.
    let decoded = unsafe { values.transmute::<T>() };

    PrimitiveArray::new(decoded.freeze(), validity)
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ZigZag;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::i8(PrimitiveArray::from_iter([i8::MIN, i8::MAX].into_iter().chain(-100..100)))]
    #[case::i16(PrimitiveArray::from_iter([i16::MIN, i16::MAX].into_iter().chain(-100..100)))]
    #[case::i32(PrimitiveArray::from_iter([i32::MIN, i32::MAX].into_iter().chain(-100..100)))]
    #[case::i64(PrimitiveArray::from_iter([i64::MIN, i64::MAX].into_iter().chain(-100..100)))]
    fn test_compress(#[case] input: PrimitiveArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = zigzag_encode(input.as_view()).unwrap().into_array();
        assert!(compressed.is::<ZigZag>());
        assert_arrays_eq!(compressed, input, &mut ctx);
    }
}
