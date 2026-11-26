// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use zigzag::ZigZag as ExternalZigZag;

use crate::ZigZag;
use crate::ZigZagArray;

pub fn zigzag_encode(parray: ArrayView<'_, Primitive>) -> VortexResult<ZigZagArray> {
    let validity = parray.validity()?;
    let encoded = match parray.ptype() {
        PType::I8 => zigzag_encode_primitive::<i8>(parray.to_buffer().as_slice(), validity),
        PType::I16 => zigzag_encode_primitive::<i16>(parray.to_buffer().as_slice(), validity),
        PType::I32 => zigzag_encode_primitive::<i32>(parray.to_buffer().as_slice(), validity),
        PType::I64 => zigzag_encode_primitive::<i64>(parray.to_buffer().as_slice(), validity),
        _ => vortex_bail!(
            "ZigZag can only encode signed integers, got {}",
            parray.ptype()
        ),
    };
    ZigZag::try_new(encoded.into_array())
}

fn zigzag_encode_primitive<T: ExternalZigZag + NativePType>(
    values: &[T],
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    let len = values.len();
    let mut out = BufferMut::<T::UInt>::with_capacity(len);
    values.map_into(&mut out.spare_capacity_mut()[..len], T::encode);

    // SAFETY: `map_into` writes every lane of the `len` elements.
    unsafe { out.set_len(len) };

    PrimitiveArray::new(out.freeze(), validity)
}

pub fn zigzag_decode(parray: PrimitiveArray) -> PrimitiveArray {
    let validity = parray
        .validity()
        .vortex_expect("zigzag validity should be derivable");
    match parray.ptype() {
        PType::U8 => zigzag_decode_primitive::<i8>(parray.try_into_buffer_mut(), validity),
        PType::U16 => zigzag_decode_primitive::<i16>(parray.try_into_buffer_mut(), validity),
        PType::U32 => zigzag_decode_primitive::<i32>(parray.try_into_buffer_mut(), validity),
        PType::U64 => zigzag_decode_primitive::<i64>(parray.try_into_buffer_mut(), validity),
        _ => vortex_panic!(
            "ZigZag can only decode unsigned integers, got {}",
            parray.ptype()
        ),
    }
}

fn zigzag_decode_primitive<T: ExternalZigZag + NativePType>(
    values: Result<BufferMut<T::UInt>, Buffer<T::UInt>>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    let decoded = match values {
        Ok(mut values) => {
            values
                .as_mut_slice()
                .map_into_in_place(|value| T::decode(value).as_());

            // SAFETY: `T` and `T::UInt` are same-width integers with identical alignment.
            // Every bit pattern is valid for both types.
            unsafe { values.transmute::<T>() }
        }
        Err(shared) => {
            let len = shared.len();
            let mut out = BufferMut::<T>::with_capacity(len);
            shared
                .as_slice()
                .map_into(&mut out.spare_capacity_mut()[..len], T::decode);

            // SAFETY: `map_into` writes every lane of the `len` elements.
            unsafe { out.set_len(len) };
            out
        }
    };

    PrimitiveArray::new(decoded.freeze(), validity)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_session::VortexSession;

    use super::*;
    use crate::array::ZigZagArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::i8(
        PrimitiveArray::from_iter([i8::MIN, i8::MAX].into_iter().chain(-100..100)),
        PrimitiveArray::from_iter([u8::MAX, u8::MAX - 1]),
    )]
    #[case::i16(
        PrimitiveArray::from_iter([i16::MIN, i16::MAX].into_iter().chain(-100..100)),
        PrimitiveArray::from_iter([u16::MAX, u16::MAX - 1]),
    )]
    #[case::i32(
        PrimitiveArray::from_iter([i32::MIN, i32::MAX].into_iter().chain(-100..100)),
        PrimitiveArray::from_iter([u32::MAX, u32::MAX - 1]),
    )]
    #[case::i64(
        PrimitiveArray::from_iter([i64::MIN, i64::MAX].into_iter().chain(-100..100)),
        PrimitiveArray::from_iter([u64::MAX, u64::MAX - 1]),
    )]
    fn test_compress(
        #[case] input: PrimitiveArray,
        #[case] encoded_prefix: PrimitiveArray,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = zigzag_encode(input.as_view())?;
        assert_arrays_eq!(
            compressed.encoded().slice(0..encoded_prefix.len())?,
            encoded_prefix,
            &mut ctx
        );
        assert_arrays_eq!(compressed.into_array(), input, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::i8(
        PrimitiveArray::from_iter([u8::MAX, 1, 0, 2, u8::MAX - 1]),
        PrimitiveArray::from_iter([i8::MIN, -1, 0, 1, i8::MAX]),
    )]
    #[case::i16(
        PrimitiveArray::from_iter([u16::MAX, 1, 0, 2, u16::MAX - 1]),
        PrimitiveArray::from_iter([i16::MIN, -1, 0, 1, i16::MAX]),
    )]
    #[case::i32(
        PrimitiveArray::from_iter([u32::MAX, 1, 0, 2, u32::MAX - 1]),
        PrimitiveArray::from_iter([i32::MIN, -1, 0, 1, i32::MAX]),
    )]
    #[case::i64(
        PrimitiveArray::from_iter([u64::MAX, 1, 0, 2, u64::MAX - 1]),
        PrimitiveArray::from_iter([i64::MIN, -1, 0, 1, i64::MAX]),
    )]
    #[case::nullable_i32(
        PrimitiveArray::from_option_iter([Some(9u32), None, Some(14), None]),
        PrimitiveArray::from_option_iter([Some(-5i32), None, Some(7), None]),
    )]
    fn test_zigzag_decode(
        #[case] encoded: PrimitiveArray,
        #[case] expected: PrimitiveArray,
        #[values(false, true)] shared: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let kept = shared.then(|| {
            let clone = encoded.clone();
            let before = clone.buffer_handle().to_host_sync().to_vec();
            (clone, before)
        });

        let decoded = zigzag_decode(encoded);
        if let Some((clone, before)) = kept {
            assert_eq!(clone.buffer_handle().to_host_sync().as_slice(), before);
        }
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }
}
