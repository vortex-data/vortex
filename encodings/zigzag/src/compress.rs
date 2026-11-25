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
    let parray = parray.into_owned();
    let validity = parray.validity()?;
    let encoded = match parray.ptype() {
        PType::I8 => zigzag_encode_primitive::<i8>(parray.try_into_buffer_mut(), validity),
        PType::I16 => zigzag_encode_primitive::<i16>(parray.try_into_buffer_mut(), validity),
        PType::I32 => zigzag_encode_primitive::<i32>(parray.try_into_buffer_mut(), validity),
        PType::I64 => zigzag_encode_primitive::<i64>(parray.try_into_buffer_mut(), validity),
        _ => vortex_bail!(
            "ZigZag can only encode signed integers, got {}",
            parray.ptype()
        ),
    };
    ZigZag::try_new(encoded.into_array())
}

fn zigzag_encode_primitive<T: ExternalZigZag + NativePType>(
    values: Result<BufferMut<T>, Buffer<T>>,
    validity: Validity,
) -> PrimitiveArray
where
    <T as ExternalZigZag>::UInt: NativePType,
{
    let encoded = match values {
        Ok(mut values) => {
            values
                .as_mut_slice()
                .map_into_in_place(|value| T::encode(value).as_());

            // SAFETY: `T` and `T::UInt` are same-width integers with identical alignment.
            // Every bit pattern is valid for both types.
            unsafe { values.transmute::<T::UInt>() }
        }
        Err(shared) => {
            let len = shared.len();
            let mut out = BufferMut::<T::UInt>::with_capacity(len);
            shared
                .as_slice()
                .map_into(&mut out.spare_capacity_mut()[..len], T::encode);

            // SAFETY: `map_into` writes every lane of the `len` elements.
            unsafe { out.set_len(len) };
            out
        }
    };

    PrimitiveArray::new(encoded.freeze(), validity)
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
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::array::ZigZagArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn assert_encode_owned_and_shared<T>(input: Vec<T>, expected: Vec<T::UInt>) -> VortexResult<()>
    where
        T: ExternalZigZag + NativePType,
        <T as ExternalZigZag>::UInt: NativePType,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let expected_arr = PrimitiveArray::from_iter(expected);

        // Uniquely owned buffer mutates in place without reallocating.
        let owned_buf = BufferMut::from_iter(input.iter().copied());
        let owned_ptr = owned_buf.as_ptr() as usize;
        let encoded_owned = zigzag_encode_primitive::<T>(Ok(owned_buf), Validity::NonNullable);
        assert_eq!(
            encoded_owned.as_slice::<T::UInt>().as_ptr() as usize,
            owned_ptr
        );
        assert_arrays_eq!(encoded_owned, expected_arr, &mut ctx);

        // Shared buffer encodes in a single pass into a new allocation without mutating the source.
        let shared_buf = Buffer::from_iter(input.iter().copied());
        let shared_keep = shared_buf.clone();
        let shared_ptr = shared_keep.as_ptr() as usize;
        let encoded_shared =
            zigzag_encode(PrimitiveArray::new(shared_buf, Validity::NonNullable).as_view())?;
        let encoded_shared_prim = encoded_shared
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_ne!(
            encoded_shared_prim.as_slice::<T::UInt>().as_ptr() as usize,
            shared_ptr
        );
        assert_eq!(shared_keep.as_slice(), input.as_slice());
        assert_arrays_eq!(encoded_shared_prim, expected_arr, &mut ctx);

        Ok(())
    }

    fn assert_decode_owned_and_shared<T>(
        encoded_input: Vec<T::UInt>,
        expected: Vec<T>,
    ) -> VortexResult<()>
    where
        T: ExternalZigZag + NativePType,
        <T as ExternalZigZag>::UInt: NativePType,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let expected_arr = PrimitiveArray::from_iter(expected);

        // Uniquely owned buffer mutates in place without reallocating.
        let owned_buf = Buffer::from_iter(encoded_input.iter().copied());
        let owned_ptr = owned_buf.as_ptr() as usize;
        let decoded_owned = zigzag_decode(PrimitiveArray::new(owned_buf, Validity::NonNullable));
        assert_eq!(decoded_owned.as_slice::<T>().as_ptr() as usize, owned_ptr);
        assert_arrays_eq!(decoded_owned, expected_arr, &mut ctx);

        // Shared buffer decodes in a single pass into a new allocation without mutating the source.
        let shared_buf = Buffer::from_iter(encoded_input.iter().copied());
        let shared_keep = shared_buf.clone();
        let shared_ptr = shared_keep.as_ptr() as usize;
        let decoded_shared = zigzag_decode(PrimitiveArray::new(shared_buf, Validity::NonNullable));
        assert_ne!(decoded_shared.as_slice::<T>().as_ptr() as usize, shared_ptr);
        assert_eq!(shared_keep.as_slice(), encoded_input.as_slice());
        assert_arrays_eq!(decoded_shared, expected_arr, &mut ctx);

        Ok(())
    }

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

    #[test]
    fn test_encode_owned_and_shared_paths() -> VortexResult<()> {
        assert_encode_owned_and_shared::<i8>(
            vec![i8::MIN, -1, 0, 1, i8::MAX],
            vec![u8::MAX, 1, 0, 2, u8::MAX - 1],
        )?;
        assert_encode_owned_and_shared::<i16>(
            vec![i16::MIN, -1, 0, 1, i16::MAX],
            vec![u16::MAX, 1, 0, 2, u16::MAX - 1],
        )?;
        assert_encode_owned_and_shared::<i32>(
            vec![i32::MIN, -1, 0, 1, i32::MAX],
            vec![u32::MAX, 1, 0, 2, u32::MAX - 1],
        )?;
        assert_encode_owned_and_shared::<i64>(
            vec![i64::MIN, -1, 0, 1, i64::MAX],
            vec![u64::MAX, 1, 0, 2, u64::MAX - 1],
        )?;
        Ok(())
    }

    #[test]
    fn test_decode_owned_and_shared_paths() -> VortexResult<()> {
        assert_decode_owned_and_shared::<i8>(
            vec![u8::MAX, 1, 0, 2, u8::MAX - 1],
            vec![i8::MIN, -1, 0, 1, i8::MAX],
        )?;
        assert_decode_owned_and_shared::<i16>(
            vec![u16::MAX, 1, 0, 2, u16::MAX - 1],
            vec![i16::MIN, -1, 0, 1, i16::MAX],
        )?;
        assert_decode_owned_and_shared::<i32>(
            vec![u32::MAX, 1, 0, 2, u32::MAX - 1],
            vec![i32::MIN, -1, 0, 1, i32::MAX],
        )?;
        assert_decode_owned_and_shared::<i64>(
            vec![u64::MAX, 1, 0, 2, u64::MAX - 1],
            vec![i64::MIN, -1, 0, 1, i64::MAX],
        )?;
        Ok(())
    }

    #[test]
    fn test_owned_and_shared_nullable_and_empty() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let validity = Validity::from_iter([true, false, true, false]);
        let shared_signed = buffer![-5i32, 99, 7, -42];
        let shared_signed_keep = shared_signed.clone();
        let encoded_shared =
            zigzag_encode(PrimitiveArray::new(shared_signed, validity.clone()).as_view())?;
        let encoded_owned = zigzag_encode_primitive::<i32>(
            Ok(BufferMut::from_iter([-5i32, 99, 7, -42])),
            validity.clone(),
        );
        assert_eq!(shared_signed_keep.as_slice(), &[-5i32, 99, 7, -42]);
        assert_arrays_eq!(encoded_shared.encoded().clone(), encoded_owned, &mut ctx);

        let shared_unsigned = buffer![9u32, 0, 14, 0];
        let shared_unsigned_keep = shared_unsigned.clone();
        let decoded_shared = zigzag_decode(PrimitiveArray::new(shared_unsigned, validity.clone()));
        let decoded_owned = zigzag_decode(PrimitiveArray::new(buffer![9u32, 0, 14, 0], validity));
        let expected = PrimitiveArray::from_option_iter([Some(-5i32), None, Some(7), None]);
        assert_eq!(shared_unsigned_keep.as_slice(), &[9u32, 0, 14, 0]);
        assert_arrays_eq!(decoded_shared, expected, &mut ctx);
        assert_arrays_eq!(decoded_owned, expected, &mut ctx);

        let empty_signed = Buffer::<i32>::empty();
        let _empty_signed_keep = empty_signed.clone();
        let empty_encoded =
            zigzag_encode(PrimitiveArray::new(empty_signed, Validity::NonNullable).as_view())?;
        assert!(empty_encoded.is_empty());

        let empty_unsigned = Buffer::<u32>::empty();
        let _empty_unsigned_keep = empty_unsigned.clone();
        let empty_decoded =
            zigzag_decode(PrimitiveArray::new(empty_unsigned, Validity::NonNullable));
        assert!(empty_decoded.is_empty());

        Ok(())
    }
}
