// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod predicate;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::RLE;
    use crate::RLEData;
    use crate::test::SESSION;

    fn encode_rle(primitive: &PrimitiveArray) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        Ok(RLEData::encode(primitive.as_view(), &mut ctx)?.into_array())
    }

    fn rle_sliced(primitive: &PrimitiveArray, start: usize, end: usize) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = encode_rle(primitive)?;
        let sliced = encoded.slice(start..end)?.execute::<ArrayRef>(&mut ctx)?;
        assert!(sliced.is::<RLE>());
        Ok(sliced)
    }

    fn rle_u8() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_iter([1u8, 1, 2, 2, 2, 3, 3]))
    }

    fn rle_i32() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_iter([-10i32, -10, 0, 5, 5, 5, 20]))
    }

    fn rle_u64() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_iter([
            100u64, 100, 200, 200, 300, 300,
        ]))
    }

    fn rle_f32() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_iter([
            1.5f32, 1.5, 2.5, 2.5, 3.5, 3.5,
        ]))
    }

    fn rle_nullable_i32() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_option_iter([
            Some(10i32),
            Some(10),
            None,
            Some(20),
            Some(20),
            None,
            Some(30),
        ]))
    }

    fn rle_single() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_iter([42u32]))
    }

    fn rle_multi_chunk() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_iter(
            (0..2500u32).map(|i| (i / 17) * 3),
        ))
    }

    fn rle_multi_chunk_nullable() -> VortexResult<ArrayRef> {
        encode_rle(&PrimitiveArray::from_option_iter(
            (0..2500i32).map(|i| (i % 13 != 0).then_some(i / 19 - 50)),
        ))
    }

    fn rle_multi_chunk_sliced() -> VortexResult<ArrayRef> {
        let primitive = PrimitiveArray::new(
            (0..2500u32).map(|i| i / 11).collect::<Buffer<_>>(),
            Validity::from_iter((0..2500).map(|i| i % 9 != 0)),
        );
        rle_sliced(&primitive, 700, 2200)
    }

    #[rstest]
    #[case::u8(rle_u8()?)]
    #[case::i32(rle_i32()?)]
    #[case::u64(rle_u64()?)]
    #[case::f32(rle_f32()?)]
    #[case::nullable_i32(rle_nullable_i32()?)]
    #[case::single(rle_single()?)]
    #[case::multi_chunk(rle_multi_chunk()?)]
    #[case::multi_chunk_nullable(rle_multi_chunk_nullable()?)]
    #[case::multi_chunk_sliced(rle_multi_chunk_sliced()?)]
    fn test_rle_conformance(#[case] array: ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        test_array_consistency(&array, &mut ctx);
        test_binary_numeric_array(&array, &mut ctx);
        test_filter_conformance(&array, &mut ctx);
        test_mask_conformance(&array, &mut ctx);
        test_take_conformance(&array, &mut ctx);
        Ok(())
    }
}
