// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::buffer;

    use crate::Zstd;
    use crate::ZstdArray;

    fn zstd_i32() -> ZstdArray {
        let values = PrimitiveArray::from_iter([100i32, 200, 300, 400, 500]);
        Zstd::from_primitive(&values, 0, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn zstd_f64() -> ZstdArray {
        let values = PrimitiveArray::from_iter([1.1f64, 2.2, 3.3, 4.4, 5.5]);
        Zstd::from_primitive(&values, 0, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn zstd_u32() -> ZstdArray {
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40, 50]);
        Zstd::from_primitive(&values, 0, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn zstd_nullable_i64() -> ZstdArray {
        let values =
            PrimitiveArray::from_option_iter([Some(1000i64), None, Some(3000), Some(4000), None]);
        Zstd::from_primitive(&values, 0, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn zstd_single() -> ZstdArray {
        let values = PrimitiveArray::new(
            buffer![42i64],
            vortex_array::validity::Validity::NonNullable,
        );
        Zstd::from_primitive(&values, 0, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn zstd_large() -> ZstdArray {
        let values = PrimitiveArray::new(
            buffer![0u32..1000],
            vortex_array::validity::Validity::NonNullable,
        );
        Zstd::from_primitive(&values, 3, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn zstd_all_same() -> ZstdArray {
        let values = PrimitiveArray::new(
            buffer![42i32; 100],
            vortex_array::validity::Validity::NonNullable,
        );
        Zstd::from_primitive(&values, 0, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn zstd_negative() -> ZstdArray {
        let values = PrimitiveArray::from_iter([-100i32, -50, 0, 50, 100]);
        Zstd::from_primitive(&values, 0, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    #[rstest]
    #[case::i32(zstd_i32())]
    #[case::f64(zstd_f64())]
    #[case::u32(zstd_u32())]
    #[case::nullable_i64(zstd_nullable_i64())]
    #[case::single(zstd_single())]
    #[case::large(zstd_large())]
    #[case::all_same(zstd_all_same())]
    #[case::negative(zstd_negative())]
    fn test_zstd_consistency(#[case] array: ZstdArray) {
        test_array_consistency(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
