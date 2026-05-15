// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod filter;
mod take;

#[cfg(test)]
mod test {
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::Sparse;

    #[fixture]
    fn array() -> ArrayRef {
        Sparse::try_new(
            buffer![2u64, 9, 15].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            20,
            Scalar::null_native::<i32>(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    fn test_filter(array: ArrayRef) {
        let mut predicate = vec![false, false, true];
        predicate.extend_from_slice(&[false; 17]);
        let mask = Mask::from_iter(predicate);

        let filtered_array = array.filter(mask).unwrap();

        // Construct expected SparseArray: index 2 was kept, which had value 33.
        // The new index is 0 (since it's the only element).
        let expected = Sparse::try_new(
            buffer![0u64].into_array(),
            PrimitiveArray::new(buffer![33_i32], Validity::AllValid).into_array(),
            1,
            Scalar::null_native::<i32>(),
        )
        .unwrap();

        assert_arrays_eq!(filtered_array, expected);
    }

    #[test]
    fn true_fill_value() {
        let mask = Mask::from_iter([false, true, false, true, false, true, true]);
        let array = Sparse::try_new(
            buffer![0_u64, 3, 6].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            7,
            Scalar::null_native::<i32>(),
        )
        .unwrap()
        .into_array();

        let filtered_array = array.filter(mask).unwrap();

        // Original indices 0, 3, 6 with values 33, 44, 55.
        // Mask keeps indices 1, 3, 5, 6 -> new indices 0, 1, 2, 3.
        // Index 3 (value 44) maps to new index 1.
        // Index 6 (value 55) maps to new index 3.
        let expected = Sparse::try_new(
            buffer![1u64, 3].into_array(),
            PrimitiveArray::new(buffer![44_i32, 55], Validity::AllValid).into_array(),
            4,
            Scalar::null_native::<i32>(),
        )
        .unwrap();

        assert_arrays_eq!(filtered_array, expected);
    }

    #[rstest]
    fn test_sparse_binary_numeric(array: ArrayRef) {
        test_binary_numeric_array(array)
    }

    #[test]
    fn test_mask_sparse_array() {
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        test_mask_conformance(
            &Sparse::try_new(
                buffer![1u64, 2, 4].into_array(),
                buffer![100i32, 200, 300]
                    .into_array()
                    .cast(null_fill_value.dtype().clone())
                    .unwrap(),
                5,
                null_fill_value,
            )
            .unwrap()
            .into_array(),
        );

        let ten_fill_value = Scalar::from(10i32);
        test_mask_conformance(
            &Sparse::try_new(
                buffer![1u64, 2, 4].into_array(),
                buffer![100i32, 200, 300].into_array(),
                5,
                ten_fill_value,
            )
            .unwrap()
            .into_array(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::f32;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;

    use crate::Sparse;
    use crate::SparseArray;

    #[rstest]
    // Basic sparse arrays
    #[case::sparse_i32_null_fill(Sparse::try_new(
        buffer![2u64, 5, 8].into_array(),
        PrimitiveArray::from_option_iter([Some(100i32), Some(200), Some(300)]).into_array(),
        10,
        Scalar::null_native::<i32>()
    ).unwrap())]
    #[case::sparse_i32_value_fill(Sparse::try_new(
        buffer![1u64, 3, 7].into_array(),
        buffer![42i32, 84, 126].into_array(),
        10,
        Scalar::from(0i32)
    ).unwrap())]
    // Different types
    #[case::sparse_u64(Sparse::try_new(
        buffer![0u64, 4, 9].into_array(),
        buffer![1000u64, 2000, 3000].into_array(),
        10,
        Scalar::from(999u64)
    ).unwrap())]
    #[case::sparse_f32(Sparse::try_new(
        buffer![2u64, 6].into_array(),
        buffer![f32::consts::PI, f32::consts::E].into_array(),
        8,
        Scalar::from(0.0f32)
    ).unwrap())]
    // Edge cases
    #[case::sparse_single_patch(Sparse::try_new(
        buffer![5u64].into_array(),
        buffer![42i32].into_array(),
        10,
        Scalar::from(-1i32)
    ).unwrap())]
    #[case::sparse_dense_patches(Sparse::try_new(
        buffer![0u64, 1, 2, 3, 4].into_array(),
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30), Some(40), Some(50)]).into_array(),
        5,
        Scalar::null_native::<i32>()
    ).unwrap())]
    // Large sparse arrays
    #[case::sparse_large(Sparse::try_new(
        buffer![100u64, 500, 900, 1500, 1999].into_array(),
        buffer![111i32, 222, 333, 444, 555].into_array(),
        2000,
        Scalar::from(0i32)
    ).unwrap())]
    // Nullable patches
    #[case::sparse_nullable_patches({
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        Sparse::try_new(
            buffer![1u64, 4, 7].into_array(),
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)])
                .into_array()
                .cast(null_fill_value.dtype().clone())
                .unwrap(),
            10,
            null_fill_value
        ).unwrap()
    })]

    fn test_sparse_consistency(#[case] array: SparseArray) {
        test_array_consistency(&array.into_array());
    }

    #[rstest]
    #[case::sparse_i32_basic(Sparse::try_new(
        buffer![2u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        10,
        Scalar::from(0i32)
    ).unwrap())]
    #[case::sparse_u32_basic(Sparse::try_new(
        buffer![1u64, 3, 7].into_array(),
        buffer![1000u32, 2000, 3000].into_array(),
        10,
        Scalar::from(100u32)
    ).unwrap())]
    #[case::sparse_i64_basic(Sparse::try_new(
        buffer![0u64, 4, 9].into_array(),
        buffer![5000i64, 6000, 7000].into_array(),
        10,
        Scalar::from(1000i64)
    ).unwrap())]
    #[case::sparse_f32_basic(Sparse::try_new(
        buffer![2u64, 6].into_array(),
        buffer![1.5f32, 2.5].into_array(),
        8,
        Scalar::from(0.5f32)
    ).unwrap())]
    #[case::sparse_f64_basic(Sparse::try_new(
        buffer![1u64, 5, 9].into_array(),
        buffer![10.1f64, 20.2, 30.3].into_array(),
        10,
        Scalar::from(5.0f64)
    ).unwrap())]
    #[case::sparse_i32_large(Sparse::try_new(
        buffer![10u64, 50, 90, 150, 199].into_array(),
        buffer![111i32, 222, 333, 444, 555].into_array(),
        200,
        Scalar::from(0i32)
    ).unwrap())]
    fn test_sparse_binary_numeric(#[case] array: SparseArray) {
        test_binary_numeric_array(array.into_array());
    }
}
