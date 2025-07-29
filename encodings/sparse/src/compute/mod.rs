// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{FilterKernel, FilterKernelAdapter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{SparseArray, SparseVTable};

mod binary_numeric;
mod invert;
mod take;

impl FilterKernel for SparseVTable {
    fn filter(&self, array: &SparseArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let new_length = mask.true_count();

        let Some(new_patches) = array.patches().filter(mask)? else {
            return Ok(ConstantArray::new(array.fill_scalar().clone(), new_length).into_array());
        };

        Ok(
            SparseArray::try_new_from_patches(new_patches, array.fill_scalar().clone())?
                .into_array(),
        )
    }
}

register_kernel!(FilterKernelAdapter(SparseVTable).lift());

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_conformance;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::{cast, filter};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::{SparseArray, SparseVTable};

    #[fixture]
    fn array() -> ArrayRef {
        SparseArray::try_new(
            buffer![2u64, 9, 15].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            20,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    fn test_filter(array: ArrayRef) {
        let mut predicate = vec![false, false, true];
        predicate.extend_from_slice(&[false; 17]);
        let mask = Mask::from_iter(predicate);

        let filtered_array = filter(&array, &mask).unwrap();
        let filtered_array = filtered_array.as_::<SparseVTable>();

        assert_eq!(filtered_array.len(), 1);
        assert_eq!(filtered_array.patches().values().len(), 1);
        assert_eq!(filtered_array.patches().indices().len(), 1);
    }

    #[test]
    fn true_fill_value() {
        let mask = Mask::from_iter([false, true, false, true, false, true, true]);
        let array = SparseArray::try_new(
            buffer![0_u64, 3, 6].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            7,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array();

        let filtered_array = filter(&array, &mask).unwrap();
        let filtered_array = filtered_array.as_::<SparseVTable>();

        assert_eq!(filtered_array.len(), 4);
        let primitive = filtered_array.patches().indices().to_primitive().unwrap();

        assert_eq!(primitive.as_slice::<u64>(), &[1, 3]);
    }

    #[rstest]
    fn test_sparse_binary_numeric(array: ArrayRef) {
        test_binary_numeric_conformance::<i32>(array)
    }

    #[test]
    fn test_mask_sparse_array() {
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        test_mask_conformance(
            SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                cast(
                    &buffer![100i32, 200, 300].into_array(),
                    null_fill_value.dtype(),
                )
                .unwrap(),
                5,
                null_fill_value,
            )
            .unwrap()
            .as_ref(),
        );

        let ten_fill_value = Scalar::from(10i32);
        test_mask_conformance(
            SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                buffer![100i32, 200, 300].into_array(),
                5,
                ten_fill_value,
            )
            .unwrap()
            .as_ref(),
        )
    }

    #[test]
    fn test_filter_sparse_array() {
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        test_filter_conformance(
            SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                cast(
                    &buffer![100i32, 200, 300].into_array(),
                    null_fill_value.dtype(),
                )
                .unwrap(),
                5,
                null_fill_value,
            )
            .unwrap()
            .as_ref(),
        );

        let ten_fill_value = Scalar::from(10i32);
        test_filter_conformance(
            SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                buffer![100i32, 200, 300].into_array(),
                5,
                ten_fill_value,
            )
            .unwrap()
            .as_ref(),
        )
    }
}

#[cfg(test)]
mod consistency_tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;
    use vortex_array::{IntoArray, compute::cast};
    use vortex_dtype::{DType, Nullability, PType};
    use crate::SparseArray;

    #[rstest]
    // Basic sparse arrays
    #[case::sparse_i32_null_fill(SparseArray::try_new(
        buffer![2u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        10,
        Scalar::null_typed::<i32>()
    ).unwrap())]
    #[case::sparse_i32_value_fill(SparseArray::try_new(
        buffer![1u64, 3, 7].into_array(),
        buffer![42i32, 84, 126].into_array(),
        10,
        Scalar::from(0i32)
    ).unwrap())]
    
    // Different types
    #[case::sparse_u64(SparseArray::try_new(
        buffer![0u64, 4, 9].into_array(),
        buffer![1000u64, 2000, 3000].into_array(),
        10,
        Scalar::from(999u64)
    ).unwrap())]
    #[case::sparse_f32(SparseArray::try_new(
        buffer![2u64, 6].into_array(),
        buffer![3.14159f32, 2.71828].into_array(),
        8,
        Scalar::from(0.0f32)
    ).unwrap())]
    
    // Edge cases
    #[case::sparse_single_patch(SparseArray::try_new(
        buffer![5u64].into_array(),
        buffer![42i32].into_array(),
        10,
        Scalar::from(-1i32)
    ).unwrap())]
    #[case::sparse_dense_patches(SparseArray::try_new(
        buffer![0u64, 1, 2, 3, 4].into_array(),
        buffer![10i32, 20, 30, 40, 50].into_array(),
        5,
        Scalar::null_typed::<i32>()
    ).unwrap())]
    
    // Large sparse arrays
    #[case::sparse_large(SparseArray::try_new(
        buffer![100u64, 500, 900, 1500, 1999].into_array(),
        buffer![111i32, 222, 333, 444, 555].into_array(),
        2000,
        Scalar::from(0i32)
    ).unwrap())]
    
    // Nullable patches
    #[case::sparse_nullable_patches({
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        SparseArray::try_new(
            buffer![1u64, 4, 7].into_array(),
            cast(
                &PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
                null_fill_value.dtype()
            ).unwrap(),
            10,
            null_fill_value
        ).unwrap()
    })]
    
    fn test_sparse_consistency(#[case] array: SparseArray) {
        test_array_consistency(array.as_ref());
    }
}
