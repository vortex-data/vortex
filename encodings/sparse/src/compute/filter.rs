// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ConstantArray;
use crate::Sparse;
use crate::SparseExt as _;
impl FilterKernel for Sparse {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_length = mask.true_count();

        let Some(new_patches) = array.patches().filter(mask, ctx)? else {
            return Ok(Some(
                ConstantArray::new(array.fill_scalar().clone(), new_length).into_array(),
            ));
        };

        Ok(Some(
            Sparse::try_new_from_patches(new_patches, array.fill_scalar().clone())?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
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

    #[test]
    fn test_filter_sparse_array() {
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        test_filter_conformance(
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
        test_filter_conformance(
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
