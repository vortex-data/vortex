// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use crate::ConstantArray;
use crate::Sparse;
use crate::SparseData;
impl TakeExecute for Sparse {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let patches_take = if array.fill_scalar().is_null() {
            array.patches().take(indices, ctx)?
        } else {
            array.patches().take_with_nulls(indices, ctx)?
        };

        let Some(new_patches) = patches_take else {
            let result_fill_scalar = array.fill_scalar().cast(
                &array
                    .dtype()
                    .union_nullability(indices.dtype().nullability()),
            )?;
            return Ok(Some(
                ConstantArray::new(result_fill_scalar, indices.len()).into_array(),
            ));
        };

        // See `SparseEncoding::slice`.
        if new_patches.array_len() == new_patches.values().len() {
            return Ok(Some(new_patches.into_values()));
        }

        Ok(Some(
            SparseData::try_new_from_patches(
                new_patches,
                array.fill_scalar().cast(
                    &array
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                )?,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::Sparse;
    use crate::SparseArray;

    fn test_array_fill_value() -> Scalar {
        // making this const is annoying
        Scalar::null_native::<f64>()
    }

    fn sparse_array() -> ArrayRef {
        Sparse::try_new(
            buffer![0u64, 37, 47, 99].into_array(),
            PrimitiveArray::new(buffer![1.23f64, 0.47, 9.99, 3.5], Validity::AllValid).into_array(),
            100,
            test_array_fill_value(),
        )
        .unwrap()
        .into_array()
    }

    #[test]
    fn take_with_non_zero_offset() {
        let sparse = sparse_array();
        let sparse = sparse.slice(30..40).unwrap();
        let taken = sparse.take(buffer![6, 7, 8].into_array()).unwrap();
        let expected = PrimitiveArray::from_option_iter([Option::<f64>::None, Some(0.47), None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn sparse_take() {
        let sparse = sparse_array();
        let taken = sparse.take(buffer![0, 47, 47, 0, 99].into_array()).unwrap();
        let expected = PrimitiveArray::from_option_iter([
            Some(1.23f64),
            Some(9.99),
            Some(9.99),
            Some(1.23),
            Some(3.5),
        ]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn nonexistent_take() {
        let sparse = sparse_array();
        let taken = sparse.take(buffer![69].into_array()).unwrap();
        let expected = ConstantArray::new(test_array_fill_value(), 1).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    fn ordered_take() {
        let sparse = sparse_array();
        // Note: take returns a canonical array, not SparseArray
        let taken = sparse.take(buffer![69, 37].into_array()).unwrap();
        // Index 69 is not in sparse array (fill value is null), index 37 has value 0.47
        let expected = PrimitiveArray::from_option_iter([Option::<f64>::None, Some(0.47f64)]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn nullable_take() {
        let arr = Sparse::try_new(
            buffer![1u32].into_array(),
            buffer![10].into_array(),
            10,
            Scalar::primitive(1, Nullability::NonNullable),
        )
        .unwrap();

        let taken = arr
            .take(
                PrimitiveArray::from_option_iter([Some(2u32), Some(1u32), Option::<u32>::None])
                    .into_array(),
            )
            .unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(1), Some(10), Option::<i32>::None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn nullable_take_with_many_patches() {
        let arr = Sparse::try_new(
            buffer![1u32, 3, 7, 8, 9].into_array(),
            buffer![10, 8, 3, 2, 1].into_array(),
            10,
            Scalar::primitive(1, Nullability::NonNullable),
        )
        .unwrap();

        let taken = arr
            .take(
                PrimitiveArray::from_option_iter([Some(2u32), Some(1u32), Option::<u32>::None])
                    .into_array(),
            )
            .unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(1), Some(10), Option::<i32>::None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[rstest]
    #[case(Sparse::try_new(
        buffer![0u64, 37, 47, 99].into_array(),
        PrimitiveArray::new(buffer![1.23f64, 0.47, 9.99, 3.5], Validity::AllValid).into_array(),
        100,
        Scalar::null_native::<f64>(),
    ).unwrap())]
    #[case(Sparse::try_new(
        buffer![1u32, 3, 7, 8, 9].into_array(),
        buffer![10, 8, 3, 2, 1].into_array(),
        10,
        Scalar::from(0i32),
    ).unwrap())]
    #[case({
        let nullable_values = PrimitiveArray::from_option_iter([Some(100i64), None, Some(300)]);
        Sparse::try_new(
            buffer![2u64, 4, 6].into_array(),
            nullable_values.into_array(),
            10,
            Scalar::null_native::<i64>(),
        ).unwrap()
    })]
    #[case(Sparse::try_new(
        buffer![5u64].into_array(),
        buffer![999i32].into_array(),
        20,
        Scalar::from(-1i32),
    ).unwrap())]
    fn test_take_sparse_conformance(#[case] sparse: SparseArray) {
        use vortex_array::compute::conformance::take::test_take_conformance;
        test_take_conformance(&sparse.into_array());
    }
}
