// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::FilterKernel;
use vortex_array::compute::FilterKernelAdapter;
use vortex_array::register_kernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::SparseArray;
use crate::SparseVTable;

mod binary_numeric;
mod cast;
mod filter;
mod invert;
mod take;

use filter::SparseFilterKernel;
use vortex_array::kernel::ParentKernelSet;

pub(crate) const PARENT_KERNELS: ParentKernelSet<SparseVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SparseFilterKernel)]);

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
    use std::sync::LazyLock;

    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FilterArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::filter;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;
    use vortex_session::VortexSession;

    use crate::SparseArray;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
    fn test_filter(array: ArrayRef) -> VortexResult<()> {
        let mut predicate = vec![false, false, true];
        predicate.extend_from_slice(&[false; 17]);
        let mask = Mask::from_iter(predicate);
        let mut ctx = SESSION.create_execution_ctx();

        let filtered_array = filter(&array, &mask)?;
        let filtered_array = filtered_array.execute::<Canonical>(&mut ctx)?;

        assert_eq!(filtered_array.len(), 1);
        assert_arrays_eq!(
            filtered_array.into_primitive(),
            PrimitiveArray::from_option_iter([Some(33_i32)])
        );

        Ok(())
    }

    #[test]
    fn true_fill_value() -> VortexResult<()> {
        let mask = Mask::from_iter([false, true, false, true, false, true, true]);
        let array = SparseArray::try_new(
            buffer![0_u64, 3, 6].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            7,
            Scalar::null_typed::<i32>(),
        )?
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let filtered_array = filter(&array, &mask)?.optimize()?;
        let filtered_array = filtered_array.execute::<Canonical>(&mut ctx)?;

        assert_eq!(filtered_array.len(), 4);
        assert_arrays_eq!(
            filtered_array.into_primitive(),
            PrimitiveArray::from_option_iter([None, Some(44_i32), None, Some(55)])
        );

        Ok(())
    }

    /// Test that the SparseFilterKernel execute_parent is invoked when executing
    /// a FilterArray wrapping a SparseArray.
    #[test]
    fn test_sparse_filter_execute_parent() -> VortexResult<()> {
        // Create a sparse array: [null, null, 100, null, 200, null]
        let sparse = SparseArray::try_new(
            buffer![2u64, 4].into_array(),
            PrimitiveArray::new(buffer![100i32, 200], Validity::AllValid).into_array(),
            6,
            Scalar::null_typed::<i32>(),
        )?
        .into_array();

        // Create a filter mask that selects indices [1, 2, 4, 5]
        // This should result in: [null, 100, 200, null]
        let mask = Mask::from_iter([false, true, true, false, true, true]);

        // Create a FilterArray directly (bypassing the filter compute function)
        let filter_array = FilterArray::new(sparse, mask).into_array();

        // Execute the filter - this should trigger execute_parent on SparseVTable
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        assert_eq!(result.len(), 4);
        assert_arrays_eq!(
            result.into_primitive(),
            PrimitiveArray::from_option_iter([None, Some(100i32), Some(200), None])
        );

        Ok(())
    }

    /// Test execute_parent with a non-null fill value
    #[test]
    fn test_sparse_filter_execute_parent_with_fill_value() -> VortexResult<()> {
        // Create a sparse array with fill value 42: [42, 42, 100, 42, 200, 42]
        let sparse = SparseArray::try_new(
            buffer![2u64, 4].into_array(),
            buffer![100i32, 200].into_array(),
            6,
            Scalar::from(42i32),
        )?
        .into_array();

        // Filter to select [1, 2, 3] -> [42, 100, 42]
        let mask = Mask::from_iter([false, true, true, true, false, false]);
        let filter_array = FilterArray::new(sparse, mask).into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        assert_eq!(result.len(), 3);
        assert_arrays_eq!(
            result.into_primitive(),
            PrimitiveArray::from_iter([42i32, 100, 42])
        );

        Ok(())
    }

    #[rstest]
    fn test_sparse_binary_numeric(array: ArrayRef) {
        test_binary_numeric_array(array)
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
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[rstest]
    // Basic sparse arrays
    #[case::sparse_i32_null_fill(SparseArray::try_new(
        buffer![2u64, 5, 8].into_array(),
        PrimitiveArray::from_option_iter([Some(100i32), Some(200), Some(300)]).into_array(),
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
        buffer![std::f32::consts::PI, std::f32::consts::E].into_array(),
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
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30), Some(40), Some(50)]).into_array(),
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

    #[rstest]
    #[case::sparse_i32_basic(SparseArray::try_new(
        buffer![2u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        10,
        Scalar::from(0i32)
    ).unwrap())]
    #[case::sparse_u32_basic(SparseArray::try_new(
        buffer![1u64, 3, 7].into_array(),
        buffer![1000u32, 2000, 3000].into_array(),
        10,
        Scalar::from(100u32)
    ).unwrap())]
    #[case::sparse_i64_basic(SparseArray::try_new(
        buffer![0u64, 4, 9].into_array(),
        buffer![5000i64, 6000, 7000].into_array(),
        10,
        Scalar::from(1000i64)
    ).unwrap())]
    #[case::sparse_f32_basic(SparseArray::try_new(
        buffer![2u64, 6].into_array(),
        buffer![1.5f32, 2.5].into_array(),
        8,
        Scalar::from(0.5f32)
    ).unwrap())]
    #[case::sparse_f64_basic(SparseArray::try_new(
        buffer![1u64, 5, 9].into_array(),
        buffer![10.1f64, 20.2, 30.3].into_array(),
        10,
        Scalar::from(5.0f64)
    ).unwrap())]
    #[case::sparse_i32_large(SparseArray::try_new(
        buffer![10u64, 50, 90, 150, 199].into_array(),
        buffer![111i32, 222, 333, 444, 555].into_array(),
        200,
        Scalar::from(0i32)
    ).unwrap())]
    fn test_sparse_binary_numeric(#[case] array: SparseArray) {
        test_binary_numeric_array(array.into_array());
    }
}
