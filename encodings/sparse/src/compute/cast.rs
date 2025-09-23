// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{SparseArray, SparseVTable};

impl CastKernel for SparseVTable {
    fn cast(&self, array: &SparseArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Cast both the patches values and the fill value
        let casted_fill = array.fill_scalar().cast(dtype)?;
        let casted_patches = array
            .patches()
            .clone()
            .map_values(|values| cast(&values, dtype))?;

        Ok(Some(
            SparseArray::try_new_from_patches(casted_patches, casted_fill)?.into_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(SparseVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[test]
    fn test_cast_sparse_i32_to_i64() {
        let sparse = SparseArray::try_new(
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            10,
            Scalar::from(0i32),
        )
        .unwrap();

        let casted = cast(
            sparse.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        let decoded = casted.to_primitive();
        let values = decoded.as_slice::<i64>();
        assert_eq!(values[2], 100);
        assert_eq!(values[5], 200);
        assert_eq!(values[8], 300);
        assert_eq!(values[0], 0); // fill value
    }

    #[test]
    fn test_cast_sparse_with_null_fill() {
        let sparse = SparseArray::try_new(
            buffer![1u64, 3, 5].into_array(),
            PrimitiveArray::from_option_iter([Some(42i32), Some(84), Some(126)]).into_array(),
            8,
            Scalar::null_typed::<i32>(),
        )
        .unwrap();

        let casted = cast(
            sparse.as_ref(),
            &DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(SparseArray::try_new(
        buffer![2u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        10,
        Scalar::from(0i32)
    ).unwrap())]
    #[case(SparseArray::try_new(
        buffer![0u64, 4, 9].into_array(),
        buffer![1.5f32, 2.5, 3.5].into_array(),
        10,
        Scalar::from(0.0f32)
    ).unwrap())]
    #[case(SparseArray::try_new(
        buffer![1u64, 3, 7].into_array(),
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
        10,
        Scalar::null_typed::<i32>()
    ).unwrap())]
    #[case(SparseArray::try_new(
        buffer![5u64].into_array(),
        buffer![42u8].into_array(),
        10,
        Scalar::from(0u8)
    ).unwrap())]
    fn test_cast_sparse_conformance(#[case] array: SparseArray) {
        test_cast_conformance(array.as_ref());
    }
}
