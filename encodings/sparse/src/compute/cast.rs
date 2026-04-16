// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::Sparse;
impl CastReduce for Sparse {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let casted_patches = array
            .patches()
            .clone()
            .map_values(|values| values.cast(dtype.clone()))?;

        let casted_fill = if array.patches().num_patches() == array.len() {
            // When every position is patched the fill scalar is unused and can be undefined.
            // We skip casting it entirely and substitute a default value for the target dtype.
            Scalar::default_value(dtype)
        } else {
            array.fill_scalar().cast(dtype)?
        };

        Ok(Some(
            Sparse::try_new_from_patches(casted_patches, casted_fill)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;

    use crate::Sparse;
    use crate::SparseArray;

    #[test]
    fn test_cast_sparse_i32_to_i64() {
        let sparse = Sparse::try_new(
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            10,
            Scalar::from(0i32),
        )
        .unwrap();

        let casted = sparse
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        let expected = PrimitiveArray::from_iter([0i64, 0, 100, 0, 0, 200, 0, 0, 300, 0]);
        #[expect(deprecated)]
        let casted_primitive = casted.to_primitive();
        assert_arrays_eq!(casted_primitive, expected);
    }

    #[test]
    fn test_cast_sparse_with_null_fill() {
        let sparse = Sparse::try_new(
            buffer![1u64, 3, 5].into_array(),
            PrimitiveArray::from_option_iter([Some(42i32), Some(84), Some(126)]).into_array(),
            8,
            Scalar::null_native::<i32>(),
        )
        .unwrap();

        let casted = sparse
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(Sparse::try_new(
        buffer![2u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        10,
        Scalar::from(0i32)
    ).unwrap())]
    #[case(Sparse::try_new(
        buffer![0u64, 4, 9].into_array(),
        buffer![1.5f32, 2.5, 3.5].into_array(),
        10,
        Scalar::from(0.0f32)
    ).unwrap())]
    #[case(Sparse::try_new(
        buffer![1u64, 3, 7].into_array(),
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
        10,
        Scalar::null_native::<i32>()
    ).unwrap())]
    #[case(Sparse::try_new(
        buffer![5u64].into_array(),
        buffer![42u8].into_array(),
        10,
        Scalar::from(0u8)
    ).unwrap())]
    fn test_cast_sparse_conformance(#[case] array: SparseArray) {
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_sparse_null_fill_all_patched_to_non_nullable() -> vortex_error::VortexResult<()> {
        // Regression test for https://github.com/vortex-data/vortex/issues/6932
        //
        // When all positions are patched the null fill is unused, so a cast to
        // non-nullable is valid.  Sparse::cast detects this case, substitutes a
        // zero fill, and keeps the result in the Sparse encoding.
        let sparse = Sparse::try_new(
            buffer![0u64, 1, 2, 3, 4].into_array(),
            buffer![10u64, 20, 30, 40, 50].into_array(),
            5,
            Scalar::null_native::<u64>(),
        )?;

        let casted = sparse
            .into_array()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?;

        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable)
        );

        let expected = PrimitiveArray::from_iter([10u64, 20, 30, 40, 50]);
        #[expect(deprecated)]
        let casted_primitive = casted.to_primitive();
        assert_arrays_eq!(casted_primitive, expected);
        Ok(())
    }

    #[test]
    fn test_fill_null_sparse_with_null_fill() -> vortex_error::VortexResult<()> {
        // Regression test for https://github.com/vortex-data/vortex/issues/6932
        // fill_null on a sparse array with null fill triggers an internal cast to
        // non-nullable, which must not panic.
        let sparse = Sparse::try_new(
            buffer![1u64, 3].into_array(),
            buffer![10u64, 20].into_array(),
            5,
            Scalar::null_native::<u64>(),
        )?;

        let filled = sparse.into_array().fill_null(Scalar::from(0u64))?;

        assert_eq!(
            filled.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable)
        );
        Ok(())
    }
}
