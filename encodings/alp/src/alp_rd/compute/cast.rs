// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::ALPRDArrayExt;
use crate::alp_rd::ALPRD;

impl CastReduce for ALPRD {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if this is just a nullability change
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        // For nullability-only changes, we need to cast the left_parts array
        // since it carries the validity information
        let new_left_parts = array.left_parts().cast(
            array
                .left_parts()
                .dtype()
                .with_nullability(dtype.nullability()),
        )?;

        // NOTE: `CastReduce::cast` has a fixed trait signature without `ExecutionCtx`, so we
        // construct a legacy ctx locally at this trait boundary.
        Ok(Some(
            unsafe {
                ALPRD::new_unchecked(
                    dtype.clone(),
                    new_left_parts,
                    array.left_parts_dictionary().clone(),
                    array.right_parts().clone(),
                    array.right_bit_width(),
                    array.left_parts_patches(),
                )
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;

    use crate::RDEncoder;

    #[test]
    fn test_cast_alprd_f32_to_f64() {
        let mut ctx = array_session().create_execution_ctx();
        let values = vec![1.0f32, 1.1, 1.2, 1.3, 1.4];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        let alprd = encoder.encode(arr.as_view(), &mut ctx);

        let casted = alprd
            .into_array()
            .cast(DType::Primitive(PType::F64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F64, Nullability::NonNullable)
        );

        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        let f64_values = decoded.as_slice::<f64>();
        assert_eq!(f64_values.len(), 5);
        assert!((f64_values[0] - 1.0).abs() < f64::EPSILON);
        assert!((f64_values[1] - 1.1).abs() < 1e-6); // Use larger epsilon for f32->f64 conversion
    }

    #[test]
    fn test_cast_alprd_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let arr =
            PrimitiveArray::from_option_iter([Some(10.0f64), None, Some(10.1), Some(10.2), None]);
        let values = vec![10.0f64, 10.1, 10.2];
        let encoder = RDEncoder::new(&values);
        let alprd = encoder.encode(arr.as_view(), &mut ctx);

        // Cast to NonNullable should fail since we have nulls. The failure surfaces during
        // execution since the reduce path defers when the validity stat is not cached.
        let result = alprd
            .clone()
            .into_array()
            .cast(DType::Primitive(PType::F64, Nullability::NonNullable))
            .and_then(|a| {
                a.execute::<PrimitiveArray>(&mut ctx)
                    .map(|p| p.into_array())
            });
        assert!(result.is_err(), "Expected error, got: {result:?}");

        // Cast to same type with Nullable should succeed
        let casted = alprd
            .into_array()
            .cast(DType::Primitive(PType::F64, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F64, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case::f32({
        let values = vec![1.23f32, 4.56, 7.89, 10.11, 12.13];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view(), &mut array_session().create_execution_ctx())
    })]
    #[case::f64({
        let values = vec![100.1f64, 200.2, 300.3, 400.4, 500.5];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view(), &mut array_session().create_execution_ctx())
    })]
    #[case::single({
        let values = vec![42.42f64];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view(), &mut array_session().create_execution_ctx())
    })]
    #[case::negative({
        let values = vec![0.0f32, -1.5, 2.5, -3.5, 4.5];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view(), &mut array_session().create_execution_ctx())
    })]
    #[case::nullable({
        let arr = PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]);
        let values = vec![1.1f32, 2.2, 3.3];
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view(), &mut array_session().create_execution_ctx())
    })]
    fn test_cast_alprd_conformance(#[case] alprd: crate::alp_rd::ALPRDArray) {
        test_cast_conformance(&alprd.into_array());
    }
}
