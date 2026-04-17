// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::patches::Patches;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;
use crate::alp::ALP;

impl CastReduce for ALP {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if this is just a nullability change
        if array.dtype().eq_ignore_nullability(dtype) {
            // For nullability-only changes, we can avoid decoding
            // Cast the encoded array (integers) to handle nullability
            let new_encoded = array.encoded().cast(
                array
                    .encoded()
                    .dtype()
                    .with_nullability(dtype.nullability()),
            )?;

            let new_patches = array
                .patches()
                .map(|p| {
                    if p.values().dtype() == dtype {
                        Ok(p)
                    } else {
                        Patches::new(
                            p.array_len(),
                            p.offset(),
                            p.indices().clone(),
                            p.values().cast(dtype.clone())?,
                            p.chunk_offsets().clone(),
                        )
                    }
                })
                .transpose()?;

            // SAFETY: casting nullability doesn't alter the invariants
            unsafe {
                Ok(Some(
                    ALP::new_unchecked(new_encoded, array.exponents(), new_patches).into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::alp::array::ALPArrayExt;
    use crate::alp_encode;

    #[test]
    fn issue_5766_test_cast_alp_with_patches_to_nullable() -> VortexResult<()> {
        let values = buffer![1.234f32, f32::NAN, 2.345, f32::INFINITY, 3.456].into_array();
        #[expect(deprecated)]
        let values_primitive = values.to_primitive();
        let alp = alp_encode(
            values_primitive.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        assert!(
            alp.patches().is_some(),
            "Test requires ALP array with patches"
        );

        let nullable_dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let casted = alp.into_array().cast(nullable_dtype.clone())?;

        let expected = values.cast(nullable_dtype)?;

        #[expect(deprecated)]
        let casted_prim = casted.to_canonical()?.into_primitive();
        assert_arrays_eq!(casted_prim, expected);

        Ok(())
    }

    #[test]
    fn test_cast_alp_f32_to_f64() -> VortexResult<()> {
        let values = buffer![1.5f32, 2.5, 3.5, 4.5].into_array();
        #[expect(deprecated)]
        let values_primitive = values.to_primitive();
        let alp = alp_encode(
            values_primitive.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let casted = alp
            .into_array()
            .cast(DType::Primitive(PType::F64, Nullability::NonNullable))?;
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F64, Nullability::NonNullable)
        );

        #[expect(deprecated)]
        let decoded = casted.to_canonical()?.into_primitive();
        let values = decoded.as_slice::<f64>();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.5).abs() < f64::EPSILON);
        assert!((values[1] - 2.5).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_cast_alp_to_int() -> VortexResult<()> {
        let values = buffer![1.0f32, 2.0, 3.0, 4.0].into_array();
        #[expect(deprecated)]
        let values_primitive = values.to_primitive();
        let alp = alp_encode(
            values_primitive.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let casted = alp
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::NonNullable))?;
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        #[expect(deprecated)]
        let decoded = casted.to_canonical()?.into_primitive();
        assert_arrays_eq!(decoded, PrimitiveArray::from_iter([1i32, 2, 3, 4]));

        Ok(())
    }

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    #[case(buffer![0.0f32, -1.5, 2.5, -3.5, 4.5].into_array())]
    fn test_cast_alp_conformance(#[case] array: vortex_array::ArrayRef) -> VortexResult<()> {
        #[expect(deprecated)]
        let array_primitive = array.to_primitive();
        let alp = alp_encode(
            array_primitive.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .vortex_expect("cannot fail");
        test_cast_conformance(&alp.into_array());

        Ok(())
    }
}
