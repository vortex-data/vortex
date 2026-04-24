// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
impl CastReduce for DecimalByteParts {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // DecimalBytePartsArray can only have Decimal dtype, so we only handle decimal-to-decimal casts
        let DType::Decimal(target_decimal, target_nullability) = dtype else {
            // Cannot cast decimal to non-decimal types - delegate to canonical form
            return Ok(None);
        };

        // Check if this is just a nullability change
        if array
            .dtype()
            .as_decimal_opt()
            .vortex_expect("must be a decimal dtype")
            == target_decimal
            && array.dtype().nullability() != *target_nullability
        {
            // Cast the msp array to handle nullability change
            let new_msp = array
                .msp()
                .cast(array.msp().dtype().with_nullability(*target_nullability))?;

            return Ok(Some(
                DecimalByteParts::try_new(new_msp, *target_decimal)?.into_array(),
            ));
        }

        // For precision/scale changes, decode to canonical and let DecimalArray handle it
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_buffer::buffer;

    use crate::DecimalByteParts;
    use crate::DecimalBytePartsArray;

    #[test]
    fn test_cast_decimal_byte_parts_nullability() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let array =
            DecimalByteParts::try_new(buffer![100i32, 200, 300, 400].into_array(), decimal_dtype)
                .unwrap();

        // Cast to nullable decimal
        let casted = array
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::Nullable)
        );

        // Verify the values are preserved
        let decoded = casted.execute::<DecimalArray>(&mut ctx).unwrap();
        assert_eq!(decoded.len(), 4);
    }

    #[test]
    fn test_cast_decimal_byte_parts_nullable_to_non_nullable() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalByteParts::try_new(
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
            decimal_dtype,
        )
        .unwrap();

        // Cast to non-nullable should fail due to nulls - force evaluation via execute::<Canonical>
        let result = array
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));
        assert!(result.is_err());
    }

    #[rstest]
    #[case::i32(DecimalByteParts::try_new(
        buffer![100i32, 200, 300, 400, 500].into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    #[case::i64(DecimalByteParts::try_new(
        buffer![1000i64, 2000, 3000, 4000].into_array(),
        DecimalDType::new(19, 4),
    ).unwrap())]
    #[case::nullable(DecimalByteParts::try_new(
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300), Some(400), None])
            .into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    #[case::single(DecimalByteParts::try_new(
        buffer![42i32].into_array(),
        DecimalDType::new(5, 1),
    ).unwrap())]
    #[case::negative(DecimalByteParts::try_new(
        buffer![-100i32, -200, 300, -400, 500].into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    fn test_cast_decimal_byte_parts_conformance(#[case] array: DecimalBytePartsArray) {
        test_cast_conformance(&array.into_array());
    }
}
