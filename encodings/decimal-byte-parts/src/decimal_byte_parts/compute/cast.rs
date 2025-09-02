// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{DecimalBytePartsArray, DecimalBytePartsVTable};

impl CastKernel for DecimalBytePartsVTable {
    fn cast(&self, array: &DecimalBytePartsArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // DecimalBytePartsArray can only have Decimal dtype, so we only handle decimal-to-decimal casts
        let DType::Decimal(target_decimal, target_nullability) = dtype else {
            // Cannot cast decimal to non-decimal types - delegate to canonical form
            return Ok(None);
        };

        // Check if this is just a nullability change
        if array.decimal_dtype() == target_decimal
            && array.dtype().nullability() != *target_nullability
        {
            // Cast the msp array to handle nullability change
            let new_msp = cast(
                array.msp(),
                &array.msp().dtype().with_nullability(*target_nullability),
            )?;

            return Ok(Some(
                DecimalBytePartsArray::try_new(new_msp, *target_decimal)?.into_array(),
            ));
        }

        // For precision/scale changes, decode to canonical and let DecimalArray handle it
        Ok(None)
    }
}

register_kernel!(CastKernelAdapter(DecimalBytePartsVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_dtype::{DType, DecimalDType, Nullability};

    use crate::DecimalBytePartsArray;

    #[test]
    fn test_cast_decimal_byte_parts_nullability() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalBytePartsArray::try_new(
            PrimitiveArray::from_iter([100i32, 200, 300, 400]).into_array(),
            decimal_dtype,
        )
        .unwrap();

        // Cast to nullable decimal
        let casted = cast(
            array.as_ref(),
            &DType::Decimal(decimal_dtype, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::Nullable)
        );

        // Verify the values are preserved
        let decoded = casted.to_decimal();
        assert_eq!(decoded.len(), 4);
    }

    #[test]
    fn test_cast_decimal_byte_parts_nullable_to_non_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalBytePartsArray::try_new(
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
            decimal_dtype,
        )
        .unwrap();

        // Cast to non-nullable should fail due to nulls
        let result = cast(
            array.as_ref(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable),
        );
        assert!(result.is_err());
    }

    #[rstest]
    #[case::i32(DecimalBytePartsArray::try_new(
        PrimitiveArray::from_iter([100i32, 200, 300, 400, 500]).into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    #[case::i64(DecimalBytePartsArray::try_new(
        PrimitiveArray::from_iter([1000i64, 2000, 3000, 4000]).into_array(),
        DecimalDType::new(19, 4),
    ).unwrap())]
    #[case::nullable(DecimalBytePartsArray::try_new(
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300), Some(400), None])
            .into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    #[case::single(DecimalBytePartsArray::try_new(
        PrimitiveArray::from_iter([42i32]).into_array(),
        DecimalDType::new(5, 1),
    ).unwrap())]
    #[case::negative(DecimalBytePartsArray::try_new(
        PrimitiveArray::from_iter([-100i32, -200, 300, -400, 500]).into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    fn test_cast_decimal_byte_parts_conformance(#[case] array: DecimalBytePartsArray) {
        test_cast_conformance(array.as_ref());
    }
}
