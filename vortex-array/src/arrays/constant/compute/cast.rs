// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Constant {
    fn cast(array: ArrayView<'_, Constant>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        match array.scalar().cast(dtype) {
            Ok(scalar) => Ok(Some(ConstantArray::new(scalar, array.len()).into_array())),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;

    #[rstest]
    #[case(ConstantArray::new(Scalar::from(42u32), 5).into_array())]
    #[case(ConstantArray::new(Scalar::from(-100i32), 10).into_array())]
    #[case(ConstantArray::new(Scalar::from(3.5f32), 3).into_array())]
    #[case(ConstantArray::new(Scalar::from(true), 7).into_array())]
    #[case(ConstantArray::new(Scalar::null_native::<i32>(), 4).into_array())]
    #[case(ConstantArray::new(Scalar::from(255u8), 1).into_array())]
    fn test_cast_constant_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(&array);
    }

    #[test]
    fn test_cast_constant_i64_to_decimal() {
        let target_dtype = DType::Decimal(DecimalDType::new(21, 2), Nullability::NonNullable);
        let casted = ConstantArray::new(Scalar::from(42i64), 5)
            .into_array()
            .cast(target_dtype.clone())
            .unwrap();

        assert_eq!(casted.dtype(), &target_dtype);
        let scalar = casted
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(
            scalar.as_decimal().decimal_value(),
            Some(DecimalValue::I128(4200))
        );
    }
}
