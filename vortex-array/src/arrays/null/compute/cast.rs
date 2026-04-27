// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Null;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Null {
    fn cast(array: ArrayView<'_, Null>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.is_nullable() {
            vortex_bail!("Cannot cast Null to {}", dtype);
        }
        if dtype == &DType::Null {
            return Ok(Some(array.array().clone()));
        }

        let scalar = Scalar::null(dtype.clone());
        Ok(Some(ConstantArray::new(scalar, array.len()).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::NullArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[test]
    fn test_cast_null_to_null() {
        let null_array = NullArray::new(5);
        let result = null_array.into_array().cast(DType::Null).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.dtype(), &DType::Null);
    }

    #[test]
    fn test_cast_null_to_nullable_succeeds() {
        let null_array = NullArray::new(5);
        let result = null_array
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))
            .unwrap();

        // Should create a ConstantArray of nulls
        assert_eq!(result.len(), 5);
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // Verify all values are null
        for i in 0..5 {
            assert!(
                result
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
                    .is_null()
            );
        }
    }

    #[test]
    fn test_cast_null_to_non_nullable_fails() {
        let null_array = NullArray::new(5);
        let result = null_array
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[rstest]
    #[case(NullArray::new(5))]
    #[case(NullArray::new(1))]
    #[case(NullArray::new(100))]
    #[case(NullArray::new(0))]
    fn test_cast_null_conformance(#[case] array: NullArray) {
        test_cast_conformance(&array.into_array());
    }
}
