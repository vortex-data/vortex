// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{ConstantArray, NullArray, NullVTable};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for NullVTable {
    fn cast(&self, array: &NullArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.is_nullable() {
            vortex_bail!("Cannot cast Null to {}", dtype);
        }
        if dtype == &DType::Null {
            return Ok(Some(array.to_array()));
        }

        let scalar = Scalar::new(dtype.clone(), ScalarValue::null());
        Ok(Some(ConstantArray::new(scalar, array.len()).into_array()))
    }
}

register_kernel!(CastKernelAdapter(NullVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::arrays::NullArray;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;

    #[test]
    fn test_cast_null_to_null() {
        let null_array = NullArray::new(5);
        let result = cast(null_array.as_ref(), &DType::Null).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.dtype(), &DType::Null);
    }

    #[test]
    fn test_cast_null_to_nullable_succeeds() {
        let null_array = NullArray::new(5);
        let result = cast(
            null_array.as_ref(),
            &DType::Primitive(PType::I32, Nullability::Nullable),
        )
        .unwrap();

        // Should create a ConstantArray of nulls
        assert_eq!(result.len(), 5);
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // Verify all values are null
        for i in 0..5 {
            assert!(result.scalar_at(i).is_null());
        }
    }

    #[test]
    fn test_cast_null_to_non_nullable_fails() {
        let null_array = NullArray::new(5);
        let result = cast(
            null_array.as_ref(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(result.is_err());
    }

    #[rstest]
    #[case(NullArray::new(5))]
    #[case(NullArray::new(1))]
    #[case(NullArray::new(100))]
    #[case(NullArray::new(0))]
    fn test_cast_null_conformance(#[case] array: NullArray) {
        test_cast_conformance(array.as_ref());
    }
}
