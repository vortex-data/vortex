// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{NullArray, NullVTable};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::{ArrayRef, register_kernel};

impl CastKernel for NullVTable {
    fn cast(&self, array: &NullArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Null can only be cast to Null
        match dtype {
            DType::Null => Ok(Some(array.to_array())),
            _ => vortex_bail!("Cannot cast Null to {}", dtype),
        }
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
    fn test_cast_null_to_other_fails() {
        let null_array = NullArray::new(5);
        let result = cast(
            null_array.as_ref(),
            &DType::Primitive(PType::I32, Nullability::Nullable),
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
