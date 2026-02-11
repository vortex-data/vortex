// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::FillNullKernel;
use crate::compute::FillNullKernelAdapter;
use crate::compute::cast;
use crate::register_kernel;

impl FillNullKernel for ConstantVTable {
    fn fill_null(&self, array: &ConstantArray, fill_value: &Scalar) -> VortexResult<ArrayRef> {
        if array.scalar().is_null() {
            Ok(ConstantArray::new(fill_value.clone(), array.len()).into_array())
        } else {
            cast(array.as_ref(), fill_value.dtype())
        }
    }
}

register_kernel!(FillNullKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod test {
    use vortex_scalar::Scalar;

    use crate::IntoArray as _;
    use crate::arrays::ConstantArray;
    use crate::arrow::IntoArrowArray as _;
    use crate::compute::fill_null;

    #[test]
    fn test_null() {
        let actual = fill_null(
            &ConstantArray::new(Scalar::null_native::<i32>(), 3).into_array(),
            &Scalar::from(1),
        )
        .unwrap();
        let expected = ConstantArray::new(Scalar::from(1), 3).into_array();

        assert!(!actual.dtype().is_nullable());

        let actual_arrow = actual.clone().into_arrow_preferred().unwrap();
        let expected_arrow = expected.clone().into_arrow_preferred().unwrap();
        assert_eq!(
            &actual_arrow,
            &expected_arrow,
            "{}, {}",
            actual.display_values(),
            expected.display_values()
        );
    }

    #[test]
    fn test_non_null() {
        let actual = fill_null(
            &ConstantArray::new(Scalar::from(Some(1)), 3).into_array(),
            &Scalar::from(1),
        )
        .unwrap();
        let expected = ConstantArray::new(Scalar::from(1), 3).into_array();

        assert!(!actual.dtype().is_nullable());

        let actual_arrow = actual.clone().into_arrow_preferred().unwrap();
        let expected_arrow = expected.clone().into_arrow_preferred().unwrap();
        assert_eq!(
            &actual_arrow,
            &expected_arrow,
            "{}, {}",
            actual.display_values(),
            expected.display_values()
        );
    }

    #[test]
    fn test_non_nullable_with_nullable() {
        let actual = fill_null(
            &ConstantArray::new(Scalar::from(1), 3).into_array(),
            &Scalar::from(Some(1)),
        )
        .unwrap();
        let expected = ConstantArray::new(Scalar::from(1), 3).into_array();

        assert!(!Scalar::from(1).dtype().is_nullable());

        assert!(actual.dtype().is_nullable());

        let actual_arrow = actual.clone().into_arrow_preferred().unwrap();
        let expected_arrow = expected.clone().into_arrow_preferred().unwrap();
        assert_eq!(
            &actual_arrow,
            &expected_arrow,
            "{}, {}",
            actual.display_values(),
            expected.display_values()
        );
    }
}
