// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::FillNullReduce;
use crate::compute::cast;

impl FillNullReduce for ConstantVTable {
    fn fill_null(array: &ConstantArray, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>> {
        if array.scalar().is_null() {
            Ok(Some(
                ConstantArray::new(fill_value.clone(), array.len()).into_array(),
            ))
        } else {
            cast(array.as_ref(), fill_value.dtype()).map(Some)
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_scalar::Scalar;

    use crate::IntoArray as _;
    use crate::arrays::ConstantArray;
    use crate::arrow::IntoArrowArray as _;
    use crate::builtins::ArrayBuiltins;

    #[test]
    fn test_null() {
        let actual = ConstantArray::new(Scalar::null_native::<i32>(), 3)
            .into_array()
            .fill_null(Scalar::from(1))
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
        let actual = ConstantArray::new(Scalar::from(Some(1)), 3)
            .into_array()
            .fill_null(Scalar::from(1))
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
        let actual = ConstantArray::new(Scalar::from(1), 3)
            .into_array()
            .fill_null(Scalar::from(Some(1)))
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
