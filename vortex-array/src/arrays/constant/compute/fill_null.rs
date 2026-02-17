// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::expr::FillNullReduce;
use crate::expr::fill_null_constant;
use crate::scalar::Scalar;

impl FillNullReduce for ConstantVTable {
    fn fill_null(array: &ConstantArray, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>> {
        fill_null_constant(array, fill_value).map(Some)
    }
}

#[cfg(test)]
mod test {
    use crate::IntoArray as _;
    use crate::arrays::ConstantArray;
    use crate::arrow::IntoArrowArray as _;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::Scalar;

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
