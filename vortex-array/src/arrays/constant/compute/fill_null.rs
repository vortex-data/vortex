// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute::{FillNullKernel, FillNullKernelAdapter};
use crate::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ConstantVTable};

impl FillNullKernel for ConstantVTable {
    fn fill_null(&self, array: &ConstantArray, fill_value: &Scalar) -> VortexResult<ArrayRef> {
        if array.scalar().is_null() {
            Ok(ConstantArray::new(fill_value.clone(), array.len()).into_array())
        } else {
            Ok(array.to_array())
        }
    }
}

register_kernel!(FillNullKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod test {
    use vortex_scalar::Scalar;

    use crate::{
        IntoArray as _, arrays::ConstantArray, arrow::IntoArrowArray as _, compute::fill_null,
    };

    #[test]
    fn test_null() {
        let actual = fill_null(
            &ConstantArray::new(Scalar::from(Some(1)), 3).into_array(),
            &Scalar::from(1),
        )
        .unwrap();
        let expected = ConstantArray::new(Scalar::from(1), 3).into_array();

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
            &ConstantArray::new(Scalar::from(None::<i32>), 3).into_array(),
            &Scalar::from(1),
        )
        .unwrap();
        let expected = ConstantArray::new(Scalar::from(1), 3).into_array();

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
