// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{NumericKernel, NumericKernelAdapter, numeric};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::NumericOperator;

use crate::{RunEndArray, RunEndVTable};

impl NumericKernel for RunEndVTable {
    fn numeric(
        &self,
        array: &RunEndArray,
        rhs: &dyn Array,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let rhs_const_array = ConstantArray::new(rhs_scalar, array.values().len()).into_array();

        // SAFETY: ends are preserved.
        unsafe {
            Ok(Some(
                RunEndArray::new_unchecked(
                    array.ends().clone(),
                    numeric(array.values(), &rhs_const_array, op)?,
                    array.offset(),
                    array.len(),
                )
                .into_array(),
            ))
        }
    }
}

register_kernel!(NumericKernelAdapter(RunEndVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;

    use crate::RunEndArray;

    #[rstest]
    #[case::runend_i32_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([10i32, 10, 10, 20, 20, 30, 30, 30, 30]).into_array()
    ).unwrap())]
    #[case::runend_u32_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([100u32, 100, 200, 200, 200]).into_array()
    ).unwrap())]
    #[case::runend_i64_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([1000i64, 1000, 2000, 2000, 3000, 3000]).into_array()
    ).unwrap())]
    #[case::runend_u64_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([5000u64, 5000, 5000, 6000, 6000]).into_array()
    ).unwrap())]
    #[case::runend_f32_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([1.5f32, 1.5, 2.5, 2.5, 3.5]).into_array()
    ).unwrap())]
    #[case::runend_f64_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([10.1f64, 10.1, 20.2, 20.2, 20.2]).into_array()
    ).unwrap())]
    #[case::runend_i32_large(RunEndArray::encode(
        PrimitiveArray::from_iter((0..100).map(|i| i / 5)).into_array()
    ).unwrap())]
    fn test_runend_binary_numeric(#[case] array: RunEndArray) {
        test_binary_numeric_array(array.into_array());
    }
}
