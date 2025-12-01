// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::NumericOperator;

use super::DictArray;
use super::DictVTable;
use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::compute::NumericKernel;
use crate::compute::NumericKernelAdapter;
use crate::compute::numeric;
use crate::register_kernel;

impl NumericKernel for DictVTable {
    fn numeric(
        &self,
        lhs: &DictArray,
        rhs: &dyn Array,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we have more values than codes, it is faster to canonicalise first.
        if lhs.values().len() > lhs.codes().len() {
            return Ok(None);
        }

        // Only push down if all values are referenced to avoid incorrect results
        // See: https://github.com/vortex-data/vortex/pull/4560
        // Unchecked operation will be fine to pushdown.
        if !lhs.has_all_values_referenced() {
            return Ok(None);
        }

        // If the RHS is constant, then we just need to apply the operation to our encoded values.
        if let Some(rhs_scalar) = rhs.as_constant() {
            let values_result = numeric(
                lhs.values(),
                ConstantArray::new(rhs_scalar, lhs.values().len()).as_ref(),
                op,
            )?;

            // SAFETY: values len preserved, codes all still point to valid values
            // all_values_referenced preserved since operation doesn't change which values are referenced
            let result = unsafe {
                DictArray::new_unchecked(lhs.codes().clone(), values_result)
                    .set_all_values_referenced(lhs.has_all_values_referenced())
                    .into_array()
            };

            return Ok(Some(result));
        }

        // It's a little more complex, but we could perform binary operations against the dictionary
        // values in the future.
        Ok(None)
    }
}

register_kernel!(NumericKernelAdapter(DictVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_scalar::NumericOperator;

    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::DictArray;
    use crate::assert_arrays_eq;
    use crate::compute::numeric;

    #[test]
    fn test_add_const() {
        // Create a dict with all_values_referenced = true
        let dict = unsafe {
            DictArray::new_unchecked(
                buffer![0u32, 1, 2, 0, 1].into_array(),
                buffer![10i32, 20, 30].into_array(),
            )
            .set_all_values_referenced(true)
        };

        let res = numeric(
            dict.as_ref(),
            ConstantArray::new(5i32, 5).as_ref(),
            NumericOperator::Add,
        )
        .unwrap();

        let expected = PrimitiveArray::from_iter([15i32, 25, 35, 15, 25]);
        assert_arrays_eq!(
            res.to_canonical().unwrap().into_array(),
            expected.to_array()
        );
    }

    #[test]
    fn test_mul_const() {
        // Create a dict with all_values_referenced = true
        let dict = unsafe {
            DictArray::new_unchecked(
                buffer![0u32, 1, 2, 1, 0].into_array(),
                buffer![2i32, 3, 5].into_array(),
            )
            .set_all_values_referenced(true)
        };

        let res = numeric(
            dict.as_ref(),
            ConstantArray::new(10i32, 5).as_ref(),
            NumericOperator::Mul,
        )
        .unwrap();

        let expected = PrimitiveArray::from_iter([20i32, 30, 50, 30, 20]);
        assert_arrays_eq!(
            res.to_canonical().unwrap().into_array(),
            expected.to_array()
        );
    }

    #[test]
    fn test_no_pushdown_when_not_all_values_referenced() {
        // Create a dict with all_values_referenced = false (default)
        let dict = DictArray::try_new(
            buffer![0u32, 1, 0, 1].into_array(),
            buffer![10i32, 20, 30].into_array(), // value at index 2 is not referenced
        )
        .unwrap();

        // Should return None, indicating no pushdown
        let res = numeric(
            dict.as_ref(),
            ConstantArray::new(5i32, 4).as_ref(),
            NumericOperator::Add,
        )
        .unwrap();

        // Verify the result by canonicalizing
        let expected = PrimitiveArray::from_iter([15i32, 25, 15, 25]);
        assert_arrays_eq!(
            res.to_canonical().unwrap().into_array(),
            expected.to_array()
        );
    }

    #[test]
    fn test_sub_const() {
        // Create a dict with all_values_referenced = true
        let dict = unsafe {
            DictArray::new_unchecked(
                buffer![0u32, 1, 2].into_array(),
                buffer![100i32, 50, 25].into_array(),
            )
            .set_all_values_referenced(true)
        };

        let res = numeric(
            dict.as_ref(),
            ConstantArray::new(10i32, 3).as_ref(),
            NumericOperator::Sub,
        )
        .unwrap();

        let expected = PrimitiveArray::from_iter([90i32, 40, 15]);
        assert_arrays_eq!(
            res.to_canonical().unwrap().into_array(),
            expected.to_array()
        );
    }
}
