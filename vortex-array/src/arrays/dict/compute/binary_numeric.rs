// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{ConstantArray, DictArray, DictVTable};
use crate::compute::{NumericKernel, NumericKernelAdapter, numeric};
use crate::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::NumericOperator;

impl NumericKernel for DictVTable {
    fn numeric(
        &self,
        array: &DictArray,
        rhs: &dyn Array,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        // if we have more values than codes, it is faster to canonicalise first.
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }

        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };
        let rhs_const_array = ConstantArray::new(rhs_scalar, array.values().len()).into_array();

        Ok(Some(
            DictArray::try_new(
                array.codes().clone(),
                numeric(array.values(), &rhs_const_array, op)?,
            )?
            .into_array(),
        ))
    }
}

register_kernel!(NumericKernelAdapter(DictVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use crate::ArrayRef;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::binary_numeric::test_binary_numeric_array;

    use crate::builders::dict::dict_encode;

    fn sliced_dict_array() -> ArrayRef {
        let reference = PrimitiveArray::from_option_iter([
            Some(42),
            Some(-9),
            None,
            Some(42),
            Some(1),
            Some(5),
        ]);
        let dict = dict_encode(reference.as_ref()).unwrap();
        dict.slice(1, 4).unwrap()
    }

    #[test]
    fn test_dict_binary_numeric() {
        let array = sliced_dict_array();
        test_binary_numeric_array(array)
    }

    use crate::IntoArray;

    #[rstest]
    #[case::dict_i32_basic(dict_encode(PrimitiveArray::from_iter([10i32, 20, 10, 30, 20, 10]).as_ref()).unwrap().into_array())]
    #[case::dict_u32_basic(dict_encode(PrimitiveArray::from_iter([100u32, 200, 100, 300, 200]).as_ref()).unwrap().into_array())]
    #[case::dict_i64_basic(dict_encode(PrimitiveArray::from_iter([1000i64, 2000, 1000, 3000, 2000, 1000]).as_ref()).unwrap().into_array())]
    #[case::dict_u64_basic(dict_encode(PrimitiveArray::from_iter([5000u64, 6000, 5000, 7000, 6000]).as_ref()).unwrap().into_array())]
    #[case::dict_f32_basic(dict_encode(PrimitiveArray::from_iter([1.5f32, 2.5, 1.5, 3.5, 2.5]).as_ref()).unwrap().into_array())]
    #[case::dict_f64_basic(dict_encode(PrimitiveArray::from_iter([10.1f64, 20.2, 10.1, 30.3, 20.2]).as_ref()).unwrap().into_array())]
    #[case::dict_i32_sliced(dict_encode(PrimitiveArray::from_iter([100i32, 200, 100, 300, 200, 100]).as_ref()).unwrap().slice(1, 5).unwrap())]
    #[case::dict_nullable(dict_encode(PrimitiveArray::from_option_iter([Some(42i32), None, Some(42), Some(1), None]).as_ref()).unwrap().into_array())]
    fn test_dict_binary_numeric_rstest(#[case] array: ArrayRef) {
        test_binary_numeric_array(array)
    }
}
