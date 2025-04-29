use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{NumericKernel, NumericKernelAdapter, numeric};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::NumericOperator;

use crate::{DictArray, DictEncoding};

impl NumericKernel for DictEncoding {
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

register_kernel!(NumericKernelAdapter(DictEncoding).lift());

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_numeric;
    use vortex_array::compute::slice;

    use crate::builders::dict_encode;

    fn sliced_dict_array() -> ArrayRef {
        let reference = PrimitiveArray::from_option_iter([
            Some(42),
            Some(-9),
            None,
            Some(42),
            Some(1),
            Some(5),
        ]);
        let dict = dict_encode(&reference).unwrap();
        slice(&dict, 1, 4).unwrap()
    }

    #[test]
    fn test_dict_binary_numeric() {
        let array = sliced_dict_array();
        test_numeric::<i32>(array)
    }
}
