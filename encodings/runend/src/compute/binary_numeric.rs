use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{BinaryNumericFn, binary_numeric};
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::BinaryNumericOperator;

use crate::{RunEndArray, RunEndEncoding};

impl BinaryNumericFn<&RunEndArray> for RunEndEncoding {
    fn binary_numeric(
        &self,
        array: &RunEndArray,
        rhs: &dyn Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let rhs_const_array = ConstantArray::new(rhs_scalar, array.values().len()).into_array();

        Ok(Some(
            RunEndArray::with_offset_and_length(
                array.ends().clone(),
                binary_numeric(array.values(), &rhs_const_array, op)?,
                array.offset(),
                array.len(),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric;

    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
    }

    #[test]
    fn test_runend_binary_numeric() {
        let array = ree_array().into_array();
        test_binary_numeric::<i32>(array)
    }
}
