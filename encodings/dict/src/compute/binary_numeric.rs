use vortex_array::array::ConstantArray;
use vortex_array::compute::{binary_numeric, BinaryNumericFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::BinaryNumericOperator;

use crate::{DictArray, DictEncoding};

impl BinaryNumericFn<DictArray> for DictEncoding {
    fn binary_numeric(
        &self,
        array: &DictArray,
        rhs: &Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<Array>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let rhs_const_array = ConstantArray::new(
            rhs_scalar.cast(array.values().dtype())?,
            array.values().len(),
        )
        .into_array();

        DictArray::try_new(
            array.codes(),
            binary_numeric(&array.values(), &rhs_const_array, op)?,
        )
        .map(IntoArray::into_array)
        .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::slice;
    use vortex_array::compute::test_harness::test_binary_numeric;
    use vortex_array::Array;

    use crate::dict_encode;

    fn sliced_dict_array() -> Array {
        let reference = PrimitiveArray::from_option_iter([
            Some(42),
            Some(-9),
            None,
            Some(42),
            Some(1),
            Some(5),
        ]);
        let dict = dict_encode(reference.as_ref()).unwrap();
        slice(dict, 1, 4).unwrap()
    }

    #[test]
    fn test_dict_binary_numeric() {
        let array = sliced_dict_array();
        test_binary_numeric::<i32>(array)
    }
}
