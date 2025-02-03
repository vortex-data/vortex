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

        let rhs_const_array = ConstantArray::new(rhs_scalar, array.values().len()).into_array();

        DictArray::try_new(
            array.codes(),
            binary_numeric(&array.values(), &rhs_const_array, op)?,
        )
        .map(IntoArray::into_array)
        .map(Some)
    }
}
