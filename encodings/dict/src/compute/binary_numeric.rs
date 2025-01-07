use vortex_array::array::ConstantArray;
use vortex_array::compute::{binary_numeric, BinaryNumericFn};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;
use vortex_scalar::BinaryNumericOperator;

use crate::{DictArray, DictEncoding};

impl BinaryNumericFn<DictArray> for DictEncoding {
    fn binary_numeric(
        &self,
        array: &DictArray,
        rhs: &ArrayData,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let rhs_const_array = ConstantArray::new(rhs_scalar, array.values().len()).into_array();

        DictArray::try_new(
            array.codes(),
            binary_numeric(&array.values(), &rhs_const_array, op)?,
        )
        .map(IntoArrayData::into_array)
        .map(Some)
    }
}
