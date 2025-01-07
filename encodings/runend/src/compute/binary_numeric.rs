use vortex_array::array::ConstantArray;
use vortex_array::compute::{binary_numeric, BinaryNumericFn};
use vortex_array::{ArrayData, ArrayLen, IntoArrayData};
use vortex_error::VortexResult;
use vortex_scalar::BinaryNumericOperator;

use crate::{RunEndArray, RunEndEncoding};

impl BinaryNumericFn<RunEndArray> for RunEndEncoding {
    fn binary_numeric(
        &self,
        array: &RunEndArray,
        rhs: &ArrayData,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let rhs_const_array = ConstantArray::new(rhs_scalar, array.values().len()).into_array();

        RunEndArray::with_offset_and_length(
            array.ends(),
            binary_numeric(&array.values(), &rhs_const_array, op)?,
            array.offset(),
            array.len(),
        )
        .map(IntoArrayData::into_array)
        .map(Some)
    }
}
