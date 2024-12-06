use vortex_error::VortexResult;
use vortex_scalar::NumericOperator;

use crate::array::{SparseArray, SparseEncoding};
use crate::compute::{binary_numeric, BinaryNumericFn};
use crate::{ArrayData, ArrayLen as _, IntoArrayData};

impl BinaryNumericFn<SparseArray> for SparseEncoding {
    fn binary_numeric(
        &self,
        array: &SparseArray,
        rhs: &ArrayData,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let new_patches = array
            .patches()
            .map_values(|values| binary_numeric(&values, rhs, op))?;
        let new_fill_value = array
            .fill_scalar()
            .as_primitive()
            .numeric_operator(rhs_scalar.as_primitive(), op)?;
        SparseArray::try_new_from_patches(
            new_patches,
            array.len(),
            array.indices_offset(),
            new_fill_value,
        )
        .map(IntoArrayData::into_array)
        .map(Some)
    }
}
