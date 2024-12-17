use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::BinaryNumericOperator;

use crate::array::{SparseArray, SparseEncoding};
use crate::compute::{binary_numeric, BinaryNumericFn};
use crate::{ArrayData, ArrayLen as _, IntoArrayData};

impl BinaryNumericFn<SparseArray> for SparseEncoding {
    fn binary_numeric(
        &self,
        array: &SparseArray,
        rhs: &ArrayData,
        op: BinaryNumericOperator,
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
            .checked_numeric_operator(rhs_scalar.as_primitive(), op)?
            .ok_or_else(|| vortex_err!("numeric overflow"))?;
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
