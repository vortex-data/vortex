use vortex_array::array::ConstantArray;
use vortex_array::compute::{binary_numeric, BinaryNumericFn};
use vortex_array::{Array, IntoArray};
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::BinaryNumericOperator;

use crate::{SparseArray, SparseEncoding};

impl BinaryNumericFn<SparseArray> for SparseEncoding {
    fn binary_numeric(
        &self,
        array: &SparseArray,
        rhs: &Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<Array>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let new_patches = array.patches().map_values(|values| {
            let rhs_const_array = ConstantArray::new(rhs_scalar.clone(), values.len()).into_array();

            binary_numeric(&values, &rhs_const_array, op)
        })?;
        let new_fill_value = array
            .fill_scalar()
            .as_primitive()
            .checked_binary_numeric(rhs_scalar.as_primitive(), op)?
            .ok_or_else(|| vortex_err!("numeric overflow"))?
            .into();
        SparseArray::try_new_from_patches(
            new_patches,
            array.len(),
            array.indices_offset(),
            new_fill_value,
        )
        .map(IntoArray::into_array)
        .map(Some)
    }
}
