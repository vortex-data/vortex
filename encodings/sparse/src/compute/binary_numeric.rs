use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{binary_numeric, BinaryNumericFn};
use vortex_array::{Array, ArrayRef};
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::BinaryNumericOperator;

use crate::{SparseArray, SparseEncoding};

impl BinaryNumericFn<&SparseArray> for SparseEncoding {
    fn binary_numeric(
        &self,
        array: &SparseArray,
        rhs: &dyn Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let new_patches = array.patches().clone().map_values(|values| {
            let rhs_const_array = ConstantArray::new(rhs_scalar.clone(), values.len()).into_array();

            binary_numeric(&values, &rhs_const_array, op)
        })?;
        let new_fill_value = array
            .fill_scalar()
            .as_primitive()
            .checked_binary_numeric(&rhs_scalar.as_primitive(), op)
            .ok_or_else(|| vortex_err!("numeric overflow"))?
            .into();
        Ok(Some(
            SparseArray::try_new_from_patches(new_patches, new_fill_value)?.into_array(),
        ))
    }
}
