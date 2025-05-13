use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{NumericKernel, NumericKernelAdapter, numeric};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::NumericOperator;

use crate::{SparseArray, SparseVTable};

impl NumericKernel for SparseVTable {
    fn numeric(
        &self,
        array: &SparseArray,
        rhs: &dyn Array,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let new_patches = array.patches().clone().map_values(|values| {
            let rhs_const_array = ConstantArray::new(rhs_scalar.clone(), values.len()).into_array();

            numeric(&values, &rhs_const_array, op)
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

register_kernel!(NumericKernelAdapter(SparseVTable).lift());
