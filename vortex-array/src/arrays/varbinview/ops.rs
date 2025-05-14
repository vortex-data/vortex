use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::arrays::{Ref, VarBinViewArray, VarBinViewVTable, varbin_scalar};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, Cost, IntoArray};

impl OperationsVTable<VarBinViewVTable> for VarBinViewVTable {
    fn slice(array: &VarBinViewArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let views = array.views().slice(start..stop);

        Ok(VarBinViewArray::try_new(
            views,
            array.buffers().to_vec(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }

    fn scalar_at(array: &VarBinViewArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index), array.dtype()))
    }

    fn is_constant(array: &VarBinViewArray, cost: Cost) -> VortexResult<Option<bool>> {
        if cost.is_negligible() {
            return Ok(None);
        }

        let mut views_iter = array.views().iter();
        let first_value = views_iter
            .next()
            .vortex_expect("is_constant is only invoked for len > 1");

        // For the array to be constant, all views must be of the same type
        if first_value.is_inlined() {
            let first_value = first_value.as_inlined();

            for view in views_iter {
                // Short circuit if the view is of the wrong type, then if both are inlined they must be equal.
                if !view.is_inlined() || view.as_inlined() != first_value {
                    return Ok(Some(false));
                }
            }
        } else {
            // Directly fetch the values for a `Ref`
            let ref_bytes = |view_ref: &Ref| {
                &array.buffer(view_ref.buffer_index() as usize).as_slice()[view_ref.to_range()]
            };

            let first_view_ref = first_value.as_view();
            let first_value_bytes = ref_bytes(first_view_ref);

            for view in views_iter {
                // Short circuit if the view is of the wrong type
                if view.is_inlined() || view.len() != first_value.len() {
                    return Ok(Some(false));
                }

                let view_ref = view.as_view();
                let value = ref_bytes(view_ref);
                if value != first_value_bytes {
                    return Ok(Some(false));
                }
            }
        }

        Ok(Some(true))
    }
}
