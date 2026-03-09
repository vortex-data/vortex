// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::arrays::varbinview::Ref;
use crate::compute::IsConstantKernel;
use crate::compute::IsConstantKernelAdapter;
use crate::compute::IsConstantOpts;
use crate::register_kernel;

impl IsConstantKernel for VarBinViewVTable {
    fn is_constant(
        &self,
        array: &VarBinViewArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let mut views_iter = array.views().iter();
        let first_value = views_iter
            .next()
            .vortex_expect("Must have at least one value");

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
                &array.buffer(view_ref.buffer_index as usize).as_slice()[view_ref.as_range()]
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

register_kernel!(IsConstantKernelAdapter(VarBinViewVTable).lift());
