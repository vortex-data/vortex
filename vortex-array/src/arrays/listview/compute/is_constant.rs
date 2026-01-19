// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::compute::IsConstantKernel;
use crate::compute::IsConstantKernelAdapter;
use crate::compute::IsConstantOpts;
use crate::compute::is_constant_opts;
use crate::register_kernel;

impl IsConstantKernel for ListViewVTable {
    fn is_constant(
        &self,
        array: &ListViewArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        // At this point, we're guaranteed:
        // - Array has at least 2 elements
        // - All elements are valid (no nulls)

        // First check if all list sizes are constant.
        if !is_constant_opts(array.sizes().as_ref(), opts)?.unwrap_or_default() {
            return Ok(Some(false));
        }

        // If checking is too expensive, return None early.
        if opts.is_negligible_cost() {
            return Ok(None);
        }

        // Get the first scalar for comparison.
        debug_assert!(
            array.len() > 1,
            "precondition for `is_constant` is incorrect"
        );
        let first_scalar = array.scalar_at(0);

        // Compare all other scalars to the first.
        for i in 1..array.len() {
            if array.scalar_at(i) != first_scalar {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(ListViewVTable).lift());
