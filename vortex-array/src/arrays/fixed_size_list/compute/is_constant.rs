// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{
    FixedSizeListArray,
    FixedSizeListVTable,
};
use crate::compute::{
    IsConstantKernel,
    IsConstantKernelAdapter,
    IsConstantOpts,
};
use crate::register_kernel;

/// IsConstant implementation for [`FixedSizeListArray`].
///
/// Compares each list scalar against the first to determine if all lists are identical.
impl IsConstantKernel for FixedSizeListVTable {
    fn is_constant(
        &self,
        array: &FixedSizeListArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        // Since all of the lists have fixed size, we just need to check that each list scalar is
        // identical. Note that this check is always "expensive".

        debug_assert!(
            array.len() > 1,
            "precondition for `is_constant` is incorrect"
        );
        let first_scalar = array.scalar_at(0); // We checked the array length above.

        // TODO(connor): There must be a more efficient way to do this. Each `scalar_at()` call
        // makes several allocations...
        for i in 1..array.len() {
            let current_scalar = array.scalar_at(i);
            if current_scalar != first_scalar {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(FixedSizeListVTable).lift());
