// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::compute::CompareKernel;
use vortex_array::compute::CompareKernelAdapter;
use vortex_array::compute::Operator;
use vortex_array::compute::compare;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;

impl CompareKernel for BitPackedVTable {
    fn compare(
        &self,
        lhs: &BitPackedArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // Handle comparison with constant by decompressing and comparing.
        // This is a fallback implementation - more optimized versions could use
        // statistics or operate directly on packed data for certain cases.
        if rhs.as_constant().is_some() {
            return compare(&lhs.clone().to_canonical()?.into_array(), rhs, operator).map(Some);
        }

        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(BitPackedVTable).lift());
