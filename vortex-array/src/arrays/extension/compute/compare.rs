// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ExtensionArray, ExtensionVTable};
use crate::compute::{self, CompareKernel, CompareKernelAdapter, Operator};
use crate::{Array, ArrayRef, register_kernel};

impl CompareKernel for ExtensionVTable {
    fn compare(
        &self,
        lhs: &ExtensionArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is a constant, we can extract the storage scalar.
        if let Some(const_ext) = rhs.as_constant() {
            let storage_scalar = const_ext.as_extension().storage();
            return compute::compare(
                lhs.storage(),
                ConstantArray::new(storage_scalar, lhs.len()).as_ref(),
                operator,
            )
            .map(Some);
        }

        // If the RHS is an extension array matching ours, we can extract the storage.
        if let Some(rhs_ext) = rhs.as_opt::<ExtensionVTable>() {
            return compute::compare(lhs.storage(), rhs_ext.storage(), operator).map(Some);
        }

        // Otherwise, we need the RHS to handle this comparison.
        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(ExtensionVTable).lift());
