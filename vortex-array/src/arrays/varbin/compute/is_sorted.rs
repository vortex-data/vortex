// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{VarBinArray, VarBinVTable};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

impl IsSortedKernel for VarBinVTable {
    fn is_sorted(&self, array: &VarBinArray) -> VortexResult<bool> {
        Ok(array.iter().is_sorted())
    }

    fn is_strict_sorted(&self, array: &VarBinArray) -> VortexResult<bool> {
        Ok(array.iter().is_strict_sorted())
    }
}

register_kernel!(IsSortedKernelAdapter(VarBinVTable).lift());
