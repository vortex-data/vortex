// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{
    VarBinArray,
    VarBinVTable,
};
use crate::compute::{
    IsSortedIteratorExt,
    IsSortedKernel,
    IsSortedKernelAdapter,
};
use crate::register_kernel;

impl IsSortedKernel for VarBinVTable {
    fn is_sorted(&self, array: &VarBinArray) -> VortexResult<Option<bool>> {
        array
            .with_iterator(|bytes_iter| bytes_iter.is_sorted())
            .map(Some)
    }

    fn is_strict_sorted(&self, array: &VarBinArray) -> VortexResult<Option<bool>> {
        array
            .with_iterator(|bytes_iter| bytes_iter.is_strict_sorted())
            .map(Some)
    }
}

register_kernel!(IsSortedKernelAdapter(VarBinVTable).lift());
