// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinViewArray, VarBinViewVTable};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

impl IsSortedKernel for VarBinViewVTable {
    fn is_sorted(&self, array: &VarBinViewArray) -> VortexResult<Option<bool>> {
        array
            .with_iterator(|bytes_iter| bytes_iter.is_sorted())
            .map(Some)
    }

    fn is_strict_sorted(&self, array: &VarBinViewArray) -> VortexResult<Option<bool>> {
        array
            .with_iterator(|bytes_iter| bytes_iter.is_strict_sorted())
            .map(Some)
    }
}

register_kernel!(IsSortedKernelAdapter(VarBinViewVTable).lift());
