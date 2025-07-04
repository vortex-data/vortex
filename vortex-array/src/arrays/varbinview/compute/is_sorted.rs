// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinViewArray, VarBinViewVTable};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

impl IsSortedKernel for VarBinViewVTable {
    fn is_sorted(&self, array: &VarBinViewArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_sorted())
    }

    fn is_strict_sorted(&self, array: &VarBinViewArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_strict_sorted())
    }
}

register_kernel!(IsSortedKernelAdapter(VarBinViewVTable).lift());
