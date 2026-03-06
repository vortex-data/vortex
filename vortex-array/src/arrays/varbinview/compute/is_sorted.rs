// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::varbinview::VarBinViewArray;
use crate::arrays::varbinview::VarBinViewVTable;
use crate::compute::IsSortedIteratorExt;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::register_kernel;

impl IsSortedKernel for VarBinViewVTable {
    fn is_sorted(&self, array: &VarBinViewArray) -> VortexResult<Option<bool>> {
        Ok(Some(
            array.with_iterator(|bytes_iter| bytes_iter.is_sorted()),
        ))
    }

    fn is_strict_sorted(&self, array: &VarBinViewArray) -> VortexResult<Option<bool>> {
        Ok(Some(
            array.with_iterator(|bytes_iter| bytes_iter.is_strict_sorted()),
        ))
    }
}

register_kernel!(IsSortedKernelAdapter(VarBinViewVTable).lift());
