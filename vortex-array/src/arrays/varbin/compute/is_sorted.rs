// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::compute::IsSortedIteratorExt;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::register_kernel;

impl IsSortedKernel for VarBin {
    fn is_sorted(&self, array: &VarBinArray) -> VortexResult<Option<bool>> {
        Ok(Some(
            array.with_iterator(|bytes_iter| bytes_iter.is_sorted()),
        ))
    }

    fn is_strict_sorted(&self, array: &VarBinArray) -> VortexResult<Option<bool>> {
        Ok(Some(
            array.with_iterator(|bytes_iter| bytes_iter.is_strict_sorted()),
        ))
    }
}

register_kernel!(IsSortedKernelAdapter(VarBin).lift());
