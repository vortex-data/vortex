// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{IsSortedKernel, IsSortedKernelAdapter, is_sorted, is_strict_sorted};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{DictArray, DictVTable};

impl IsSortedKernel for DictVTable {
    fn is_sorted(&self, array: &DictArray) -> VortexResult<Option<bool>> {
        if Some((true, true)) == is_sorted(array.values())?.zip(is_sorted(array.codes())?) {
            return Ok(Some(true));
        }
        Ok(None)
    }

    fn is_strict_sorted(&self, array: &DictArray) -> VortexResult<Option<bool>> {
        if Some((true, true))
            == is_strict_sorted(array.values())?.zip(is_strict_sorted(array.codes())?)
        {
            return Ok(Some(true));
        }
        Ok(None)
    }
}

register_kernel!(IsSortedKernelAdapter(DictVTable).lift());
