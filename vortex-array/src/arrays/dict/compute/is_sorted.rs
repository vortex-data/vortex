// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::compute::is_sorted;
use crate::compute::is_strict_sorted;
use crate::register_kernel;

impl IsSortedKernel for Dict {
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

register_kernel!(IsSortedKernelAdapter(Dict).lift());
