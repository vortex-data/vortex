// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{IsSortedKernel, IsSortedKernelAdapter, is_sorted, is_strict_sorted};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{FoRArray, FoRVTable};

impl IsSortedKernel for FoRVTable {
    fn is_sorted(&self, array: &FoRArray) -> VortexResult<Option<bool>> {
        is_sorted(array.encoded())
    }

    fn is_strict_sorted(&self, array: &FoRArray) -> VortexResult<Option<bool>> {
        is_strict_sorted(array.encoded())
    }
}

register_kernel!(IsSortedKernelAdapter(FoRVTable).lift());
