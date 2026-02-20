// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::zero;
use vortex_array::compute::IsSortedKernel;
use vortex_array::compute::IsSortedKernelAdapter;
use vortex_array::match_each_native_ptype;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::SequenceArray;
use crate::SequenceVTable;

impl IsSortedKernel for SequenceVTable {
    fn is_sorted(&self, array: &SequenceArray) -> VortexResult<Option<bool>> {
        let m = array.multiplier();
        match_each_native_ptype!(m.ptype(), |P| {
            m.cast::<P>().map(|x| Some(x >= zero::<P>()))
        })
    }

    fn is_strict_sorted(&self, array: &SequenceArray) -> VortexResult<Option<bool>> {
        let m = array.multiplier();
        match_each_native_ptype!(m.ptype(), |P| {
            m.cast::<P>().map(|x| Some(x > zero::<P>()))
        })
    }
}

register_kernel!(IsSortedKernelAdapter(SequenceVTable).lift());
