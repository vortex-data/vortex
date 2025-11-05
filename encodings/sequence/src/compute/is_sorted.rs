// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::zero;
use vortex_array::compute::{IsSortedKernel, IsSortedKernelAdapter};
use vortex_array::register_kernel;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::{SequenceArray, SequenceVTable};

impl IsSortedKernel for SequenceVTable {
    fn is_sorted(&self, array: &SequenceArray) -> VortexResult<Option<bool>> {
        let m = array.multiplier();
        match_each_native_ptype!(m.ptype(), |P| { Ok(Some(m.cast::<P>() >= zero::<P>())) })
    }

    fn is_strict_sorted(&self, array: &SequenceArray) -> VortexResult<Option<bool>> {
        let m = array.multiplier();
        match_each_native_ptype!(m.ptype(), |P| { Ok(Some(m.cast::<P>() > zero::<P>())) })
    }
}

register_kernel!(IsSortedKernelAdapter(SequenceVTable).lift());
