// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::compute::FilterKernel;
use crate::compute::FilterKernelAdapter;
use crate::register_kernel;

impl FilterKernel for NullVTable {
    fn filter(&self, _array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(mask.true_count()).into_array())
    }
}

register_kernel!(FilterKernelAdapter(NullVTable).lift());
