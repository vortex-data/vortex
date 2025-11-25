// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::FilterKernel;
use crate::compute::FilterKernelAdapter;
use crate::register_kernel;

impl FilterKernel for ConstantVTable {
    fn filter(&self, array: &ConstantArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().clone(), mask.true_count()).into_array())
    }
}

register_kernel!(FilterKernelAdapter(ConstantVTable).lift());
