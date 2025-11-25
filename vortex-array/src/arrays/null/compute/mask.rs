// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::register_kernel;

impl MaskKernel for NullVTable {
    fn mask(&self, array: &NullArray, _mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(array.to_array())
    }
}

register_kernel!(MaskKernelAdapter(NullVTable).lift());
