// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{NullArray, NullVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::{ArrayRef, register_kernel};

impl MaskKernel for NullVTable {
    fn mask(&self, array: &NullArray, _mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(array.to_array())
    }
}

register_kernel!(MaskKernelAdapter(NullVTable).lift());
