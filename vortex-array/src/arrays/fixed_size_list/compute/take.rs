// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl TakeKernel for FixedSizeListVTable {
    fn take(&self, array: &FixedSizeListArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        todo!()
    }
}

register_kernel!(TakeKernelAdapter(FixedSizeListVTable).lift());
