// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{self, SumKernel, SumKernelAdapter};
use crate::register_kernel;

impl SumKernel for ExtensionVTable {
    fn sum(&self, array: &ExtensionArray) -> VortexResult<Scalar> {
        compute::sum(array.storage())
    }
}

register_kernel!(SumKernelAdapter(ExtensionVTable).lift());
