// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{self, SumKernel, SumKernelAdapter};
use crate::register_kernel;

impl SumKernel for ExtensionVTable {
    fn sum(&self, array: &ExtensionArray, initial_value: &Scalar) -> VortexResult<Scalar> {
        compute::sum_with_initial(array.storage(), initial_value)
    }
}

register_kernel!(SumKernelAdapter(ExtensionVTable).lift());
