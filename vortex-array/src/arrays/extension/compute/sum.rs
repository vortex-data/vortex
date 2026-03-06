// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::compute::SumKernel;
use crate::compute::SumKernelAdapter;
use crate::compute::{self};
use crate::register_kernel;
use crate::scalar::Scalar;

impl SumKernel for ExtensionVTable {
    fn sum(&self, array: &ExtensionArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        compute::sum_with_accumulator(array.storage(), accumulator)
    }
}

register_kernel!(SumKernelAdapter(ExtensionVTable).lift());
