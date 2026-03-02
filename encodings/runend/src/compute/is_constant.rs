// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::DynArray;
use vortex_array::compute::IsConstantKernel;
use vortex_array::compute::IsConstantKernelAdapter;
use vortex_array::compute::IsConstantOpts;
use vortex_array::compute::is_constant_opts;
use vortex_array::expr::stats::Stat;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::RunEndVTable;

impl IsConstantKernel for RunEndVTable {
    fn is_constant(
        &self,
        array: &Self::Array,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        // If there are known to be me 0 len runs then we can check if constant on the values.
        debug_assert_eq!(
            array
                .ends()
                .statistics()
                .compute_as::<bool>(Stat::IsStrictSorted),
            Some(true)
        );
        is_constant_opts(array.values(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(RunEndVTable).lift());
