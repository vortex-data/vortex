// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantVTable;
use vortex_array::compute::BetweenKernel;
use vortex_array::compute::BetweenOptions;
use vortex_array::compute::between;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;

impl BetweenKernel for BitPackedVTable {
    fn between(
        array: &BitPackedArray,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !lower.is::<ConstantVTable>() || !upper.is::<ConstantVTable>() {
            return Ok(None);
        };

        between(
            &array.clone().to_canonical()?.into_array(),
            lower,
            upper,
            options,
        )
        .map(Some)
    }
}
