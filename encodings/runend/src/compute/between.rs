// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;
use crate::decompress_bool::runend_decode_bools;

impl BetweenKernel for RunEnd {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };
        let values = array.values().clone().between(
            ConstantArray::new(lower, array.values().len()).into_array(),
            ConstantArray::new(upper, array.values().len()).into_array(),
            options.clone(),
        )?;
        runend_decode_bools(
            array.ends().clone().execute::<PrimitiveArray>(ctx)?,
            values.execute::<BoolArray>(ctx)?,
            array.offset(),
            array.len(),
            ctx,
        )
        .map(Some)
    }
}

#[cfg(test)]
mod tests;
