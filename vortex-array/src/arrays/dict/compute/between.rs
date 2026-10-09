// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::between::BetweenKernel;
use crate::scalar_fn::fns::between::BetweenOptions;

impl BetweenKernel for Dict {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };
        let values = array.values().clone().between(
            ConstantArray::new(lower, array.values().len()).into_array(),
            ConstantArray::new(upper, array.values().len()).into_array(),
            options.clone(),
        )?;
        // SAFETY: between preserves the dictionary length and the codes are unchanged.
        let result = unsafe {
            DictArray::new_unchecked(array.codes().clone(), values)
                .set_all_values_referenced(array.has_all_values_referenced())
        };
        result
            .into_array()
            .execute::<Canonical>(ctx)
            .map(|result| Some(result.into_array()))
    }
}

#[cfg(test)]
mod tests;
