// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::BetweenReduce;

impl BetweenReduce for Masked {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        array
            .child()
            .clone()
            .between(lower.clone(), upper.clone(), options.clone())?
            .mask(array.masked_validity().to_array(array.len()))
            .map(Some)
    }
}

#[cfg(test)]
mod tests;
