// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::constant_fits_values;
use crate::arrays::widen::compute::constant_in_values_dtype;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::BetweenReduce;
use crate::scalar_fn::fns::between::as_two_compares;

impl BetweenReduce for Widen {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = array.values();

        if let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant())
            && constant_fits_values(lower, values.dtype()) == Some(true)
            && constant_fits_values(upper, values.dtype()) == Some(true)
        {
            let lower = constant_in_values_dtype(lower, values, array.len())?;
            let upper = constant_in_values_dtype(upper, values, array.len())?;
            return values
                .clone()
                .between(lower, upper, options.clone())
                .map(Some);
        }

        // Each compare of a `Widen` with a constant or another `Widen` runs the compare kernel of
        // `Widen`. Any other bound declines, so that `Between` decodes the `Widen` once.
        let is_operand = |bound: &ArrayRef| bound.is::<Constant>() || bound.is::<Widen>();
        if !is_operand(lower) || !is_operand(upper) {
            return Ok(None);
        }

        as_two_compares(array.array(), lower, upper, options).map(Some)
    }
}
