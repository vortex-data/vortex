// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::cast_keeping_nullability;
use crate::arrays::widen::compute::ptype_holding_both_values;
use crate::arrays::widen::compute::ptype_holding_values_and_scalar;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::zip::ZipKernel;

/// Zips the values of a [`Widen`] with another `Widen` or a constant at the type that holds both.
/// The adaptor normalizes either branch to `array` and adjusts the mask accordingly.
impl ZipKernel for Widen {
    fn zip(
        array: ArrayView<'_, Self>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (if_false_values, ptype) = if let Some(other) = if_false.as_opt::<Widen>() {
            let Some(ptype) = ptype_holding_both_values(array, other) else {
                return Ok(None);
            };
            (other.values().clone(), ptype)
        } else if let Some(constant) = if_false.as_constant() {
            let Some(ptype) = ptype_holding_values_and_scalar(array, constant)? else {
                return Ok(None);
            };
            (if_false.clone(), ptype)
        } else {
            return Ok(None);
        };

        let if_true_values = cast_keeping_nullability(array.values(), ptype)?;
        let if_false_values = cast_keeping_nullability(&if_false_values, ptype)?;

        let zipped = mask.zip(if_true_values, if_false_values)?;
        let dtype = array.dtype().with_nullability(zipped.dtype().nullability());

        wrap_values(zipped, dtype).map(Some)
    }
}
