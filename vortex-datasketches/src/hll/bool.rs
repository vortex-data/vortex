// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::HllPartial;

pub(super) fn accumulate_bool(
    partial: &mut HllPartial,
    array: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let values = array.to_bit_buffer();
    match array.validity()?.execute_mask(array.as_ref().len(), ctx)? {
        Mask::AllTrue(_) => {
            for value in values.iter() {
                partial.update_value(value);
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(validity) => {
            for (value, valid) in values.iter().zip(validity.bit_buffer().iter()) {
                if valid {
                    partial.update_value(value);
                }
            }
        }
    }
    Ok(())
}
