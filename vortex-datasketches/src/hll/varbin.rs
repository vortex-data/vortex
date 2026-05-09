// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::HllPartial;

pub(super) fn update_utf8(partial: &mut HllPartial, value: &str) {
    partial.update_value(value.as_bytes());
}

pub(super) fn update_binary(partial: &mut HllPartial, value: &[u8]) {
    partial.update_value(value);
}

pub(super) fn accumulate_varbinview(
    partial: &mut HllPartial,
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match array
        .varbinview_validity()
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllTrue(_) => {
            for idx in 0..array.len() {
                let value = array.bytes_at(idx);
                update_binary(partial, value.as_slice());
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(validity) => {
            for (idx, valid) in validity.bit_buffer().iter().enumerate() {
                if valid {
                    let value = array.bytes_at(idx);
                    update_binary(partial, value.as_slice());
                }
            }
        }
    }
    Ok(())
}
