// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::IsSortedIteratorExt;
use crate::ExecutionCtx;
use crate::arrays::VarBinViewArray;

pub(super) fn check_varbinview_sorted(
    array: &VarBinViewArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let mask = array
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    // Walk the views directly, borrowing each value (inlined bytes or a slice of a data
    // buffer) rather than materializing an owned `ByteBuffer` per element.
    let views = array.views();
    let buffers = array
        .data_buffers()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();
    let iter = views.iter().zip(mask.iter()).map(|(view, valid)| {
        valid.then(|| {
            if view.is_inlined() {
                view.as_inlined().value()
            } else {
                let view_ref = view.as_view();
                &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
            }
        })
    });
    Ok(if strict {
        iter.is_strict_sorted()
    } else {
        iter.is_sorted()
    })
}
