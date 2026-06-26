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
    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let iter = (0..array.len()).map(|i| mask.value(i).then(|| array.bytes_at(i)));
    Ok(if strict {
        iter.is_strict_sorted()
    } else {
        iter.is_sorted()
    })
}
