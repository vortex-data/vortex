// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::VarBinViewArray;

pub(super) fn check_varbinview_identical(
    lhs: &VarBinViewArray,
    rhs: &VarBinViewArray,
) -> VortexResult<bool> {
    if lhs.views().len() != rhs.views().len() {
        return Ok(false);
    }

    for (lhs_view, rhs_view) in lhs.views().iter().zip(rhs.views().iter()) {
        if lhs_view.is_inlined() != rhs_view.is_inlined() {
            return Ok(false);
        }

        if lhs_view.is_inlined() {
            if lhs_view.as_inlined() != rhs_view.as_inlined() {
                return Ok(false);
            }
            continue;
        }

        let lhs_bytes = &lhs
            .buffer(lhs_view.as_view().buffer_index as usize)
            .as_slice()[lhs_view.as_view().as_range()];
        let rhs_bytes = &rhs
            .buffer(rhs_view.as_view().buffer_index as usize)
            .as_slice()[rhs_view.as_view().as_range()];
        if lhs_bytes != rhs_bytes {
            return Ok(false);
        }
    }

    Ok(true)
}
