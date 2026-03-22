// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::IsSortedIteratorExt;
use crate::accessor::ArrayAccessor;
use crate::arrays::VarBinViewArray;

pub(super) fn check_varbinview_sorted(array: &VarBinViewArray, strict: bool) -> VortexResult<bool> {
    Ok(if strict {
        array.with_iterator(|bytes_iter| bytes_iter.is_strict_sorted())
    } else {
        array.with_iterator(|bytes_iter| bytes_iter.is_sorted())
    })
}
