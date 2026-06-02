// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::bool::BoolArrayExt;

pub(super) fn check_bool_identical<L, R>(lhs: &L, rhs: &R) -> VortexResult<bool>
where
    L: BoolArrayExt,
    R: BoolArrayExt,
{
    Ok(lhs.bit_buffer_view() == rhs.bit_buffer_view())
}
