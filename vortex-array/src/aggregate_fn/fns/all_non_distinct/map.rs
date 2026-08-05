// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::list::check_list_identical;
use crate::ExecutionCtx;
use crate::arrays::ListView;
use crate::arrays::MapArray;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::MapArraySlotsExt;

pub(super) fn check_map_identical(
    lhs: &MapArray,
    rhs: &MapArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if lhs.map_dtype() != rhs.map_dtype() {
        return Ok(false);
    }

    check_list_identical(
        &lhs.entries().as_::<ListView>().into_owned(),
        &rhs.entries().as_::<ListView>().into_owned(),
        ctx,
    )
}
