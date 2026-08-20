// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::list::check_listview_constant;
use crate::ExecutionCtx;
use crate::arrays::ListView;
use crate::arrays::MapArray;
use crate::arrays::map::MapArraySlotsExt;

pub(super) fn check_map_constant(map: &MapArray, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    check_listview_constant(&map.entries().as_::<ListView>().into_owned(), ctx)
}
