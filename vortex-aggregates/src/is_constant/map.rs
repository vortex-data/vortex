// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListView;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_error::VortexResult;

use super::list::check_listview_constant;

pub(super) fn check_map_constant(map: &MapArray, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    check_listview_constant(&map.entries().as_::<ListView>().into_owned(), ctx)
}
