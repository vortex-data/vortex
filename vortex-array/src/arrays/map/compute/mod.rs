// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Map array compute-kernel namespace.

mod cast;
mod filter;
mod mask;
pub(crate) mod rules;
mod slice;
mod take;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::MapArray;
use crate::dtype::MapDType;

fn rebuild_map(map_dtype: MapDType, entries: ListViewArray) -> VortexResult<ArrayRef> {
    MapArray::try_new(map_dtype, entries).map(IntoArray::into_array)
}

fn rebuild_map_from_array(map_dtype: MapDType, entries: ArrayRef) -> VortexResult<ArrayRef> {
    let entries = entries
        .as_opt::<ListView>()
        .ok_or_else(|| {
            vortex_err!(
                "Map entries operation expected vortex.listview/ListView, got {}",
                entries.encoding_id()
            )
        })?
        .into_owned();
    rebuild_map(map_dtype, entries)
}
