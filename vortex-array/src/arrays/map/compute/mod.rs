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
use crate::arrays::MapArray;
use crate::dtype::MapDType;

fn rebuild_map_from_array(map_dtype: MapDType, entries: ArrayRef) -> VortexResult<ArrayRef> {
    let map_entries = entries.try_downcast::<ListView>().map_err(|arr| {
        vortex_err!(
            "Map entries operation expected vortex.listview/ListView, got {}",
            arr.encoding_id()
        )
    })?;

    MapArray::try_new(map_dtype, map_entries).map(IntoArray::into_array)
}
