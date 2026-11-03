// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Map;
use crate::arrays::map::MapArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<Map> for Map {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Map>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // The map scalar holds the row's entries as they are, rather than a scalar per entry.
        let entries = array.entries_at(index, ctx)?;
        Scalar::map_from_entries_array(array.dtype().clone(), entries, ctx)
    }
}
