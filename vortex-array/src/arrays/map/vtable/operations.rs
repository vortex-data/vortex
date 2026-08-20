// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Map;
use crate::arrays::StructArray;
use crate::arrays::map::MapArrayExt;
use crate::arrays::struct_::StructArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<Map> for Map {
    fn scalar_at(
        array: ArrayView<'_, Map>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let entries = array.entries_at(index)?.execute::<StructArray>(ctx)?;
        let keys = entries.unmasked_field(0);
        let values = entries.unmasked_field(1);
        let pairs = (0..entries.len())
            .map(|entry_index| {
                Ok((
                    keys.execute_scalar(entry_index, ctx)?,
                    values.execute_scalar(entry_index, ctx)?,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Scalar::try_map(array.dtype().clone(), pairs)
    }
}
