// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::array::TurboQuant;
use crate::array::TurboQuantArray;

impl OperationsVTable<TurboQuant> for TurboQuant {
    fn scalar_at(
        array: &TurboQuantArray,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Slice to single row, decompress that one row.
        let Some(sliced) = <TurboQuant as SliceReduce>::slice(array, index..index + 1)? else {
            vortex_bail!("slice returned None for index {index}")
        };
        let decoded = sliced.execute::<FixedSizeListArray>(ctx)?;
        decoded.scalar_at(0)
    }
}
