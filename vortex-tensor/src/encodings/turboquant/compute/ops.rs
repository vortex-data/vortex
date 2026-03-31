// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::slice::SliceReduce;
use vortex::array::scalar::Scalar;
use vortex::array::vtable::OperationsVTable;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

use crate::encodings::turboquant::array::TurboQuant;
use crate::encodings::turboquant::array::TurboQuantArray;

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
