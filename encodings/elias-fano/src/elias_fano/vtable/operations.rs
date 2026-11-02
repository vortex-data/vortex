// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use super::EliasFano;

impl OperationsVTable<EliasFano> for EliasFano {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, EliasFano>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Naive for now: decode the whole array.
        array
            .array()
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_array()
            .execute_scalar(index, ctx)
    }
}
