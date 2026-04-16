// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Variant;
use crate::arrays::variant::VariantArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<Variant> for Variant {
    fn scalar_at(
        array: ArrayView<'_, Variant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.child().execute_scalar(index, ctx)
    }
}
