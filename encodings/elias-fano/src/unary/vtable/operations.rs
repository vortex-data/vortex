// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use super::Unary;

impl OperationsVTable<Unary> for Unary {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Unary>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let value = array.value_at(index)?;
        // `validate` keeps every value within the ptype.
        Ok(match_each_unsigned_integer_ptype!(
            array.dtype().as_ptype(),
            |T| { Scalar::primitive::<T>(value.as_(), NonNullable) }
        ))
    }
}
