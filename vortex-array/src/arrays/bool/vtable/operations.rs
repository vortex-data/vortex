// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Bool;
use crate::arrays::bool::BoolArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<Bool> for Bool {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Bool>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        Ok(Scalar::bool(
            array.bit_buffer_view().value(index),
            array.dtype().nullability(),
        ))
    }

    fn scalar_at(
        array: ArrayView<'_, Bool>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::assert_arrays_eq;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_slice_hundred_elements() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = BoolArray::from_iter(iter::repeat_n(Some(true), 100));
        let sliced_arr = arr
            .into_array()
            .slice(8..16)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(sliced_arr.len(), 8);
        assert_eq!(sliced_arr.to_bit_buffer().len(), 8);
        assert_eq!(sliced_arr.to_bit_buffer().offset(), 0);
        assert_arrays_eq!(sliced_arr, BoolArray::from_iter([Some(true); 8]), &mut ctx);
    }

    #[test]
    fn test_slice() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);
        let sliced_arr = arr.into_array().slice(1..4).unwrap();

        assert_arrays_eq!(
            sliced_arr,
            BoolArray::from_iter([Some(true), None, Some(false)]),
            &mut ctx
        );
    }
}
