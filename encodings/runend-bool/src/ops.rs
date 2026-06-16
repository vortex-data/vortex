// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;
use crate::compress::value_at_index;

impl OperationsVTable<RunEndBool> for RunEndBool {
    fn scalar_at(
        array: ArrayView<'_, RunEndBool>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Honor validity: a null logical element produces a null scalar.
        let validity_mask = array
            .bool_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        if !validity_mask.value(index) {
            return Ok(Scalar::null(array.as_ref().dtype().clone()));
        }

        let run_index = array.find_physical_index(index)?;
        let value = value_at_index(run_index, array.start());
        Ok(Scalar::bool(value, array.nullability()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEndBool;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn slice_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // [T,T,F,F,F,T,T,T,T,T] sliced 3..8 => [F,F,T,T,T]
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?
        .slice(3..8)?;
        assert_eq!(arr.len(), 5);
        let expected = BoolArray::from(BitBuffer::from(vec![false, false, true, true, true]));
        assert_arrays_eq!(arr, expected);
        Ok(())
    }

    #[test]
    fn double_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?
        .slice(3..8)?;
        let doubly_sliced = arr.slice(0..3)?;
        let expected = BoolArray::from(BitBuffer::from(vec![false, false, true]));
        assert_arrays_eq!(doubly_sliced, expected);
        Ok(())
    }

    #[test]
    fn slice_to_empty() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?;
        let sliced = arr.slice(arr.len()..arr.len())?;
        assert!(sliced.is_empty());
        Ok(())
    }
}
