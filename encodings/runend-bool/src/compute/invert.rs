// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::scalar_fn::fns::not::NotKernel;
use vortex_error::VortexResult;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;

impl NotKernel for RunEndBool {
    fn invert(
        array: ArrayView<'_, RunEndBool>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Inverting a boolean run-end array negates the value of every run, which is equivalent to
        // negating the `start` flag. Ends and validity are unchanged.
        // SAFETY: ends and offset are copied unchanged from a valid array.
        let inverted = unsafe {
            RunEndBool::new_unchecked(
                array.ends().clone(),
                !array.start(),
                array.offset(),
                array.as_ref().len(),
                array.bool_validity(),
            )
        };
        Ok(Some(inverted.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
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
    fn invert_runend_bool() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // [T,T,F,F,F,T,T,T,T,T]
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?;
        let inverted = arr.into_array().not()?;
        let expected = BoolArray::from(BitBuffer::from(vec![
            false, false, true, true, true, false, false, false, false, false,
        ]));
        assert_arrays_eq!(inverted, expected);
        Ok(())
    }
}
