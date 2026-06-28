// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;
use crate::compress::value_at_index;

impl TakeExecute for RunEndBool {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "index cast to usize inside macro"
    )]
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let primitive_indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let len = array.as_ref().len();
        let start = array.start();

        let mut bits = BitBufferMut::with_capacity(primitive_indices.len());
        match_each_integer_ptype!(primitive_indices.ptype(), |P| {
            for idx in primitive_indices.as_slice::<P>().iter().copied() {
                let usize_idx = idx as usize;
                if usize_idx >= len {
                    vortex_bail!(OutOfBounds: usize_idx, 0, len);
                }
                let run_index = array.find_physical_index(usize_idx)?;
                bits.append(value_at_index(run_index, start));
            }
        });

        let validity = array
            .bool_validity()
            .take(&primitive_indices.into_array())?;
        Ok(Some(BoolArray::new(bits.freeze(), validity).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEndBool;
    use crate::RunEndBoolArray;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn ree_array() -> RunEndBoolArray {
        let mut ctx = SESSION.create_execution_ctx();
        // [T,T,F,F,F,T,T,T,T,T]
        RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap()
    }

    #[test]
    fn ree_take() -> VortexResult<()> {
        let taken = ree_array().take(buffer![0u32, 2, 5, 9].into_array())?;
        let expected = BoolArray::from(BitBuffer::from(vec![true, false, true, true]));
        assert_arrays_eq!(taken, expected);
        Ok(())
    }

    #[rstest]
    #[case(ree_array())]
    #[case({
        let mut ctx = SESSION.create_execution_ctx();
        RunEndBool::try_new(
            buffer![1u32, 3, 4].into_array(),
            false,
            Validity::from(BitBuffer::from(vec![true, false, false, true])),
            &mut ctx,
        ).unwrap()
    })]
    fn test_take_conformance_runend_bool(#[case] array: RunEndBoolArray) {
        test_take_conformance(&array.into_array());
    }
}
