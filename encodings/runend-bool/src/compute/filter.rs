// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;
use crate::compress::value_at_index;

impl FilterKernel for RunEndBool {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask_values = mask
            .values()
            .vortex_expect("FilterKernel precondition: mask is Mask::Values");

        let start = array.start();
        let mut bits = BitBufferMut::with_capacity(mask_values.true_count());
        for idx in mask_values.indices().iter().copied() {
            let run_index = array.find_physical_index(idx)?;
            bits.append(value_at_index(run_index, start));
        }

        let validity = filter_validity(&array.bool_validity(), mask)?;
        Ok(Some(BoolArray::new(bits.freeze(), validity).into_array()))
    }
}

fn filter_validity(validity: &Validity, mask: &Mask) -> VortexResult<Validity> {
    Ok(match validity {
        Validity::NonNullable => Validity::NonNullable,
        Validity::AllValid => Validity::AllValid,
        Validity::AllInvalid => Validity::AllInvalid,
        Validity::Array(a) => Validity::Array(a.filter(mask.clone())?),
    })
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
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::RunEndBool;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn filter_runend_bool() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // [T,T,F,F,F,T,T,T,T,T]
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?;
        let filtered = arr.filter(Mask::from_iter([
            true, false, true, false, true, false, true, false, true, false,
        ]))?;
        // keep indices 0,2,4,6,8 => [T,F,F,T,T]
        let expected = BoolArray::from(BitBuffer::from(vec![true, false, false, true, true]));
        assert_arrays_eq!(filtered, expected);
        Ok(())
    }
}
