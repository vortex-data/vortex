// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod filter;
pub(crate) mod invert;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;
pub(crate) mod min_max;
pub(crate) mod take;

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::RunEndBool;
    use crate::RunEndBoolArray;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn alternating() -> RunEndBoolArray {
        let mut ctx = SESSION.create_execution_ctx();
        RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap()
    }

    fn single_run() -> RunEndBoolArray {
        let mut ctx = SESSION.create_execution_ctx();
        RunEndBool::try_new(
            buffer![6u32].into_array(),
            false,
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap()
    }

    fn nullable() -> RunEndBoolArray {
        let mut ctx = SESSION.create_execution_ctx();
        RunEndBool::try_new(
            buffer![2u32, 4].into_array(),
            true,
            Validity::from(BitBuffer::from(vec![true, false, false, true])),
            &mut ctx,
        )
        .unwrap()
    }

    #[rstest]
    #[case::alternating(alternating())]
    #[case::single_run(single_run())]
    #[case::nullable(nullable())]
    fn test_runend_bool_consistency(#[case] array: RunEndBoolArray) {
        test_array_consistency(&array.into_array());
    }
}
