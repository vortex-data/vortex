// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod fill_null;
pub(crate) mod filter;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;
pub(crate) mod min_max;
pub(crate) mod sum;
pub(crate) mod take;
pub(crate) mod take_from;

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;

/// Returns the active physical run range covering `[offset..offset + len)` for a non-empty
/// [`RunEnd`] array.
pub(super) fn active_physical_range(
    run_end: ArrayView<'_, RunEnd>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Range<usize>> {
    debug_assert!(!run_end.is_empty());

    let ends = run_end.ends();
    let num_runs = ends.len();
    if num_runs == 1 {
        return Ok(0..1);
    }

    if run_end.offset() == 0 {
        let last_end = usize::try_from(&ends.execute_scalar(num_runs - 1, ctx)?)?;
        if last_end == run_end.len() {
            let first_end = ends.execute_scalar(0, ctx)?;
            if !first_end.is_null() && usize::try_from(&first_end)? > 0 {
                return Ok(0..num_runs);
            }
        }
    }

    run_end.find_physical_range(0..run_end.len(), ctx)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::filter::test_filter_conformance;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn encode_runend(values: ArrayRef) -> ArrayRef {
        RunEnd::encode(values, &mut SESSION.create_execution_ctx())
            .unwrap()
            .into_array()
    }

    #[rstest]
    // Simple run-end arrays
    #[case::runend_i32(encode_runend(buffer![1i32, 1, 1, 2, 2, 3, 3, 3, 3].into_array()))]
    #[case::runend_single_run(encode_runend(buffer![5i32, 5, 5, 5, 5].into_array()))]
    #[case::runend_alternating(encode_runend(buffer![1i32, 2, 1, 2, 1, 2].into_array()))]
    // Different types
    #[case::runend_u64(encode_runend(buffer![100u64, 100, 200, 200, 200].into_array()))]
    // Edge cases
    #[case::runend_single(encode_runend(buffer![42i32].into_array()))]
    #[case::runend_nullable(encode_runend(
        PrimitiveArray::from_option_iter([
            Some(1i32),
            Some(1),
            None,
            None,
            None,
            Some(2),
            Some(2),
            None,
            Some(3),
        ])
        .into_array()
    ))]
    #[case::runend_large(encode_runend(
        PrimitiveArray::from_iter((0..1000).map(|i| i / 10)).into_array()
    ))]
    #[case::runend_sliced(
        encode_runend(PrimitiveArray::from_iter((0..256).map(|i| i / 8)).into_array())
            .slice(13..240)
            .unwrap()
    )]
    #[case::runend_nullable_sliced(
        encode_runend(
            PrimitiveArray::from_option_iter(
                (0..200).map(|i| ((i / 10) % 3 != 1).then_some(i / 10))
            )
            .into_array()
        )
        .slice(7..185)
        .unwrap()
    )]
    fn test_runend_conformance(#[case] array: ArrayRef) {
        let mut ctx = SESSION.create_execution_ctx();
        test_array_consistency(&array, &mut ctx);
        test_filter_conformance(&array, &mut ctx);
    }
}
