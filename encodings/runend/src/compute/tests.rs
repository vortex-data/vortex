// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::RunEnd;
use crate::RunEndArray;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

#[rstest]
// Simple run-end arrays
#[case::runend_i32(RunEnd::encode(
    buffer![1i32, 1, 1, 2, 2, 3, 3, 3, 3].into_array(),
    &mut SESSION.create_execution_ctx(),
).unwrap())]
#[case::runend_single_run(RunEnd::encode(
    buffer![5i32, 5, 5, 5, 5].into_array(),
    &mut SESSION.create_execution_ctx(),
).unwrap())]
#[case::runend_alternating(RunEnd::encode(
    buffer![1i32, 2, 1, 2, 1, 2].into_array(),
    &mut SESSION.create_execution_ctx(),
).unwrap())]
// Different types
#[case::runend_u64(RunEnd::encode(
    buffer![100u64, 100, 200, 200, 200].into_array(),
    &mut SESSION.create_execution_ctx(),
).unwrap())]
// Edge cases
#[case::runend_single(RunEnd::encode(
    buffer![42i32].into_array(),
    &mut SESSION.create_execution_ctx(),
).unwrap())]
#[case::runend_large(RunEnd::encode(
    PrimitiveArray::from_iter((0..1000).map(|i| i / 10)).into_array(),
    &mut SESSION.create_execution_ctx(),
).unwrap())]

fn test_runend_consistency(#[case] array: RunEndArray) {
    test_array_consistency(&array.into_array(), &mut SESSION.create_execution_ctx());
}

#[rstest]
#[case::empty(0, 0, vec![])]
#[case::first_run(0, 2, vec![Some(9), Some(9)])]
#[case::trailing_run_excluded(0, 4, vec![Some(9), Some(9), None, None])]
#[case::all_null(2, 2, vec![None, None])]
#[case::one_null(2, 1, vec![None])]
#[case::leading_run_excluded(2, 4, vec![None, None, Some(1), Some(1)])]
fn aggregates_respect_logical_range(
    #[case] offset: usize,
    #[case] length: usize,
    #[case] expected_values: Vec<Option<i32>>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = RunEnd::try_new_offset_length(
        buffer![2u32, 4, 6].into_array(),
        PrimitiveArray::from_option_iter([Some(9i32), None, Some(1)]).into_array(),
        offset,
        length,
        &mut ctx,
    )?
    .into_array();
    let expected = PrimitiveArray::from_option_iter(expected_values).into_array();
    let suffix = PrimitiveArray::from_option_iter([Some(4i32)]).into_array();
    for aggregate in [
        MinMax.bind(NumericalAggregateOpts::skip_nans()),
        IsConstant.bind(EmptyOptions),
        IsSorted.bind(IsSortedOptions { strict: false }),
        IsSorted.bind(IsSortedOptions { strict: true }),
    ] {
        let mut actual_acc = aggregate.accumulator(array.dtype())?;
        let mut expected_acc = aggregate.accumulator(expected.dtype())?;
        actual_acc.accumulate(&array, &mut ctx)?;
        if !expected.is_empty() {
            expected_acc.accumulate(&expected, &mut ctx)?;
        }
        assert_eq!(
            actual_acc.final_scalar()?,
            expected_acc.final_scalar()?,
            "{aggregate}"
        );

        // An empty batch must not poison later batches, and flags must include boundaries.
        actual_acc.accumulate(&suffix, &mut ctx)?;
        expected_acc.accumulate(&suffix, &mut ctx)?;
        assert_eq!(
            actual_acc.final_scalar()?,
            expected_acc.final_scalar()?,
            "{aggregate}"
        );
    }
    Ok(())
}
