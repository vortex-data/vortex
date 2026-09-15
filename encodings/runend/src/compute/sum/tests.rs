// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
#[cfg(not(codspeed))]
use vortex_array::test_harness::trace::trace_op;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::RunEndSumKernel;
use super::runs::add_unsigned_run;
use super::runs::sum_range;
use crate::RunEnd;
use crate::tests::SESSION;

/// Compare registered dispatch and the direct kernel with a decoded primitive reference.
#[track_caller]
fn check_sum(array: ArrayRef, options: NumericalAggregateOpts) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decoded = array
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();

    for aggregate in [Sum.bind(options), SumV2.bind(options)] {
        let mut reference = aggregate.accumulator(array.dtype())?;
        reference.accumulate(&decoded, &mut ctx)?;
        let expected = reference.finish()?;

        let partial = RunEndSumKernel
            .aggregate(&aggregate, &array, &mut ctx)?
            .vortex_expect("The fixture has a primitive dtype supported by the run-end sum kernel");
        let mut direct = aggregate.accumulator(array.dtype())?;
        direct.combine_partials(partial)?;

        let mut dispatched = aggregate.accumulator(array.dtype())?;
        dispatched.accumulate(&array, &mut ctx)?;

        for actual in [direct.finish()?, dispatched.finish()?] {
            if expected.as_primitive().is_nan() {
                assert!(actual.as_primitive().is_nan());
            } else {
                assert_eq!(actual, expected);
            }
        }
    }

    Ok(())
}

#[rstest]
#[case::unsigned(buffer![1u64, 3, 7].into_array())]
#[case::signed(buffer![-1i32, 3, -7].into_array())]
#[case::float(buffer![1.25f64, 3.5, 7.75].into_array())]
#[case::nullable(PrimitiveArray::from_option_iter([Some(-3i32), None, Some(7)]).into_array())]
#[case::nulls(PrimitiveArray::from_option_iter([None::<i32>; 3]).into_array())]
fn sliced_sums(#[case] values: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array =
        RunEnd::try_new_offset_length(buffer![2u32, 5, 9].into_array(), values, 1, 7, &mut ctx)?
            .into_array();

    check_sum(array, NumericalAggregateOpts::default())
}

#[rstest]
#[case::whole(0..9, 24, false)]
#[case::ends_inside_run(0..1, 2, false)]
#[case::ends_at_run_boundary(0..2, 4, false)]
#[case::clipped(1..8, 17, false)]
#[case::starts_in_null_run(2..6, 5, false)]
#[case::starts_at_run_boundary(5..7, 10, false)]
#[case::only_nulls(2..5, 0, true)]
#[case::empty_inside_run(1..1, 0, true)]
#[case::empty_at_end(9..9, 0, true)]
fn range_sum_preserves_validity_representation(
    #[case] range: Range<usize>,
    #[case] expected_sum: u64,
    #[case] expected_empty: bool,
    #[values(false, true)] cached: bool,
) {
    // Exercise a validity bitmap that starts inside a byte.
    let bits = BitBuffer::from_iter([false, false, false, false, false, true, false, true]);
    let validity = Mask::from_buffer(bits.slice(5..));
    let Mask::Values(mask) = &validity else {
        unreachable!("The fixture contains both valid and null runs");
    };

    if cached {
        mask.indices();
    }

    let result = sum_range(
        &[2u32, 5, 9],
        &[2u64, u64::MAX, 5],
        &validity,
        range,
        add_unsigned_run,
    );

    assert_eq!(result, (Some(expected_sum), expected_empty));
    assert_eq!(mask.cached_indices().is_some(), cached);
}

#[cfg(not(codspeed))]
#[test]
fn all_invalid_skips_decoding() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = RunEnd::try_new(
        ConstantArray::new(8u32, 1).into_array(),
        ConstantArray::new(Scalar::null(DType::Primitive(PType::I32, Nullable)), 1).into_array(),
        &mut ctx,
    )?
    .into_array();

    for aggregate in [
        Sum.bind(NumericalAggregateOpts::default()),
        SumV2.bind(NumericalAggregateOpts::default()),
    ] {
        let traced = trace_op(|| -> VortexResult<()> {
            assert!(
                RunEndSumKernel
                    .aggregate(&aggregate, &array, &mut ctx)?
                    .is_some()
            );
            Ok(())
        })?;

        assert!(!traced.trace.to_string().contains("execute_until"));
    }

    Ok(())
}

#[rstest]
#[case::signed_overflow(
    buffer![i64::MAX, 1, -1].into_array(),
    buffer![1u64, 3, 4].into_array(),
)]
#[case::signed_underflow(
    buffer![i64::MIN, -1, 1].into_array(),
    buffer![1u64, 3, 4].into_array(),
)]
#[case::positive_cancellation(
    buffer![-i64::MAX, i64::MAX].into_array(),
    buffer![1u64, 3].into_array(),
)]
#[case::negative_cancellation(
    buffer![i64::MAX, -i64::MAX].into_array(),
    buffer![1u64, 3].into_array(),
)]
#[case::unsigned_product(buffer![u64::MAX].into_array(), buffer![2u64].into_array())]
#[case::unsigned_addition(buffer![u64::MAX, 1].into_array(), buffer![1u64, 2].into_array())]
fn integer_overflow(#[case] values: ArrayRef, #[case] ends: ArrayRef) -> VortexResult<()> {
    let array = RunEnd::try_new(ends, values, &mut SESSION.create_execution_ctx())?.into_array();

    check_sum(array, NumericalAggregateOpts::default())
}

#[rstest]
#[case::nan(buffer![f64::NAN, 1.25, 2.5].into_array())]
#[case::all_nan(buffer![f64::NAN, f64::NAN, f64::NAN].into_array())]
#[case::infinities(buffer![f64::INFINITY, f64::NEG_INFINITY, 2.5].into_array())]
fn floats(#[case] values: ArrayRef, #[values(true, false)] skip_nans: bool) -> VortexResult<()> {
    let array = RunEnd::try_new(
        buffer![2u64, 4, 7].into_array(),
        values,
        &mut SESSION.create_execution_ctx(),
    )?
    .into_array();

    check_sum(array, NumericalAggregateOpts { skip_nans })
}

#[rstest]
fn float_run_product_cancellation(#[values(1e308, -1e308)] value: f64) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = RunEnd::try_new(
        buffer![1u64, 3].into_array(),
        buffer![-value, value].into_array(),
        &mut ctx,
    )?
    .into_array();

    check_sum(array, NumericalAggregateOpts::default())
}

#[rstest]
#[case::clipped_non_finite_runs(
    buffer![2u64, 5, 8].into_array(),
    buffer![f64::NAN, 3.0, f64::INFINITY].into_array(),
    2..5,
)]
#[case::empty_children(
    PrimitiveArray::empty::<u64>(NonNullable).into_array(),
    PrimitiveArray::empty::<i32>(NonNullable).into_array(),
    0..0,
)]
#[case::empty_slice(
    buffer![2u64].into_array(),
    buffer![f64::NAN].into_array(),
    0..0,
)]
fn empty_arrays_and_clipped_runs(
    #[case] ends: ArrayRef,
    #[case] values: ArrayRef,
    #[case] range: Range<usize>,
) -> VortexResult<()> {
    let array = RunEnd::try_new_offset_length(
        ends,
        values,
        range.start,
        range.len(),
        &mut SESSION.create_execution_ctx(),
    )?
    .into_array();

    check_sum(array, NumericalAggregateOpts::include_nans())
}

#[test]
fn decimal_kernel_declines() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DecimalArray::new(
        buffer![100i64, 200],
        DecimalDType::new(10, 2),
        Validity::NonNullable,
    )
    .into_array();
    let array = RunEnd::try_new(buffer![2u64, 4].into_array(), values, &mut ctx)?.into_array();

    for aggregate in [
        Sum.bind(NumericalAggregateOpts::default()),
        SumV2.bind(NumericalAggregateOpts::default()),
    ] {
        assert!(
            RunEndSumKernel
                .aggregate(&aggregate, &array, &mut ctx)?
                .is_none()
        );
    }

    Ok(())
}
