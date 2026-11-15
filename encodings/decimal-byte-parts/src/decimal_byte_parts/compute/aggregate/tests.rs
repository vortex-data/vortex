// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kernel selection, decimal partials, and fallback equivalence.

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::first::First;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::WidenArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::DecimalBytePartsAggregateKernel;
use crate::DecimalByteParts;
use crate::decimal_byte_parts::testing::i128_parts;
use crate::decimal_byte_parts::testing::i256_parts;

/// Checks direct dispatch before the public accumulator, so cached results cannot hide a decline.
#[track_caller]
fn check(array: &ArrayRef, aggregate: AggregateFnRef, served: bool) -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let kernel = session
        .aggregate_fns()
        .find_aggregate_kernel(array.encoding_id(), aggregate.id())
        .expect("initialize registers every aggregate exercised by check");
    let partial = kernel.aggregate(&aggregate, array, &mut ctx)?;
    assert_eq!(partial.is_some(), served, "{aggregate}");

    let decoded = array
        .clone()
        .execute::<DecimalArray>(&mut ctx)?
        .into_array();
    let mut reference = aggregate.accumulator(decoded.dtype())?;
    reference.accumulate(&decoded, &mut ctx)?;
    let expected = reference.finish()?;
    if let Some(partial) = partial {
        assert_eq!(
            partial.dtype(),
            &aggregate
                .state_dtype(array.dtype())
                .expect("decimal state dtype")
        );
        let mut direct = aggregate.accumulator(array.dtype())?;
        direct.combine_partials(&partial)?;
        assert_eq!(direct.finish()?, expected, "direct {aggregate}");
    }

    let mut actual = aggregate.accumulator(array.dtype())?;
    actual.accumulate(array, &mut ctx)?;
    assert_eq!(actual.finish()?, expected, "dispatched {aggregate}");

    Ok(())
}

fn aggregates() -> Vec<AggregateFnRef> {
    let options = NumericalAggregateOpts::default();
    vec![
        MinMax.bind(options),
        Min.bind(options),
        Max.bind(options),
        IsSorted.bind(IsSortedOptions { strict: false }),
        IsSorted.bind(IsSortedOptions { strict: true }),
        Sum.bind(options),
    ]
}

#[rstest]
#[case::ascending(vec![-9, -7, -5, -3, -1, 0, 2, 4, 6, 8], Validity::NonNullable)]
#[case::duplicate(vec![-9, -7, -5, -3, -1, 0, 2, 4, 6, 6], Validity::AllValid)]
#[case::unsorted(vec![8, -7, -5, -3, -1, 0, 2, 4, 6, -9], Validity::NonNullable)]
#[case::partially_valid(vec![-9, -7, -5, -3, -1, 0, 2, 4, 6, 8], Validity::from_iter([false, true, true, true, true, true, true, true, true, true]))]
#[case::all_null(vec![0; 10], Validity::AllInvalid)]
#[case::empty(vec![], Validity::NonNullable)]
fn test_aggregates(
    #[case] values: Vec<i64>,
    #[case] validity: Validity,
    #[values(0, 2)] scale: i8,
    #[values(PType::I8, PType::I16, PType::I32, PType::I64)] ptype: PType,
) -> VortexResult<()> {
    let msp = PrimitiveArray::new(values, validity).into_array();
    let nullability = msp.dtype().nullability();
    let msp = msp.cast(DType::Primitive(ptype, nullability))?;
    // The MSP of precision 19 is logically `i64`, so narrower storage is an integer `Widen`.
    let msp = if ptype == PType::I64 {
        msp
    } else {
        WidenArray::try_new(msp, DType::Primitive(PType::I64, nullability))?.into_array()
    };
    let array = DecimalByteParts::try_new(msp, DecimalDType::new(19, scale))?.into_array();
    for aggregate in aggregates() {
        check(&array, aggregate, true)?;
    }

    Ok(())
}

#[rstest]
#[case::dict(false)]
#[case::widen(true)]
fn test_encoded_msp(#[case] widen: bool, #[values(0, 2)] scale: i8) -> VortexResult<()> {
    let msp = if widen {
        WidenArray::try_new(
            buffer![-9i8, 0, 7].into_array(),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .into_array()
    } else {
        DictArray::try_new(
            buffer![0u8, 1, 2, 2].into_array(),
            buffer![-9i64, 0, 7].into_array(),
        )?
        .into_array()
    };
    let array = DecimalByteParts::try_new(msp, DecimalDType::new(19, scale))?.into_array();
    for aggregate in aggregates() {
        check(&array, aggregate, true)?;
    }

    Ok(())
}

#[rstest]
#[case::i128(i128_parts(vec![1 << 70, -(1 << 70)], Validity::NonNullable).into_array())]
#[case::i256(i256_parts(vec![i256::from_parts(0, 1), i256::from_parts(0, -1)], Validity::NonNullable).into_array())]
fn test_lower_parts_decline(#[case] array: ArrayRef) -> VortexResult<()> {
    for aggregate in aggregates() {
        check(&array, aggregate, false)?;
    }

    Ok(())
}

#[rstest]
#[case::positive(vec![i64::MAX, 1])]
#[case::negative(vec![i64::MIN, -1])]
#[case::cancellation(vec![i64::MAX, 1, -1])]
fn test_integer_sum_overflow(
    #[case] values: Vec<i64>,
    #[values(0, 2)] scale: i8,
) -> VortexResult<()> {
    let array = DecimalByteParts::try_new(
        PrimitiveArray::from_iter(values).into_array(),
        DecimalDType::new(19, scale),
    )?
    .into_array();
    check(&array, Sum.bind(NumericalAggregateOpts::default()), false)
}

#[rstest]
#[case::at_bound(999_999_999_999_999i64)]
#[case::outside_bound(1_000_000_000_000_000i64)]
fn test_sum_precision(#[case] total: i64, #[values(0, 2)] scale: i8) -> VortexResult<()> {
    // Construction permits stored values outside input precision. Sum checks its result against
    // the precision 15 of its state dtype. Precision 5 has a logical `i32` MSP, so the batch
    // reaches the total with copies of `i32::MAX` plus a remainder, and the `i64` sum never
    // overflows.
    let max = i64::from(i32::MAX);
    let remainder = i32::try_from(total % max)?;
    let count = usize::try_from(total / max)?;
    let values = std::iter::repeat_n(i32::MAX, count)
        .chain([remainder])
        .collect::<Buffer<i32>>();
    let array =
        DecimalByteParts::try_new(values.into_array(), DecimalDType::new(5, scale))?.into_array();
    check(&array, Sum.bind(NumericalAggregateOpts::default()), true)
}

#[rstest]
#[case::min(-100i8)]
#[case::max(100i8)]
fn test_extrema_precision(#[case] value: i8) -> VortexResult<()> {
    let session = VortexSession::empty();
    let mut ctx = session.create_execution_ctx();
    let array =
        DecimalByteParts::try_new(buffer![0i8, value].into_array(), DecimalDType::new(2, 2))?
            .into_array();
    let decoded = array
        .clone()
        .execute::<DecimalArray>(&mut ctx)?
        .into_array();
    let options = NumericalAggregateOpts::default();
    for aggregate in [MinMax.bind(options), Min.bind(options), Max.bind(options)] {
        let error = DecimalBytePartsAggregateKernel
            .aggregate(&aggregate, &array, &mut ctx)
            .unwrap_err();
        assert!(error.to_string().contains("does not fit in precision"));
        let mut reference = aggregate.accumulator(decoded.dtype())?;
        assert!(reference.accumulate(&decoded, &mut ctx).is_err());
    }

    Ok(())
}

#[rstest]
#[case::sum(Sum.bind(NumericalAggregateOpts { skip_nans: false }))]
#[case::unsupported(First.bind(EmptyOptions))]
fn test_unsupported_decline(#[case] aggregate: AggregateFnRef) -> VortexResult<()> {
    let session = VortexSession::empty();
    let mut ctx = session.create_execution_ctx();
    let array = DecimalByteParts::try_new(buffer![1i16, -2].into_array(), DecimalDType::new(3, 2))?
        .into_array();
    assert!(
        DecimalBytePartsAggregateKernel
            .aggregate(&aggregate, &array, &mut ctx)?
            .is_none()
    );
    let decoded = array
        .clone()
        .execute::<DecimalArray>(&mut ctx)?
        .into_array();
    crate::initialize(&session);
    let mut actual = aggregate.accumulator(array.dtype())?;
    let mut expected = aggregate.accumulator(decoded.dtype())?;
    actual.accumulate(&array, &mut ctx)?;
    expected.accumulate(&decoded, &mut ctx)?;
    assert_eq!(actual.finish()?, expected.finish()?);

    Ok(())
}

#[rstest]
#[case::null_payload(vec![vec![-9, 100, 9]], vec![Validity::from_iter([true, false, true])])]
#[case::ascending(vec![vec![-9, -1], vec![0, 9]], vec![Validity::NonNullable; 2])]
#[case::equal_boundary(vec![vec![-9, 0], vec![0, 9]], vec![Validity::NonNullable; 2])]
#[case::descending_boundary(vec![vec![0, 9], vec![-9, -1]], vec![Validity::NonNullable; 2])]
fn test_aggregate_batches(
    #[case] batches: Vec<Vec<i8>>,
    #[case] validities: Vec<Validity>,
    #[values(0, 2)] scale: i8,
) -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    for aggregate in aggregates() {
        let dtype = DType::Decimal(DecimalDType::new(2, scale), validities[0].nullability());
        let mut actual = aggregate.accumulator(&dtype)?;
        let mut expected = aggregate.accumulator(&dtype)?;
        for (values, validity) in batches.iter().zip(&validities) {
            let array = DecimalByteParts::try_new(
                PrimitiveArray::new(values.clone(), validity.clone()).into_array(),
                DecimalDType::new(2, scale),
            )?
            .into_array();
            check(&array, aggregate.clone(), true)?;
            let decoded = array
                .clone()
                .execute::<DecimalArray>(&mut ctx)?
                .into_array();
            actual.accumulate(&array, &mut ctx)?;
            expected.accumulate(&decoded, &mut ctx)?;
        }
        assert_eq!(actual.finish()?, expected.finish()?, "{aggregate}");
    }

    Ok(())
}
