// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtensionStoragePushdown`] over the order-based aggregates.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::ExtensionStoragePushdown;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::Time;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::is_sorted::IsSorted;
use crate::is_sorted::IsSortedOptions;
use crate::min_max::MinMax;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[test]
fn pushes_down_into_storage() -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let aggregate = ExtensionStoragePushdown::new(MinMax, Timestamp);
    session.aggregate_fns().register(aggregate.clone());
    assert!(
        session
            .aggregate_fns()
            .find_plugin(&aggregate.id())
            .is_some()
    );

    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let storage =
        PrimitiveArray::from_option_iter([Some(3_000i64), None, Some(1_000)]).into_array();
    let array = ExtensionArray::new(ext_dtype.clone(), storage.clone()).into_array();

    let mut ctx = session.create_execution_ctx();
    let mut acc = Accumulator::try_new(
        aggregate,
        NumericalAggregateOpts::default(),
        DType::Extension(ext_dtype),
    )?;
    acc.accumulate(&array, &mut ctx)?;

    let mut expected = Accumulator::try_new(
        MinMax,
        NumericalAggregateOpts::default(),
        storage.dtype().clone(),
    )?;
    expected.accumulate(&storage, &mut ctx)?;

    assert_eq!(acc.finish()?, expected.finish()?);
    Ok(())
}

#[test]
fn combines_flushed_partials() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let batch = |values: [Option<i64>; 2]| {
        ExtensionArray::new(
            ext_dtype.clone(),
            PrimitiveArray::from_option_iter(values).into_array(),
        )
        .into_array()
    };
    let accumulator = || {
        Accumulator::try_new(
            ExtensionStoragePushdown::new(MinMax, Timestamp),
            NumericalAggregateOpts::default(),
            DType::Extension(ext_dtype.clone()),
        )
    };

    let mut first = accumulator()?;
    first.accumulate(&batch([Some(3_000), None]), &mut ctx)?;
    let mut second = accumulator()?;
    second.accumulate(&batch([Some(1_000), Some(500)]), &mut ctx)?;
    first.combine_partials(&second.flush()?)?;

    let mut expected = accumulator()?;
    expected.accumulate(&batch([Some(3_000), None]), &mut ctx)?;
    expected.accumulate(&batch([Some(1_000), Some(500)]), &mut ctx)?;

    assert_eq!(first.finish()?, expected.finish()?);
    Ok(())
}

#[test]
fn accumulates_constant_batches() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let constant = |value: i64| {
        ConstantArray::new(
            Scalar::extension_ref(
                ext_dtype.clone(),
                Scalar::primitive(value, Nullability::Nullable),
            ),
            3,
        )
        .into_array()
    };

    let mut acc = Accumulator::try_new(
        ExtensionStoragePushdown::new(MinMax, Timestamp),
        NumericalAggregateOpts::default(),
        DType::Extension(ext_dtype.clone()),
    )?;
    acc.accumulate(&constant(2_000), &mut ctx)?;
    acc.accumulate(&constant(1_000), &mut ctx)?;

    let mut expected = Accumulator::try_new(
        MinMax,
        NumericalAggregateOpts::default(),
        ext_dtype.storage_dtype().clone(),
    )?;
    expected.accumulate(
        &PrimitiveArray::from_option_iter([Some(2_000i64), Some(1_000)]).into_array(),
        &mut ctx,
    )?;

    assert_eq!(acc.finish()?, expected.finish()?);
    Ok(())
}

#[test]
fn rejects_other_extension_types() {
    let date = DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased());
    assert!(
        ExtensionStoragePushdown::new(MinMax, Timestamp)
            .return_dtype(&NumericalAggregateOpts::default(), &date)
            .is_none()
    );
}

#[test]
fn default_session_registers_temporal_order_aggregates() {
    let aggregate_fns = SESSION.aggregate_fns();
    for id in [
        ExtensionStoragePushdown::new(IsSorted, Date).id(),
        ExtensionStoragePushdown::new(IsSorted, Time).id(),
        ExtensionStoragePushdown::new(IsSorted, Timestamp).id(),
        ExtensionStoragePushdown::new(MinMax, Date).id(),
        ExtensionStoragePushdown::new(MinMax, Time).id(),
        ExtensionStoragePushdown::new(MinMax, Timestamp).id(),
    ] {
        assert!(
            aggregate_fns.find_plugin(&id).is_some(),
            "{id} is not registered"
        );
    }
}

#[rstest]
// Nulls sort first.
#[case::sorted([None, Some(1_000), Some(2_000), Some(2_000)], false, true)]
#[case::unsorted([Some(2_000), Some(1_000), None, Some(3_000)], false, false)]
#[case::strict_with_duplicates([None, Some(1_000), Some(2_000), Some(2_000)], true, false)]
fn is_sorted_over_timestamps(
    #[case] values: [Option<i64>; 4],
    #[case] strict: bool,
    #[case] expected: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let array = ExtensionArray::new(
        ext_dtype.clone(),
        PrimitiveArray::from_option_iter(values).into_array(),
    )
    .into_array();

    let mut acc = Accumulator::try_new(
        ExtensionStoragePushdown::new(IsSorted, Timestamp),
        IsSortedOptions { strict },
        DType::Extension(ext_dtype),
    )?;
    acc.accumulate(&array, &mut ctx)?;

    assert_eq!(bool::try_from(&acc.finish()?)?, expected);
    Ok(())
}

#[test]
fn min_max_over_dates_matches_storage() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ext_dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
    let storage =
        PrimitiveArray::from_option_iter([Some(20i32), None, Some(-3), Some(7)]).into_array();
    let array = ExtensionArray::new(ext_dtype.clone(), storage.clone()).into_array();

    let mut acc = Accumulator::try_new(
        ExtensionStoragePushdown::new(MinMax, Date),
        NumericalAggregateOpts::default(),
        DType::Extension(ext_dtype),
    )?;
    acc.accumulate(&array, &mut ctx)?;

    let mut expected = Accumulator::try_new(
        MinMax,
        NumericalAggregateOpts::default(),
        storage.dtype().clone(),
    )?;
    expected.accumulate(&storage, &mut ctx)?;

    assert_eq!(acc.finish()?, expected.finish()?);
    Ok(())
}
