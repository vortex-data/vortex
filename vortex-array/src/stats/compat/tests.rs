// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use rstest::rstest;
use vortex_error::VortexResult;

use super::legacy_stats_to_results;
use super::load_node_summary;
use super::read_summary;
use super::truncate_summary;
use super::write_node_summary;
use super::write_summary;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::array_session;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::flatbuffers::WriteFlatBuffer;
use crate::flatbuffers::array as fba;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::stats::AggregateResults;
use crate::stats::StatsSet;

#[test]
fn legacy_writer_fields_keep_the_same_bytes_and_types() -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let mut legacy = StatsSet::default();
    for stat in [Stat::IsConstant, Stat::IsSorted, Stat::IsStrictSorted] {
        legacy.set(stat, Precision::Exact(false.into()));
    }
    legacy.set(Stat::Min, Precision::Inexact((-1i32).into()));
    legacy.set(Stat::Max, Precision::Exact(9i32.into()));
    legacy.set(Stat::Sum, Precision::Exact(7i64.into()));
    for stat in [
        Stat::NullCount,
        Stat::NaNCount,
        Stat::UncompressedSizeInBytes,
    ] {
        legacy.set(stat, Precision::Exact(0u64.into()));
    }
    let results = legacy_stats_to_results(&dtype, &legacy)?;
    let mut old = FlatBufferBuilder::new();
    let offset = legacy.write_flatbuffer(&mut old)?;
    old.finish(offset, None);
    let mut new = FlatBufferBuilder::new();
    let offset = write_summary(&results, &dtype, &mut new)?;
    new.finish(offset, None);
    assert_eq!(old.finished_data(), new.finished_data());
    let mut node = FlatBufferBuilder::new();
    let offset = write_node_summary(&results, &dtype, &mut node)?;
    node.finish(offset, None);
    assert_eq!(old.finished_data(), node.finished_data());

    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(new.finished_data())?;
    let decoded = read_summary(&fb, &dtype, &array_session())?;
    assert_eq!(decoded.iter().count(), 9);
    for stat in Stat::all() {
        assert_eq!(
            decoded.get_result(stat.finalized_aggregate_fn()),
            results.get_result(stat.finalized_aggregate_fn())
        );
    }
    assert_eq!(
        decoded
            .get_result(Stat::Min.finalized_aggregate_fn())
            .into_inner()
            .unwrap()
            .dtype(),
        &dtype
    );
    assert_eq!(
        decoded
            .get_result(Stat::Sum.finalized_aggregate_fn())
            .into_inner()
            .unwrap()
            .dtype(),
        &DType::Primitive(PType::I64, Nullable)
    );
    Ok(())
}

#[rstest]
#[case::min_exact(Stat::Min, true)]
#[case::min_inexact(Stat::Min, false)]
#[case::max_exact(Stat::Max, true)]
#[case::max_inexact(Stat::Max, false)]
#[case::sum_exact(Stat::Sum, true)]
fn nullable_null_is_present_and_missing_is_absent(
    #[case] stat: Stat,
    #[case] exact: bool,
) -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I64, Nullable);
    let scalar = Scalar::null(dtype.clone());
    let value = if exact {
        Precision::Exact(scalar)
    } else {
        Precision::Inexact(scalar)
    };
    let results = AggregateResults::from_validated(vec![(
        stat.finalized_aggregate_fn().clone(),
        value.clone(),
    )]);
    let mut fbb = FlatBufferBuilder::new();
    let offset = write_summary(&results, &dtype, &mut fbb)?;
    fbb.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
    let session = array_session();
    let decoded = read_summary(&fb, &dtype, &session)?;
    assert_eq!(decoded.get_result(stat.finalized_aggregate_fn()), value);
    let mut node = FlatBufferBuilder::new();
    let offset = write_node_summary(&results, &dtype, &mut node)?;
    node.finish(offset, None);
    assert_eq!(fbb.finished_data(), node.finished_data());
    assert!(
        StatsSet::from_flatbuffer(&fb, &dtype, &session)?
            .get(stat)
            .is_absent()
    );

    let mut missing = FlatBufferBuilder::new();
    let offset = fba::ArrayStats::create(&mut missing, &fba::ArrayStatsArgs::default());
    missing.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(missing.finished_data())?;
    assert!(
        read_summary(&fb, &dtype, &session)?
            .get_result(stat.finalized_aggregate_fn())
            .is_absent()
    );
    Ok(())
}

#[test]
fn current_nullable_min_writes_a_historical_non_nullable_field() -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let results = AggregateResults::try_new(
        &dtype,
        [(
            min.clone(),
            Precision::Exact(Scalar::primitive(4i32, Nullable)),
        )],
    )?;
    let mut fbb = FlatBufferBuilder::new();
    let offset = write_summary(&results, &dtype, &mut fbb)?;
    fbb.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
    assert_eq!(
        read_summary(&fb, &dtype, &array_session())?.get_result(&min),
        Precision::Exact(Scalar::primitive(4i32, NonNullable))
    );

    let null = AggregateResults::try_new(
        &dtype,
        [(
            min.clone(),
            Precision::Exact(Scalar::null(dtype.as_nullable())),
        )],
    )?;
    assert!(write_summary(&null, &dtype, &mut FlatBufferBuilder::new()).is_err());
    let float = AggregateResults::try_new(
        &DType::from(PType::F32),
        [(min, Precision::Exact(Scalar::primitive(4f32, Nullable)))],
    )?;
    assert!(write_summary(&float, &dtype, &mut FlatBufferBuilder::new()).is_err());
    assert!(write_node_summary(&float, &dtype, &mut FlatBufferBuilder::new()).is_err());
    Ok(())
}

#[rstest]
#[case::min(Stat::Min)]
#[case::max(Stat::Max)]
fn non_nullable_null_and_unknown_precision_are_rejected(#[case] stat: Stat) -> VortexResult<()> {
    for (value, precision) in [
        (None, fba::Precision::Exact),
        (Some(1i32.into()), fba::Precision(99)),
    ] {
        let mut fbb = FlatBufferBuilder::new();
        let bytes = fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(value.as_ref()));
        let mut args = fba::ArrayStatsArgs::default();
        match stat {
            Stat::Min => {
                args.min = Some(bytes);
                args.min_precision = precision;
            }
            _ => {
                args.max = Some(bytes);
                args.max_precision = precision;
            }
        }
        let offset = fba::ArrayStats::create(&mut fbb, &args);
        fbb.finish(offset, None);
        let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
        assert!(read_summary(&fb, &PType::I32.into(), &array_session()).is_err());
    }
    Ok(())
}

#[test]
fn historical_nested_and_extension_fields_do_not_require_current_kernels() -> VortexResult<()> {
    let scalar = Scalar::list(DType::from(PType::I32), vec![1i32.into()], NonNullable);
    let mut legacy = StatsSet::default();
    legacy.set(Stat::Min, Precision::Exact(scalar.value().unwrap().clone()));
    let results = legacy_stats_to_results(scalar.dtype(), &legacy)?;
    let mut fbb = FlatBufferBuilder::new();
    let offset = write_summary(&results, scalar.dtype(), &mut fbb)?;
    fbb.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
    assert_eq!(
        read_summary(&fb, scalar.dtype(), &array_session())?
            .get_result(Stat::Min.finalized_aggregate_fn()),
        Precision::Exact(scalar)
    );

    let dtype = DType::Extension(Timestamp::new(TimeUnit::Milliseconds, NonNullable).erased());
    let mut legacy = StatsSet::default();
    legacy.set(Stat::Sum, Precision::Exact(12i64.into()));
    let results = legacy_stats_to_results(&dtype, &legacy)?;
    let mut fbb = FlatBufferBuilder::new();
    let offset = write_summary(&results, &dtype, &mut fbb)?;
    fbb.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
    assert_eq!(
        read_summary(&fb, &dtype, &array_session())?.get_result(Stat::Sum.finalized_aggregate_fn()),
        Precision::Exact(Scalar::primitive(12i64, Nullable))
    );
    Ok(())
}

#[test]
fn unsupported_options_and_inexact_non_extrema_are_rejected() -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let min = Min.bind(NumericalAggregateOpts::include_nans());
    let results = AggregateResults::try_new(
        &dtype,
        [(min, Precision::Exact(Scalar::primitive(4i32, Nullable)))],
    )?;
    assert!(write_summary(&results, &dtype, &mut FlatBufferBuilder::new()).is_err());
    let results = AggregateResults::from_validated(vec![(
        Stat::NullCount.finalized_aggregate_fn().clone(),
        Precision::Inexact(0u64.into()),
    )]);
    assert!(write_summary(&results, &dtype, &mut FlatBufferBuilder::new()).is_err());
    Ok(())
}

#[test]
fn node_projection_omits_unrepresentable_results() -> VortexResult<()> {
    let dtype = DType::from(PType::I64);
    let results = AggregateResults::from_validated(vec![
        (
            Stat::Min.finalized_aggregate_fn().clone(),
            Precision::Exact(Scalar::null(dtype.as_nullable())),
        ),
        (
            Stat::Max.finalized_aggregate_fn().clone(),
            Precision::Exact(Scalar::null(dtype.as_nullable())),
        ),
        (
            Stat::Sum.finalized_aggregate_fn().clone(),
            Precision::Exact(Scalar::null(dtype.as_nullable())),
        ),
        (
            Stat::NullCount.finalized_aggregate_fn().clone(),
            Precision::Inexact(0u64.into()),
        ),
        (
            Min.bind(NumericalAggregateOpts::include_nans()),
            Precision::Exact(Scalar::primitive(4i64, Nullable)),
        ),
        (
            SumV2.bind(NumericalAggregateOpts::skip_nans()),
            Precision::Exact(Scalar::primitive(4i64, Nullable)),
        ),
    ]);
    let mut fbb = FlatBufferBuilder::new();
    let offset = write_node_summary(&results, &dtype, &mut fbb)?;
    fbb.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
    let decoded = read_summary(&fb, &dtype, &array_session())?;
    assert_eq!(decoded.iter().count(), 1);
    assert_eq!(
        decoded.get_result(Stat::Sum.finalized_aggregate_fn()),
        Precision::Exact(Scalar::null(dtype.as_nullable()))
    );
    assert!(write_summary(&results, &dtype, &mut FlatBufferBuilder::new()).is_err());
    assert_eq!(results.iter().count(), 6);
    Ok(())
}

#[test]
fn node_load_normalizes_current_root_nullability() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter([1i64, 2]).into_array();
    let results = AggregateResults::from_validated(vec![
        (
            Stat::Min.finalized_aggregate_fn().clone(),
            Precision::Inexact(Scalar::primitive(0i64, NonNullable)),
        ),
        (
            Stat::Sum.finalized_aggregate_fn().clone(),
            Precision::Exact(Scalar::null(array.dtype().as_nullable())),
        ),
    ]);
    let mut fbb = FlatBufferBuilder::new();
    let offset = write_node_summary(&results, array.dtype(), &mut fbb)?;
    fbb.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
    load_node_summary(&array, &fb, &array_session())?;
    assert_eq!(
        array
            .aggregations()
            .get_result(Stat::Min.finalized_aggregate_fn()),
        Precision::Inexact(Scalar::primitive(0i64, Nullable))
    );
    assert_eq!(
        array
            .aggregations()
            .get_result(Stat::Sum.finalized_aggregate_fn()),
        results.get_result(Stat::Sum.finalized_aggregate_fn())
    );
    Ok(())
}

#[test]
fn node_load_preserves_extension_storage_sum() -> VortexResult<()> {
    let array = ExtensionArray::try_new(
        Timestamp::new(TimeUnit::Milliseconds, NonNullable).erased(),
        PrimitiveArray::from_iter([12i64]).into_array(),
    )?
    .into_array();
    let results = AggregateResults::from_validated(vec![(
        Stat::Sum.finalized_aggregate_fn().clone(),
        Precision::Exact(Scalar::primitive(12i64, Nullable)),
    )]);
    let mut fbb = FlatBufferBuilder::new();
    let offset = write_node_summary(&results, array.dtype(), &mut fbb)?;
    fbb.finish(offset, None);
    let fb = flatbuffers::root::<fba::ArrayStats<'_>>(fbb.finished_data())?;
    load_node_summary(&array, &fb, &array_session())?;
    assert_eq!(
        array
            .aggregations()
            .get_result(Stat::Sum.finalized_aggregate_fn()),
        results.get_result(Stat::Sum.finalized_aggregate_fn())
    );
    Ok(())
}

#[rstest]
#[case::utf8(
    Scalar::utf8("123456", NonNullable),
    Scalar::utf8("123", NonNullable),
    Scalar::utf8("124", NonNullable)
)]
#[case::binary(Scalar::binary(vec![1, 2, 3, 4], NonNullable), Scalar::binary(vec![1, 2, 3], NonNullable), Scalar::binary(vec![1, 2, 4], NonNullable))]
fn detached_truncation_preserves_bounds(
    #[case] original: Scalar,
    #[case] lower: Scalar,
    #[case] upper: Scalar,
) -> VortexResult<()> {
    let results = AggregateResults::from_validated(vec![
        (
            Stat::Min.finalized_aggregate_fn().clone(),
            Precision::Exact(original.clone()),
        ),
        (
            Stat::Max.finalized_aggregate_fn().clone(),
            Precision::Inexact(original.clone()),
        ),
    ]);
    let truncated = truncate_summary(&results, 3)?;
    assert_eq!(
        truncated.get_result(Stat::Min.finalized_aggregate_fn()),
        Precision::Inexact(lower)
    );
    assert_eq!(
        truncated.get_result(Stat::Max.finalized_aggregate_fn()),
        Precision::Inexact(upper)
    );
    let unchanged = truncate_summary(&results, 8)?;
    for (aggregate, value) in results.iter() {
        assert_eq!(unchanged.get_result(aggregate), *value);
    }
    assert_eq!(
        results.get_result(Stat::Min.finalized_aggregate_fn()),
        Precision::Exact(original.clone())
    );
    assert_eq!(
        results.get_result(Stat::Max.finalized_aggregate_fn()),
        Precision::Inexact(original)
    );
    Ok(())
}

#[test]
fn detached_truncation_omits_unbounded_max_and_retains_nulls() -> VortexResult<()> {
    let results = AggregateResults::from_validated(vec![(
        Stat::Max.finalized_aggregate_fn().clone(),
        Precision::Exact(Scalar::binary(vec![255, 255], NonNullable)),
    )]);
    assert!(
        truncate_summary(&results, 1)?
            .get_result(Stat::Max.finalized_aggregate_fn())
            .is_absent()
    );
    let null = Precision::Inexact(Scalar::null(DType::Utf8(Nullable)));
    let results = AggregateResults::from_validated(vec![(
        Stat::Min.finalized_aggregate_fn().clone(),
        null.clone(),
    )]);
    assert_eq!(
        truncate_summary(&results, 1)?.get_result(Stat::Min.finalized_aggregate_fn()),
        null
    );
    Ok(())
}
