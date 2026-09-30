// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use rstest::rstest;
use vortex_error::VortexResult;

use super::read_summary;
use super::validate_selection;
use super::write_summary;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::array_session;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::extension::datetime::Date;
use crate::extension::datetime::TimeUnit;
use crate::flatbuffers::array as fba;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::stats::AggregateResults;
use crate::stats::compat::LegacyStat;

#[test]
fn every_historical_field_keeps_its_value_and_precision() -> VortexResult<()> {
    let session = array_session();
    let dtype = DType::from(PType::F64);
    let mut builder = FlatBufferBuilder::new();
    let min = builder.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
        &(-1.0f64).into(),
    )));
    let max = builder.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
        &9.0f64.into(),
    )));
    let sum = builder.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
        &12.0f64.into(),
    )));
    let root = fba::ArrayStats::create(
        &mut builder,
        &fba::ArrayStatsArgs {
            min: Some(min),
            min_precision: fba::Precision::Inexact,
            max: Some(max),
            max_precision: fba::Precision::Exact,
            sum: Some(sum),
            is_constant: Some(false),
            is_sorted: Some(true),
            is_strict_sorted: Some(false),
            null_count: Some(2),
            nan_count: Some(1),
            uncompressed_size_in_bytes: Some(64),
        },
    );
    builder.finish_minimal(root);
    let historical = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    let results = read_summary(&historical, &dtype, &session)?;
    assert_eq!(results.iter().count(), 9);
    for (stat, expected) in [
        (
            LegacyStat::Min,
            Precision::Inexact(Scalar::primitive(-1.0f64, Nullability::Nullable)),
        ),
        (
            LegacyStat::Max,
            Precision::Exact(Scalar::primitive(9.0f64, Nullability::Nullable)),
        ),
        (
            LegacyStat::Sum,
            Precision::Exact(Scalar::primitive(12.0f64, Nullability::Nullable)),
        ),
        (LegacyStat::IsConstant, Precision::Exact(false.into())),
        (LegacyStat::IsSorted, Precision::Exact(true.into())),
        (LegacyStat::IsStrictSorted, Precision::Exact(false.into())),
        (LegacyStat::NullCount, Precision::Exact(2u64.into())),
        (LegacyStat::NaNCount, Precision::Exact(1u64.into())),
        (
            LegacyStat::UncompressedSizeInBytes,
            Precision::Exact(64u64.into()),
        ),
    ] {
        assert_eq!(results.get(&stat.finalized_fn()), expected, "{stat}");
    }

    // Read new output through the unchanged historical wire accessors.
    let mut builder = FlatBufferBuilder::new();
    let root = write_summary(&results, &dtype, &mut builder)?;
    builder.finish_minimal(root);
    let written = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    assert_eq!(written.min_precision(), fba::Precision::Inexact);
    assert_eq!(written.max_precision(), fba::Precision::Exact);
    assert_eq!(written.null_count(), Some(2));
    assert_eq!(written.nan_count(), Some(1));
    assert_eq!(written.uncompressed_size_in_bytes(), Some(64));
    assert_eq!(written.is_sorted(), Some(true));
    let reread = read_summary(&written, &dtype, &session)?;
    for (aggregate, value) in results.iter() {
        assert_eq!(&reread.get(aggregate), value);
    }
    Ok(())
}

#[test]
fn missing_fields_remain_missing() -> VortexResult<()> {
    let mut builder = FlatBufferBuilder::new();
    let root = fba::ArrayStats::create(&mut builder, &fba::ArrayStatsArgs::default());
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    let results = read_summary(&wire, &DType::from(PType::I32), &array_session())?;
    assert_eq!(results.iter().count(), 0);
    Ok(())
}

#[test]
fn incompatible_scalar_is_rejected() -> VortexResult<()> {
    let mut builder = FlatBufferBuilder::new();
    let min = builder.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
        &"invalid".into(),
    )));
    let root = fba::ArrayStats::create(
        &mut builder,
        &fba::ArrayStatsArgs {
            min: Some(min),
            min_precision: fba::Precision::Exact,
            ..Default::default()
        },
    );
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    assert!(read_summary(&wire, &DType::from(PType::I32), &array_session()).is_err());
    Ok(())
}

#[rstest]
#[case::nullable(Nullability::Nullable)]
#[case::non_nullable(Nullability::NonNullable)]
fn extension_extrema_preserve_logical_type(#[case] nullability: Nullability) -> VortexResult<()> {
    let ext = Date::new(TimeUnit::Days, nullability).erased();
    let dtype = DType::Extension(ext.clone());
    let aggregate = LegacyStat::Min.finalized_fn();
    let value = Scalar::extension_ref(ext, Scalar::primitive(3i32, nullability))
        .cast(&dtype.as_nullable())?;
    let results = AggregateResults::try_new(
        &dtype,
        [(aggregate.clone(), Precision::Exact(value.clone()))],
    )?;
    let mut builder = FlatBufferBuilder::new();
    let root = write_summary(&results, &dtype, &mut builder)?;
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    assert_eq!(
        read_summary(&wire, &dtype, &array_session())?.get(&aggregate),
        Precision::Exact(value)
    );
    Ok(())
}

#[rstest]
#[case::nan_sum(vec![Sum.bind(NumericalAggregateOpts::include_nans())])]
#[case::nan_min(vec![Min.bind(NumericalAggregateOpts::include_nans())])]
#[case::nan_max(vec![Max.bind(NumericalAggregateOpts::include_nans())])]
#[case::combined_extrema(vec![MinMax.bind(NumericalAggregateOpts::skip_nans())])]
#[case::duplicate(vec![NullCount.bind(EmptyOptions), NullCount.bind(EmptyOptions)])]
fn incompatible_selection_is_rejected(#[case] aggregates: Vec<AggregateFnRef>) {
    assert!(validate_selection(&aggregates).is_err());
}

#[test]
fn flags_for_types_without_sortedness_kernels_are_preserved() -> VortexResult<()> {
    let dtype = DType::Struct(StructFields::empty(), Nullability::NonNullable);
    let aggregate = LegacyStat::IsSorted.finalized_fn();
    assert!(aggregate.return_dtype(&dtype).is_none());
    let mut builder = FlatBufferBuilder::new();
    let root = fba::ArrayStats::create(
        &mut builder,
        &fba::ArrayStatsArgs {
            is_sorted: Some(true),
            ..Default::default()
        },
    );
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    assert_eq!(
        read_summary(&wire, &dtype, &array_session())?.get(&aggregate),
        Precision::Exact(true.into())
    );
    Ok(())
}

#[test]
fn counts_for_types_without_nan_count_kernels_are_preserved() -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let aggregate = LegacyStat::NaNCount.finalized_fn();
    assert!(aggregate.return_dtype(&dtype).is_none());
    let mut builder = FlatBufferBuilder::new();
    let root = fba::ArrayStats::create(
        &mut builder,
        &fba::ArrayStatsArgs {
            nan_count: Some(0),
            ..Default::default()
        },
    );
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    assert_eq!(
        read_summary(&wire, &dtype, &array_session())?.get(&aggregate),
        Precision::Exact(0u64.into())
    );
    Ok(())
}

#[test]
fn historical_extension_sum_retains_its_storage_result_type() -> VortexResult<()> {
    let dtype = DType::Extension(Date::new(TimeUnit::Days, Nullability::NonNullable).erased());
    let aggregate = Sum.bind(NumericalAggregateOpts::skip_nans());
    assert!(aggregate.return_dtype(&dtype).is_none());
    let mut builder = FlatBufferBuilder::new();
    let sum = builder.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&12i64.into())));
    let root = fba::ArrayStats::create(
        &mut builder,
        &fba::ArrayStatsArgs {
            sum: Some(sum),
            ..Default::default()
        },
    );
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    let results = read_summary(&wire, &dtype, &array_session())?;
    assert_eq!(
        results.get(&aggregate),
        Precision::Exact(Scalar::primitive(12i64, Nullability::Nullable))
    );
    let mut builder = FlatBufferBuilder::new();
    let root = write_summary(&results, &dtype, &mut builder)?;
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    assert_eq!(
        read_summary(&wire, &dtype, &array_session())?.get(&aggregate),
        results.get(&aggregate)
    );
    Ok(())
}

#[test]
fn exact_null_sum_survives_footer_encoding() -> VortexResult<()> {
    let dtype = DType::from(PType::I64);
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let value = Scalar::null(dtype.as_nullable());
    let results =
        AggregateResults::try_new(&dtype, [(sum.clone(), Precision::Exact(value.clone()))])?;
    let mut builder = FlatBufferBuilder::new();
    let root = write_summary(&results, &dtype, &mut builder)?;
    builder.finish_minimal(root);
    let wire = flatbuffers::root::<fba::ArrayStats>(builder.finished_data())?;
    assert_eq!(
        read_summary(&wire, &dtype, &array_session())?.get(&sum),
        Precision::Exact(value)
    );
    Ok(())
}

#[test]
fn inexact_counts_cannot_be_written_as_exact() -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let results = AggregateResults::try_new(
        &dtype,
        [(
            NullCount.bind(EmptyOptions),
            Precision::Inexact(3u64.into()),
        )],
    )?;
    assert!(write_summary(&results, &dtype, &mut FlatBufferBuilder::new()).is_err());
    Ok(())
}
