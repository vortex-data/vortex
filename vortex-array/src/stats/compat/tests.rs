// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use rstest::rstest;
use vortex_error::VortexResult;

use super::legacy_stats_to_results;
use super::read_summary;
use super::write_summary;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min::Min;
use crate::array_session;
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
