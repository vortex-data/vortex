// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::aggregate_fn::AggregateFnRef;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::flatbuffers::array as fba;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::stats::AggregateResults;
use crate::stats::StatsSet;

/// Convert legacy writer output without reading array values or computing aggregates.
///
/// Only values present in the legacy set are supplied. The legacy set cannot contain nulls.
pub fn legacy_stats_to_results(
    input_dtype: &DType,
    stats: &StatsSet,
) -> VortexResult<AggregateResults> {
    let mut entries = Vec::new();
    for stat in Stat::all() {
        let value = stats.get(stat);
        if value.is_absent() {
            continue;
        }
        let dtype = historical_dtype(stat, input_dtype)
            .ok_or_else(|| vortex_err!("File statistic {stat} does not support {input_dtype}"))?;
        let value = value
            .map(|value| Scalar::try_new(dtype, Some(value)))
            .transpose()?;
        entries.push((stat.finalized_aggregate_fn().clone(), value));
    }
    Ok(AggregateResults::from_validated(entries))
}

/// Decode finalized results using historical field types.
///
/// Present nullable nulls retain their precision. Missing fields are absent. Nulls in non-nullable
/// fields and invalid precision tags are rejected. Numerical keys use NaN-skipping options.
pub fn read_summary(
    fb: &fba::ArrayStats<'_>,
    input_dtype: &DType,
    session: &VortexSession,
) -> VortexResult<AggregateResults> {
    let mut entries = Vec::new();
    for stat in Stat::all() {
        let Some(dtype) = historical_dtype(stat, input_dtype) else {
            continue;
        };
        let value = match stat {
            Stat::Min | Stat::Max | Stat::Sum => {
                let bytes = match stat {
                    Stat::Min => fb.min(),
                    Stat::Max => fb.max(),
                    _ => fb.sum(),
                };
                let Some(bytes) = bytes else { continue };
                let value = ScalarValue::from_proto_bytes(bytes.bytes(), &dtype, session)?;
                let value = Scalar::try_new(dtype, value)?;
                let precision = match stat {
                    Stat::Min => fb.min_precision(),
                    Stat::Max => fb.max_precision(),
                    _ => fba::Precision::Exact,
                };
                match precision {
                    fba::Precision::Exact => Precision::Exact(value),
                    fba::Precision::Inexact => Precision::Inexact(value),
                    other => vortex_bail!("Corrupted {stat} precision: {other:?}"),
                }
            }
            Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {
                let value = match stat {
                    Stat::IsConstant => fb.is_constant(),
                    Stat::IsSorted => fb.is_sorted(),
                    _ => fb.is_strict_sorted(),
                };
                let Some(value) = value else { continue };
                Precision::Exact(Scalar::try_new(dtype, Some(value.into()))?)
            }
            Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                let value = match stat {
                    Stat::NullCount => fb.null_count(),
                    Stat::NaNCount => fb.nan_count(),
                    _ => fb.uncompressed_size_in_bytes(),
                };
                let Some(value) = value else { continue };
                Precision::Exact(Scalar::try_new(dtype, Some(value.into()))?)
            }
        };
        entries.push((stat.finalized_aggregate_fn().clone(), value));
    }
    Ok(AggregateResults::from_validated(entries))
}

/// Validate selected functions and options against the historical footer fields.
///
/// Writers call this before producing bytes. Field dtype support is checked separately, since
/// unsupported field types omit individual results.
pub fn validate_summary_aggregates<'a>(
    aggregates: impl IntoIterator<Item = &'a AggregateFnRef>,
) -> VortexResult<()> {
    for aggregate in aggregates {
        vortex_ensure!(
            Stat::all().any(|stat| stat.finalized_aggregate_fn() == aggregate),
            "File footer cannot represent aggregate {aggregate}"
        );
    }

    Ok(())
}

/// Write finalized results into the existing footer fields.
///
/// Only fixed historical functions and options are supported. Only extrema can be inexact.
/// Scalar types must match the historical field, allowing a change to root nullability. Nulls
/// require a nullable historical field. Legacy `StatsSet` decoding omits those null results.
pub fn write_summary<'fb>(
    results: &AggregateResults,
    input_dtype: &DType,
    fbb: &mut FlatBufferBuilder<'fb>,
) -> VortexResult<WIPOffset<fba::ArrayStats<'fb>>> {
    validate_summary_aggregates(results.iter().map(|(aggregate, _)| aggregate))?;

    let mut args = fba::ArrayStatsArgs::default();
    // Preserve the legacy serializer's scalar-vector order so unchanged writer output has the same
    // bytes. The table's optional numeric fields do not allocate vectors.
    for stat in [
        Stat::Min,
        Stat::Max,
        Stat::Sum,
        Stat::IsConstant,
        Stat::IsSorted,
        Stat::IsStrictSorted,
        Stat::NullCount,
        Stat::UncompressedSizeInBytes,
        Stat::NaNCount,
    ] {
        let value = results.get_result(stat.finalized_aggregate_fn());
        let Some(scalar) = value.as_ref().into_inner() else {
            continue;
        };
        let dtype = historical_dtype(stat, input_dtype)
            .ok_or_else(|| vortex_err!("File statistic {stat} does not support {input_dtype}"))?;
        vortex_ensure!(
            scalar.dtype().as_nullable() == dtype.as_nullable(),
            "File statistic {stat} requires result dtype {dtype}, got {}",
            scalar.dtype()
        );
        let scalar = Scalar::try_new(dtype, scalar.value().cloned())?;
        vortex_ensure!(
            value.is_exact() || matches!(stat, Stat::Min | Stat::Max),
            "File footer cannot represent an inexact {stat} result"
        );
        match stat {
            Stat::Min | Stat::Max | Stat::Sum => {
                let bytes = ScalarValue::to_proto_bytes::<Vec<u8>>(scalar.value());
                let bytes = Some(fbb.create_vector(&bytes));
                let precision = if value.is_exact() {
                    fba::Precision::Exact
                } else {
                    fba::Precision::Inexact
                };
                match stat {
                    Stat::Min => {
                        args.min = bytes;
                        args.min_precision = precision;
                    }
                    Stat::Max => {
                        args.max = bytes;
                        args.max_precision = precision;
                    }
                    _ => args.sum = bytes,
                }
            }
            Stat::IsConstant => args.is_constant = Some(bool::try_from(&scalar)?),
            Stat::IsSorted => args.is_sorted = Some(bool::try_from(&scalar)?),
            Stat::IsStrictSorted => args.is_strict_sorted = Some(bool::try_from(&scalar)?),
            Stat::NullCount => args.null_count = Some(u64::try_from(&scalar)?),
            Stat::NaNCount => args.nan_count = Some(u64::try_from(&scalar)?),
            Stat::UncompressedSizeInBytes => {
                args.uncompressed_size_in_bytes = Some(u64::try_from(&scalar)?);
            }
        }
    }
    Ok(fba::ArrayStats::create(fbb, &args))
}

fn historical_dtype(stat: Stat, input_dtype: &DType) -> Option<DType> {
    match stat {
        // These wire fields exist for every input type, even where a current kernel declines.
        Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => Some(PType::U64.into()),
        Stat::Sum => stat.dtype(input_dtype).or_else(|| {
            // Older writers also summed extension storage values.
            if let DType::Extension(ext) = input_dtype {
                historical_dtype(stat, ext.storage_dtype())
            } else {
                None
            }
        }),
        _ => stat.dtype(input_dtype),
    }
}
