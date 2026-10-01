// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Translate historical node and footer fields to finalized aggregate results.
//!
//! The codec preserves the existing wire representation and validates scalar types using the
//! historical field definitions. It never reads array values or creates mergeable states.

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::LegacyStat;
use crate::aggregate_fn::AggregateFnRef;
use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::flatbuffers::array as fba;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::stats::AggregateResults;

/// Reject selections that the existing footer cannot represent, including unsupported options.
pub fn validate_selection(aggregates: &[AggregateFnRef]) -> VortexResult<()> {
    for (index, aggregate) in aggregates.iter().enumerate() {
        vortex_ensure!(
            LegacyStat::from_finalized_fn(aggregate).is_some(),
            "File footer cannot represent aggregate {aggregate}"
        );
        vortex_ensure!(
            !aggregates[..index].contains(aggregate),
            "Duplicate file aggregate: {aggregate}"
        );
    }

    Ok(())
}

/// Read finalized results without reading array values or creating accumulator states.
pub fn read_summary(
    fb: &fba::ArrayStats<'_>,
    input_dtype: &DType,
    session: &VortexSession,
) -> VortexResult<AggregateResults> {
    let mut entries = Vec::new();

    for stat in LegacyStat::all() {
        let aggregate = stat.finalized_fn();
        let Some(dtype) = stat.finalized_dtype(input_dtype) else {
            continue;
        };
        let value = match stat {
            LegacyStat::Min | LegacyStat::Max | LegacyStat::Sum => {
                let bytes = match stat {
                    LegacyStat::Min => fb.min(),
                    LegacyStat::Max => fb.max(),
                    _ => fb.sum(),
                };
                let Some(bytes) = bytes else { continue };
                let value = ScalarValue::from_proto_bytes(bytes.bytes(), &dtype, session)?;
                let value = Scalar::try_new(dtype, value)?;
                let precision = match stat {
                    LegacyStat::Min => fb.min_precision(),
                    LegacyStat::Max => fb.max_precision(),
                    _ => fba::Precision::Exact,
                };
                match precision {
                    fba::Precision::Exact => Precision::Exact(value),
                    fba::Precision::Inexact => Precision::Inexact(value),
                    other => vortex_bail!("Corrupted {stat} precision: {other:?}"),
                }
            }
            LegacyStat::IsConstant | LegacyStat::IsSorted | LegacyStat::IsStrictSorted => {
                let value = match stat {
                    LegacyStat::IsConstant => fb.is_constant(),
                    LegacyStat::IsSorted => fb.is_sorted(),
                    _ => fb.is_strict_sorted(),
                };
                let Some(value) = value else { continue };
                Precision::Exact(Scalar::try_new(dtype, Some(value.into()))?)
            }
            LegacyStat::NullCount | LegacyStat::NaNCount | LegacyStat::UncompressedSizeInBytes => {
                let value = match stat {
                    LegacyStat::NullCount => fb.null_count(),
                    LegacyStat::NaNCount => fb.nan_count(),
                    _ => fb.uncompressed_size_in_bytes(),
                };
                let Some(value) = value else { continue };
                Precision::Exact(Scalar::try_new(dtype, Some(value.into()))?)
            }
        };
        entries.push((aggregate, value));
    }

    Ok(AggregateResults::from_validated(entries))
}

/// Write finalized results into the existing footer representation.
///
/// Only min and max fields can store inexact bounds. Other inexact results are rejected.
pub fn write_summary<'fb>(
    results: &AggregateResults,
    input_dtype: &DType,
    fbb: &mut FlatBufferBuilder<'fb>,
) -> VortexResult<WIPOffset<fba::ArrayStats<'fb>>> {
    write_fields(results, input_dtype, fbb, false)
}

/// Project cached finalized results into historical array-node hints.
///
/// Functions and options without a historical field are omitted. Only extrema can preserve an
/// inexact result because the other wire fields have no precision flag.
pub fn write_node_summary<'fb>(
    results: &AggregateResults,
    input_dtype: &DType,
    fbb: &mut FlatBufferBuilder<'fb>,
) -> VortexResult<WIPOffset<fba::ArrayStats<'fb>>> {
    write_fields(results, input_dtype, fbb, true)
}

fn write_fields<'fb>(
    results: &AggregateResults,
    input_dtype: &DType,
    fbb: &mut FlatBufferBuilder<'fb>,
    project_node: bool,
) -> VortexResult<WIPOffset<fba::ArrayStats<'fb>>> {
    let mut args = fba::ArrayStatsArgs::default();

    for (aggregate, value) in results.iter() {
        let Some(stat) = LegacyStat::from_finalized_fn(aggregate) else {
            if project_node {
                continue;
            }
            vortex_bail!("File footer cannot represent aggregate {aggregate}");
        };
        if project_node && !value.is_exact() && !matches!(stat, LegacyStat::Min | LegacyStat::Max) {
            continue;
        }
        let Some(scalar) = value.as_ref().into_inner() else {
            continue;
        };
        let dtype = stat.finalized_dtype(input_dtype).ok_or_else(|| {
            vortex_err!("File aggregate {aggregate} does not support {input_dtype}")
        })?;
        vortex_ensure!(
            scalar.dtype() == &dtype,
            "Aggregate {aggregate} requires result dtype {dtype}, got {}",
            scalar.dtype()
        );
        vortex_ensure!(
            value.is_exact() || matches!(stat, LegacyStat::Min | LegacyStat::Max),
            "File footer cannot represent an inexact {aggregate} result"
        );
        match stat {
            LegacyStat::Min | LegacyStat::Max | LegacyStat::Sum => {
                let bytes = ScalarValue::to_proto_bytes::<Vec<u8>>(scalar.value());
                let bytes = Some(fbb.create_vector(&bytes));
                let precision = if value.is_exact() {
                    fba::Precision::Exact
                } else {
                    fba::Precision::Inexact
                };
                match stat {
                    LegacyStat::Min => {
                        args.min = bytes;
                        args.min_precision = precision;
                    }
                    LegacyStat::Max => {
                        args.max = bytes;
                        args.max_precision = precision;
                    }
                    _ => args.sum = bytes,
                }
            }
            LegacyStat::IsConstant => args.is_constant = Some(bool::try_from(scalar)?),
            LegacyStat::IsSorted => args.is_sorted = Some(bool::try_from(scalar)?),
            LegacyStat::IsStrictSorted => args.is_strict_sorted = Some(bool::try_from(scalar)?),
            LegacyStat::NullCount => args.null_count = Some(u64::try_from(scalar)?),
            LegacyStat::NaNCount => args.nan_count = Some(u64::try_from(scalar)?),
            LegacyStat::UncompressedSizeInBytes => {
                args.uncompressed_size_in_bytes = Some(u64::try_from(scalar)?);
            }
        }
    }

    Ok(fba::ArrayStats::create(fbb, &args))
}
