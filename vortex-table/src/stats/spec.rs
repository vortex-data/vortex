// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The written form of a [`TableStats`]'s schema, kept in a metadata segment of the manifest.
//!
//! The values live in the manifest's `stats` column. Its fields name each aggregate by its
//! display form, which cannot be parsed back, so the segment records, for each column of the
//! stats table, the aggregate behind each of its fields, as zoned layouts record theirs: an id
//! and serialized options, resolved through the session's aggregate functions. It also records
//! the files' dtype, which a filter binds to before its proof is bound to the stats.

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::ColumnStats;
use super::TableStats;

/// The key of the manifest metadata segment holding the stats schema.
pub(crate) const STATS_SPEC_KEY: &str = "vortex.table.stats";

#[derive(Clone, PartialEq, ::prost::Message)]
struct TableStatsSpecProto {
    /// The files' dtype.
    #[prost(message, optional, tag = "1")]
    dtype: Option<pb::DType>,
    /// One entry per field of the stats table, in field order.
    #[prost(message, repeated, tag = "2")]
    columns: Vec<ColumnSpecProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ColumnSpecProto {
    /// One entry per field of the column's stats struct, in field order.
    #[prost(message, repeated, tag = "1")]
    aggregates: Vec<AggregateSpecProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct AggregateSpecProto {
    #[prost(string, tag = "1")]
    id: String,
    #[prost(bytes, tag = "2")]
    options: Vec<u8>,
}

/// The stats schema of `stats`, for a manifest metadata segment.
pub(crate) fn write_spec(stats: &TableStats) -> VortexResult<Vec<u8>> {
    let columns = stats
        .columns()
        .iter()
        .map(|column| {
            let aggregates = column
                .aggregates
                .iter()
                .map(|(aggregate_fn, _)| {
                    let options = aggregate_fn.options().serialize()?.ok_or_else(|| {
                        vortex_err!(
                            "aggregate function '{}' is not serializable",
                            aggregate_fn.id()
                        )
                    })?;
                    Ok(AggregateSpecProto {
                        id: aggregate_fn.id().to_string(),
                        options,
                    })
                })
                .collect::<VortexResult<_>>()?;
            Ok(ColumnSpecProto { aggregates })
        })
        .collect::<VortexResult<_>>()?;
    Ok(TableStatsSpecProto {
        dtype: Some(pb::DType::try_from(stats.dtype())?),
        columns,
    }
    .encode_to_vec())
}

/// The statistics a manifest stores: `table` from its `stats` column, `row_counts` from its
/// `row_count` column, and `spec` from its metadata segment.
///
/// An aggregate the session does not know is left unbound when the session allows unknown
/// plugins, so a proof needing it keeps every file, and is an error otherwise.
pub(crate) fn read_spec(
    spec: &[u8],
    table: ArrayRef,
    row_counts: ArrayRef,
    session: &VortexSession,
) -> VortexResult<TableStats> {
    let spec = TableStatsSpecProto::decode(spec)
        .map_err(|e| vortex_err!("invalid table stats spec: {e}"))?;
    let dtype = DType::from_proto(
        spec.dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("table stats spec has no dtype"))?,
        session,
    )?;
    let fields = table
        .dtype()
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("the manifest's stats column is not a struct"))?;
    vortex_ensure_eq!(fields.nfields(), spec.columns.len());
    let mut columns = Vec::with_capacity(spec.columns.len());
    for ((name, column_dtype), column) in
        fields.names().iter().zip(fields.fields()).zip(spec.columns)
    {
        let aggregate_fields = column_dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("the stats of column {name} are not a struct"))?;
        vortex_ensure_eq!(aggregate_fields.nfields(), column.aggregates.len());
        let mut aggregates = Vec::with_capacity(column.aggregates.len());
        for (field, aggregate) in aggregate_fields.names().iter().zip(column.aggregates) {
            #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
            let id = AggregateFnId::new(aggregate.id.as_str());
            let Some(plugin) = session.aggregate_fns().find_plugin(&id) else {
                if session.allows_unknown() {
                    continue;
                }
                vortex_bail!("unknown aggregate function id: {}", aggregate.id);
            };
            let aggregate_fn = plugin.deserialize(&aggregate.options, session)?;
            vortex_ensure_eq!(aggregate_fn.id(), id);
            aggregates.push((aggregate_fn, field.clone()));
        }
        columns.push(ColumnStats {
            name: name.clone(),
            aggregates,
        });
    }
    TableStats::from_parts(dtype, columns, table, row_counts, session)
}
