// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared helpers for the zoned layout's auxiliary stats-table schema.

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::stats::Stat;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

pub const MAX_IS_TRUNCATED: &str = "max_is_truncated";
pub const MIN_IS_TRUNCATED: &str = "min_is_truncated";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AggregateSpec {
    id: String,
    options: Vec<u8>,
}

impl AggregateSpec {
    pub(crate) fn try_from_aggregate_fn(aggregate_fn: &AggregateFnRef) -> VortexResult<Self> {
        let options = aggregate_fn.options().serialize()?.ok_or_else(|| {
            vortex_err!(
                "Aggregate function '{}' is not serializable",
                aggregate_fn.id()
            )
        })?;

        Ok(Self {
            id: aggregate_fn.id().to_string(),
            options,
        })
    }

    pub(crate) fn to_aggregate_fn(&self, session: &VortexSession) -> VortexResult<AggregateFnRef> {
        let aggregate_fn_id = AggregateFnId::new(self.id.as_str());
        let Some(plugin) = session.aggregate_fns().find_plugin(&aggregate_fn_id) else {
            vortex_bail!("unknown aggregate function id: {}", self.id);
        };

        let aggregate_fn = plugin.deserialize(&self.options, session)?;
        if aggregate_fn.id() != aggregate_fn_id {
            vortex_bail!(
                "Aggregate function ID mismatch: expected {}, got {}",
                aggregate_fn_id,
                aggregate_fn.id()
            );
        }

        Ok(aggregate_fn)
    }

    pub(super) fn from_proto(proto: AggregateSpecProto) -> Self {
        Self {
            id: proto.id,
            options: proto.options,
        }
    }

    pub(super) fn to_proto(&self) -> AggregateSpecProto {
        AggregateSpecProto {
            id: self.id.clone(),
            options: self.options.clone(),
        }
    }

    pub(crate) fn display_name(&self) -> String {
        if self.options.is_empty() {
            format!("{}()", self.id)
        } else {
            format!("{}({} option bytes)", self.id, self.options.len())
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub(super) struct AggregateSpecProto {
    #[prost(string, tag = "1")]
    id: String,
    #[prost(bytes, tag = "2")]
    options: Vec<u8>,
}

/// Return the auxiliary stats-table schema for a zoned layout.
pub(crate) fn aggregate_stats_table_dtype(
    column_dtype: &DType,
    aggregate_fns: &[AggregateFnRef],
) -> DType {
    DType::Struct(
        StructFields::from_iter(aggregate_fns.iter().filter_map(|aggregate_fn| {
            aggregate_state_dtype(column_dtype, aggregate_fn)
                .map(|dtype| (aggregate_fn.to_string(), dtype.as_nullable()))
        })),
        Nullability::NonNullable,
    )
}

pub(crate) fn legacy_stats_table_dtype(column_dtype: &DType, present_stats: &[Stat]) -> DType {
    assert!(present_stats.is_sorted(), "Stats must be sorted");
    DType::Struct(
        StructFields::from_iter(
            present_stats
                .iter()
                .filter_map(|stat| {
                    stat.dtype(column_dtype)
                        .or_else(|| {
                            // Backward compat: older files may have stored stats (e.g. Sum)
                            // for extension types by resolving through the storage dtype.
                            if let DType::Extension(ext) = column_dtype {
                                stat.dtype(ext.storage_dtype())
                            } else {
                                None
                            }
                        })
                        .map(|dtype| (stat, dtype.as_nullable()))
                })
                .flat_map(|(stat, dtype)| match stat {
                    Stat::Max => vec![
                        (stat.name(), dtype),
                        (MAX_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
                    ],
                    Stat::Min => vec![
                        (stat.name(), dtype),
                        (MIN_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
                    ],
                    _ => vec![(stat.name(), dtype)],
                }),
        ),
        Nullability::NonNullable,
    )
}

pub(crate) fn aggregate_specs_from_fns(
    aggregate_fns: &[AggregateFnRef],
) -> VortexResult<Arc<[AggregateSpec]>> {
    aggregate_fns
        .iter()
        .map(AggregateSpec::try_from_aggregate_fn)
        .collect::<VortexResult<Vec<_>>>()
        .map(Into::into)
}

pub(crate) fn aggregate_fns_from_specs(
    aggregate_specs: &[AggregateSpec],
    session: &VortexSession,
) -> VortexResult<Arc<[AggregateFnRef]>> {
    aggregate_specs
        .iter()
        .map(|aggregate_spec| aggregate_spec.to_aggregate_fn(session))
        .collect::<VortexResult<Vec<_>>>()
        .map(Into::into)
}

pub(crate) fn aggregate_state_dtype(
    column_dtype: &DType,
    aggregate_fn: &AggregateFnRef,
) -> Option<DType> {
    aggregate_fn.state_dtype(column_dtype).or_else(|| {
        if let DType::Extension(ext) = column_dtype {
            aggregate_fn.state_dtype(ext.storage_dtype())
        } else {
            None
        }
    })
}

pub(crate) fn default_bounded_stat_max_bytes() -> std::num::NonZeroUsize {
    // SAFETY: 64 is non-zero.
    unsafe { std::num::NonZeroUsize::new_unchecked(64) }
}

#[cfg(test)]
mod tests {
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;

    use super::*;

    #[test]
    fn stats_table_dtype_adds_truncation_flags() {
        let dtype = legacy_stats_table_dtype(
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            &[Stat::Max, Stat::Min, Stat::Sum],
        );

        assert_eq!(
            dtype.as_struct_fields().names().as_ref(),
            &[
                Stat::Max.name(),
                MAX_IS_TRUNCATED,
                Stat::Min.name(),
                MIN_IS_TRUNCATED,
                Stat::Sum.name(),
            ]
        );
    }

    #[test]
    fn stats_table_dtype_uses_storage_dtype_for_extensions() {
        let dtype = DType::Extension(Date::new(TimeUnit::Days, Nullability::NonNullable).erased());
        let stats_dtype = legacy_stats_table_dtype(&dtype, &[Stat::Max, Stat::Min]);

        assert_eq!(
            stats_dtype.as_struct_fields().names().as_ref(),
            &[
                Stat::Max.name(),
                MAX_IS_TRUNCATED,
                Stat::Min.name(),
                MIN_IS_TRUNCATED,
            ]
        );
    }

    #[test]
    fn aggregate_stats_table_dtype_uses_display_names() {
        let dtype = aggregate_stats_table_dtype(
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            &[
                Max.bind(EmptyOptions),
                Min.bind(EmptyOptions),
                Sum.bind(EmptyOptions),
            ],
        );

        assert_eq!(
            dtype.as_struct_fields().names().as_ref(),
            &[
                Max.bind(EmptyOptions).to_string(),
                Min.bind(EmptyOptions).to_string(),
                Sum.bind(EmptyOptions).to_string(),
            ]
        );
    }
}
