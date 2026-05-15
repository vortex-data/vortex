// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared helpers for the zoned layout's auxiliary stats-table schema.

use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::stats::Stat;
use vortex_error::VortexExpect;

pub const MAX_IS_TRUNCATED: &str = "max_is_truncated";
pub const MIN_IS_TRUNCATED: &str = "min_is_truncated";

/// Return the auxiliary stats-table schema for a zoned layout.
pub(crate) fn aggregate_stats_table_dtype(
    column_dtype: &DType,
    aggregate_fns: &[AggregateFnRef],
) -> DType {
    DType::Struct(
        StructFields::from_iter(aggregate_fns.iter().filter_map(|aggregate_fn| {
            aggregate_state_dtype(column_dtype, aggregate_fn)
                .map(|dtype| (aggregate_descriptor(aggregate_fn), dtype.as_nullable()))
        })),
        Nullability::NonNullable,
    )
}

pub(crate) fn descriptor_stats_table_dtype(
    column_dtype: &DType,
    present_aggregates: &[String],
) -> DType {
    DType::Struct(
        StructFields::from_iter(present_aggregates.iter().filter_map(|descriptor| {
            descriptor_aggregate_fn(descriptor).and_then(|aggregate_fn| {
                aggregate_state_dtype(column_dtype, &aggregate_fn)
                    .map(|dtype| (descriptor.as_str(), dtype.as_nullable()))
            })
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

pub(crate) fn aggregate_descriptor(aggregate_fn: &AggregateFnRef) -> String {
    aggregate_fn.to_string()
}

pub(crate) fn descriptors_from_legacy_stats(stats: &[Stat]) -> Arc<[String]> {
    stats
        .iter()
        .filter_map(Stat::aggregate_fn)
        .map(|aggregate_fn| aggregate_descriptor(&aggregate_fn))
        .collect::<Vec<_>>()
        .into()
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

pub(crate) fn descriptor_aggregate_fn(descriptor: &str) -> Option<AggregateFnRef> {
    builtin_aggregate_fns()
        .into_iter()
        .find(|aggregate_fn| aggregate_descriptor(aggregate_fn) == descriptor)
}

fn builtin_aggregate_fns() -> [AggregateFnRef; 12] {
    [
        AllNan.bind(EmptyOptions),
        AllNonNan.bind(EmptyOptions),
        AllNonNull.bind(EmptyOptions),
        AllNull.bind(EmptyOptions),
        BoundedMax.bind(BoundedMaxOptions {
            max_bytes: default_bounded_stat_max_bytes(),
        }),
        BoundedMin.bind(BoundedMinOptions {
            max_bytes: default_bounded_stat_max_bytes(),
        }),
        Max.bind(EmptyOptions),
        Min.bind(EmptyOptions),
        NanCount.bind(EmptyOptions),
        NullCount.bind(EmptyOptions),
        Sum.bind(EmptyOptions),
        UncompressedSizeInBytes.bind(EmptyOptions),
    ]
}

fn default_bounded_stat_max_bytes() -> NonZeroUsize {
    NonZeroUsize::new(64).vortex_expect("non-zero default bounded stat byte size")
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
    fn aggregate_stats_table_dtype_uses_descriptors_as_names() {
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
