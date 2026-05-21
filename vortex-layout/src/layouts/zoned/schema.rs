// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared helpers for the zoned layout's auxiliary stats-table schema.

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::stats::Stat;

pub const MAX_IS_TRUNCATED: &str = "max_is_truncated";
pub const MIN_IS_TRUNCATED: &str = "min_is_truncated";

/// Return the auxiliary stats-table schema for a zoned layout.
pub(crate) fn stats_table_dtype(column_dtype: &DType, present_stats: &[Stat]) -> DType {
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

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;

    use super::*;

    #[test]
    fn stats_table_dtype_adds_truncation_flags() {
        let dtype = stats_table_dtype(
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
        let stats_dtype = stats_table_dtype(&dtype, &[Stat::Max, Stat::Min]);

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
}
