// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Statistics-based file ordering for TopK scans.

use std::cmp::Ordering;

use arrow_schema::Schema;
use datafusion_common::ScalarValue;
use datafusion_datasource::PartitionedFile;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::expressions::Column;

/// Reorders `files` so the files most likely to hold the first rows of `sort_order` are read
/// first, which tightens a TopK dynamic filter early and lets later files be pruned.
///
/// Files are compared lexicographically over the longest prefix of `sort_order` made of plain
/// columns. An ascending key compares file `min` values ascending and a descending key compares
/// file `max` values descending, so the leading file holds the most extreme value in the requested
/// direction. Files without the needed statistic sort after files that have it.
///
/// Returns `files` unchanged when there is no sort order or the leading key is not a plain column
/// of `table_schema`.
pub(crate) fn reorder_files_by_statistics(
    mut files: Vec<PartitionedFile>,
    sort_order: Option<&LexOrdering>,
    table_schema: &Schema,
) -> Vec<PartitionedFile> {
    let Some(sort_order) = sort_order else {
        return files;
    };

    // The leading key must resolve; later keys only break ties, so stop at the first one that
    // does not.
    let mut keys: Vec<(usize, bool)> = Vec::with_capacity(sort_order.len());
    for sort_expr in sort_order.iter() {
        let Some(index) = sort_expr
            .expr
            .downcast_ref::<Column>()
            .and_then(|col| table_schema.index_of(col.name()).ok())
        else {
            break;
        };
        keys.push((index, sort_expr.options.descending));
    }
    if keys.is_empty() || files.len() <= 1 {
        return files;
    }

    files.sort_by(|a, b| {
        keys.iter()
            .map(|&(index, descending)| {
                match (
                    file_bound(a, index, descending),
                    file_bound(b, index, descending),
                ) {
                    (Some(a), Some(b)) => {
                        let ord = a.partial_cmp(b).unwrap_or(Ordering::Equal);
                        if descending { ord.reverse() } else { ord }
                    }
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                }
            })
            .find(|ord| ord.is_ne())
            .unwrap_or(Ordering::Equal)
    });

    tracing::debug!(
        files = files.len(),
        %sort_order,
        "Reordered files by statistics for TopK"
    );

    files
}

/// Returns the file's `max` for a descending key and `min` for an ascending key.
fn file_bound(file: &PartitionedFile, index: usize, descending: bool) -> Option<&ScalarValue> {
    let column = file.statistics.as_ref()?.column_statistics.get(index)?;
    let bound = if descending {
        &column.max_value
    } else {
        &column.min_value
    };
    bound.get_value().filter(|value| !value.is_null())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::SortOptions;
    use datafusion_common::ColumnStatistics;
    use datafusion_common::Statistics;
    use datafusion_common::stats::Precision;
    use datafusion_physical_expr::PhysicalExprRef;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::lit;
    use rstest::rstest;

    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ])
    }

    fn stat(value: Option<i32>) -> Precision<ScalarValue> {
        value.map_or(Precision::Absent, |v| {
            Precision::Exact(ScalarValue::Int32(Some(v)))
        })
    }

    /// A file whose columns have the given `(min, max)` bounds.
    fn file(name: &str, bounds: &[(Option<i32>, Option<i32>)]) -> PartitionedFile {
        let mut file = PartitionedFile::new(name.to_string(), 0);
        let column_statistics = bounds
            .iter()
            .map(|&(min, max)| ColumnStatistics {
                min_value: stat(min),
                max_value: stat(max),
                ..ColumnStatistics::new_unknown()
            })
            .collect();
        file.statistics = Some(Arc::new(Statistics {
            column_statistics,
            ..Statistics::new_unknown(&schema())
        }));
        file
    }

    fn order(keys: &[(&str, usize, bool)]) -> Option<LexOrdering> {
        LexOrdering::new(keys.iter().map(|&(name, index, descending)| {
            PhysicalSortExpr::new(
                Arc::new(Column::new(name, index)) as PhysicalExprRef,
                SortOptions {
                    descending,
                    nulls_first: false,
                },
            )
        }))
    }

    fn names(files: &[PartitionedFile]) -> Vec<&str> {
        files
            .iter()
            .map(|f| f.object_meta.location.as_ref())
            .collect()
    }

    fn files() -> Vec<PartitionedFile> {
        vec![
            file("mid", &[(Some(10), Some(20))]),
            file("missing", &[(None, None)]),
            file("low_wide", &[(Some(0), Some(100))]),
            file("high", &[(Some(30), Some(40))]),
        ]
    }

    #[rstest]
    #[case::asc_by_min(false, vec!["low_wide", "mid", "high", "missing"])]
    #[case::desc_by_max(true, vec!["low_wide", "high", "mid", "missing"])]
    fn reorders_by_leading_column(#[case] descending: bool, #[case] expected: Vec<&str>) {
        let sort_order = order(&[("a", 0, descending)]);
        let reordered = reorder_files_by_statistics(files(), sort_order.as_ref(), &schema());
        assert_eq!(names(&reordered), expected);
    }

    #[test]
    fn breaks_ties_with_secondary_column() {
        let files = vec![
            file("b_high", &[(Some(1), Some(1)), (Some(50), Some(60))]),
            file("b_low", &[(Some(1), Some(1)), (Some(5), Some(6))]),
            file("a_high", &[(Some(2), Some(2)), (Some(0), Some(0))]),
        ];
        let sort_order = order(&[("a", 0, false), ("b", 1, true)]);
        let reordered = reorder_files_by_statistics(files, sort_order.as_ref(), &schema());
        assert_eq!(names(&reordered), vec!["b_high", "b_low", "a_high"]);
    }

    #[test]
    fn no_op_without_sort_order() {
        let reordered = reorder_files_by_statistics(files(), None, &schema());
        assert_eq!(names(&reordered), names(&files()));
    }

    #[test]
    fn no_op_when_leading_key_is_not_a_column() {
        let sort_order = LexOrdering::new([PhysicalSortExpr::new_default(lit(1i32))]);
        let reordered = reorder_files_by_statistics(files(), sort_order.as_ref(), &schema());
        assert_eq!(names(&reordered), names(&files()));
    }

    #[test]
    fn no_op_when_leading_column_is_unknown() {
        let sort_order = order(&[("missing_col", 0, false)]);
        let reordered = reorder_files_by_statistics(files(), sort_order.as_ref(), &schema());
        assert_eq!(names(&reordered), names(&files()));
    }
}
