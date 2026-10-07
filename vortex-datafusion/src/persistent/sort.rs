// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inexact sort pushdown: reading data in an order that helps a TopK above the scan.
//!
//! A TopK's dynamic filter can only skip data once its threshold is tight, and the threshold
//! tightens fastest when the most promising rows are read first. When the requested ordering is
//! not already guaranteed, the scan still keeps the TopK but changes what it reads first:
//!
//! - files are queued by their statistics, so an ascending sort starts with the file holding the
//!   smallest values and a descending sort with the file holding the largest;
//! - within a file, a descending sort reads splits from the end of the file towards its start,
//!   which reaches the largest values first when data is stored in ascending order, as it
//!   typically is for time-ordered data.

use std::cmp::Ordering;

use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_datasource::PartitionedFile;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_plan::expressions::Column;

/// How an [`Inexact`](datafusion_physical_plan::SortOrderPushdownResult::Inexact) sort pushdown
/// changes the scan's read order.
#[derive(Debug, Clone)]
pub(crate) struct ReadOrder {
    /// The requested ordering, used to queue files by their statistics.
    pub(crate) sort_order: LexOrdering,
    /// Whether to read each file's splits from its end towards its start.
    pub(crate) reverse_splits: bool,
}

/// Plans the read order for a sort request that `eq_properties` does not already satisfy.
///
/// Returns `None` when changing the read order is not expected to help: the leading sort key is
/// not a column of the files, or the source already declares a proper prefix of the request, in
/// which case the TopK's own prefix-aware early termination works better with the declared order.
pub(crate) fn plan_read_order(
    order: &[PhysicalSortExpr],
    eq_properties: &EquivalenceProperties,
    file_schema: &Schema,
) -> DFResult<Option<ReadOrder>> {
    for prefix_len in 1..order.len() {
        if eq_properties.ordering_satisfy(order[..prefix_len].iter().cloned())? {
            return Ok(None);
        }
    }

    let Some(sort_order) = LexOrdering::new(order.iter().cloned()) else {
        return Ok(None);
    };
    let leading = sort_order.first();
    let leading_is_file_column = leading
        .expr
        .downcast_ref::<Column>()
        .is_some_and(|col| file_schema.field_with_name(col.name()).is_ok());
    if !leading_is_file_column {
        return Ok(None);
    }

    // A declared ordering that satisfies the request when reversed means the files are stored in
    // the opposite order, so reading them backwards approximates the request directly.
    let reversed_satisfies = {
        let mut reversed = eq_properties.clone();
        reversed.clear_orderings();
        reversed.add_orderings(eq_properties.oeq_class().iter().map(|ordering| {
            ordering
                .iter()
                .map(|expr| expr.reverse())
                .collect::<Vec<_>>()
        }));
        reversed.ordering_satisfy(order.iter().cloned())?
    };

    Ok(Some(ReadOrder {
        reverse_splits: reversed_satisfies || leading.options.descending,
        sort_order,
    }))
}

/// Orders `files` so that those most likely to hold the first rows of `sort_order` come first.
///
/// Files are compared lexicographically on the leading plain-column sort keys: ascending keys by
/// their minimum, descending keys by their maximum. Files without statistics go last.
pub(crate) fn reorder_files_by_statistics(
    mut files: Vec<PartitionedFile>,
    sort_order: &LexOrdering,
    table_schema: &Schema,
) -> Vec<PartitionedFile> {
    let keys: Vec<(usize, bool)> = sort_order
        .iter()
        .map_while(|sort_expr| {
            let col = sort_expr.expr.downcast_ref::<Column>()?;
            let idx = table_schema.index_of(col.name()).ok()?;
            Some((idx, sort_expr.options.descending))
        })
        .collect();
    if keys.is_empty() {
        return files;
    }

    files.sort_by(|a, b| {
        keys.iter()
            .map(|&(idx, descending)| {
                match (
                    bound(a, idx, descending).as_ref(),
                    bound(b, idx, descending).as_ref(),
                ) {
                    (Some(a), Some(b)) => {
                        let ordering = a.partial_cmp(b).unwrap_or(Ordering::Equal);
                        if descending {
                            ordering.reverse()
                        } else {
                            ordering
                        }
                    }
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                }
            })
            .find(|ordering| ordering.is_ne())
            .unwrap_or(Ordering::Equal)
    });
    files
}

/// The best value a file can contribute for a sort key: its maximum when descending, otherwise
/// its minimum.
fn bound(file: &PartitionedFile, column: usize, descending: bool) -> Option<ScalarValue> {
    let stats = file.statistics.as_ref()?.column_statistics.get(column)?;
    let value = if descending {
        &stats.max_value
    } else {
        &stats.min_value
    };
    value.get_value().filter(|value| !value.is_null()).cloned()
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
    use rstest::rstest;

    use super::*;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]))
    }

    fn sort(name: &str, index: usize, descending: bool) -> PhysicalSortExpr {
        PhysicalSortExpr::new(
            Arc::new(Column::new(name, index)),
            // DataFusion's defaults: ASC NULLS LAST, DESC NULLS FIRST.
            SortOptions {
                descending,
                nulls_first: descending,
            },
        )
    }

    fn file(name: &str, a: Option<(i32, i32)>) -> PartitionedFile {
        let column = |range: Option<(i32, i32)>| match range {
            Some((min, max)) => ColumnStatistics {
                min_value: Precision::Exact(ScalarValue::Int32(Some(min))),
                max_value: Precision::Exact(ScalarValue::Int32(Some(max))),
                ..ColumnStatistics::new_unknown()
            },
            None => ColumnStatistics::new_unknown(),
        };
        PartitionedFile::new(name, 100).with_statistics(Arc::new(Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Absent,
            column_statistics: vec![column(a), ColumnStatistics::new_unknown()],
        }))
    }

    fn names(files: &[PartitionedFile]) -> Vec<String> {
        files.iter().map(|f| f.path().to_string()).collect()
    }

    #[rstest]
    // Ascending: smallest minimum first.
    #[case(false, vec!["low", "mid", "high", "none"])]
    // Descending: largest maximum first, even when its minimum is not the largest.
    #[case(true, vec!["wide", "high", "mid", "none"])]
    fn reorders_files_by_leading_key(#[case] descending: bool, #[case] expected: Vec<&str>) {
        let files = vec![
            file("none", None),
            file("mid", Some((40, 60))),
            file("high", Some((80, 90))),
            file("low", Some((0, 10))),
        ];
        let files = if descending {
            let mut files = files;
            files.retain(|f| f.path().as_ref() != "low");
            files.push(file("wide", Some((5, 99))));
            files
        } else {
            files
        };
        let order = LexOrdering::new([sort("a", 0, descending)]).expect("non-empty ordering");

        let reordered = reorder_files_by_statistics(files, &order, &schema());

        assert_eq!(names(&reordered), expected);
    }

    #[test]
    fn non_column_leading_key_keeps_file_order() {
        let files = vec![file("b", Some((5, 6))), file("a", Some((1, 2)))];
        let order = LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(datafusion_physical_plan::expressions::Literal::new(
                ScalarValue::Int32(Some(1)),
            )),
            SortOptions::default(),
        )])
        .expect("non-empty ordering");

        let reordered = reorder_files_by_statistics(files, &order, &schema());

        assert_eq!(names(&reordered), vec!["b", "a"]);
    }

    #[rstest]
    #[case(false, false)]
    #[case(true, true)]
    fn plans_reverse_splits_for_descending(#[case] descending: bool, #[case] reverse: bool) {
        let schema = schema();
        let eq_properties = EquivalenceProperties::new(Arc::clone(&schema));

        let plan = plan_read_order(&[sort("a", 0, descending)], &eq_properties, &schema)
            .expect("planning succeeds")
            .expect("leading file column gets a read order");

        assert_eq!(plan.reverse_splits, reverse);
    }

    /// Reading in reverse helps a descending request over data declared ascending, and an
    /// ascending request over data declared descending.
    #[rstest]
    #[case(false, true, true)]
    #[case(true, false, true)]
    fn plans_reverse_splits_for_reversed_declared_ordering(
        #[case] declared_descending: bool,
        #[case] requested_descending: bool,
        #[case] reverse: bool,
    ) {
        let schema = schema();
        let eq_properties = EquivalenceProperties::new_with_orderings(
            Arc::clone(&schema),
            [vec![sort("a", 0, declared_descending)]],
        );

        let plan = plan_read_order(
            &[sort("a", 0, requested_descending)],
            &eq_properties,
            &schema,
        )
        .expect("planning succeeds")
        .expect("leading file column gets a read order");

        assert_eq!(plan.reverse_splits, reverse);
    }

    #[test]
    fn declines_when_a_prefix_is_already_ordered() {
        let schema = schema();
        let eq_properties = EquivalenceProperties::new_with_orderings(
            Arc::clone(&schema),
            [vec![sort("a", 0, false)]],
        );

        let plan = plan_read_order(
            &[sort("a", 0, false), sort("b", 1, false)],
            &eq_properties,
            &schema,
        )
        .expect("planning succeeds");

        assert!(plan.is_none());
    }

    #[test]
    fn declines_for_columns_outside_the_files() {
        let schema = schema();
        let file_schema = Schema::new(vec![Field::new("b", DataType::Int32, true)]);
        let eq_properties = EquivalenceProperties::new(Arc::clone(&schema));

        let plan = plan_read_order(&[sort("a", 0, false)], &eq_properties, &file_schema)
            .expect("planning succeeds");

        assert!(plan.is_none());
    }
}
