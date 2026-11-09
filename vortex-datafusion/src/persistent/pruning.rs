// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::BooleanArray;
use arrow_schema::SchemaRef;
use datafusion_common::Column;
use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::pruning::PrunableStatistics;
use datafusion_common::pruning::PruningStatistics;
use datafusion_common::stats::Precision;

/// Preserve known bounds when other files have unknown statistics. DataFusion's adapter uses
/// untyped nulls for unknown bounds, which cannot be combined with typed values in an Arrow array.
pub(super) struct FilePruningStatistics<'a> {
    statistics: &'a [Arc<Statistics>],
    schema: SchemaRef,
    inner: PrunableStatistics,
}

impl<'a> FilePruningStatistics<'a> {
    pub(super) fn new(statistics: &'a [Arc<Statistics>], schema: SchemaRef) -> Self {
        let inner = PrunableStatistics::new(statistics.to_vec(), Arc::clone(&schema));
        Self {
            statistics,
            schema,
            inner,
        }
    }

    fn bounds(
        &self,
        column: &Column,
        get_bound: impl Fn(&ColumnStatistics) -> &Precision<ScalarValue>,
    ) -> Option<ArrayRef> {
        let index = self.schema.index_of(column.name()).ok()?;
        let dtype = self.schema.field(index).data_type();
        let unknown = ScalarValue::try_new_null(dtype).ok()?;
        let mut has_value = false;
        let array = ScalarValue::iter_to_array(self.statistics.iter().map(|statistics| {
            if let Some(statistics) = statistics.column_statistics.get(index)
                && let Precision::Exact(value) = get_bound(statistics)
                && !value.is_null()
                && value.data_type() == *dtype
            {
                has_value = true;
                value.clone()
            } else {
                unknown.clone()
            }
        }))
        .ok()?;
        has_value.then_some(array)
    }
}

impl PruningStatistics for FilePruningStatistics<'_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, |statistics| &statistics.min_value)
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, |statistics| &statistics.max_value)
    }

    fn num_containers(&self) -> usize {
        self.inner.num_containers()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        self.inner.null_counts(column)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        self.inner.row_counts()
    }

    #[expect(clippy::disallowed_types, reason = "required by trait signature")]
    fn contained(
        &self,
        column: &Column,
        values: &std::collections::HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        self.inner.contained(column, values)
    }
}
