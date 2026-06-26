// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::stream;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_metrics::MetricsRegistry;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::ScanRequest;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::scan_builder::ScanBuilder;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
    split_max_row_count: u64,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

impl LayoutReaderDataSource {
    /// Creates a new [`LayoutReaderDataSource`].
    ///
    /// By default, the entire scan is returned as a single split. This best preserves V1
    /// `ScanBuilder` behavior where one scan covers the full row range, allowing the internal
    /// I/O pipeline and `SplitBy::Layout` chunking to operate without per-split overhead from
    /// redundant expression resolution and layout tree traversal.
    pub fn new(reader: LayoutReaderRef, session: VortexSession) -> Self {
        Self {
            reader,
            session,
            split_max_row_count: u64::MAX,
            metrics_registry: None,
        }
    }

    /// Sets the maximum number of rows per Scan API split.
    ///
    /// Each split drives a [`ScanBuilder`] over its row range, which internally handles
    /// physical layout alignment and I/O pipelining. This controls the engine-level
    /// parallelism granularity, not the I/O granularity.
    pub fn with_split_max_row_count(mut self, row_count: u64) -> Self {
        self.split_max_row_count = row_count;
        self
    }

    /// Sets the metrics registry for tracking scan performance.
    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics);
        self
    }

    /// Optionally sets the metrics registry for tracking scan performance.
    pub fn with_some_metrics_registry(mut self, metrics: Option<Arc<dyn MetricsRegistry>>) -> Self {
        self.metrics_registry = metrics;
        self
    }
}

#[async_trait]
impl DataSource for LayoutReaderDataSource {
    fn dtype(&self) -> &DType {
        self.reader.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.reader.row_count())
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("LayoutReader splits are not yet serializable");
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let total_rows = self.reader.row_count();
        let row_range = scan_request.row_range.unwrap_or(0..total_rows);

        let dtype = scan_request.projection.return_dtype(self.reader.dtype())?;

        // If the dtype is an empty struct, and there is no filter, we can return a special
        // length-only scan.
        if let DType::Struct(fields, Nullability::NonNullable) = &dtype
            && fields.nfields() == 0
            && scan_request.filter.is_none()
        {
            // FIXME(ngates): extract out maybe?
            let row_count = row_range.end - row_range.start;
            let row_count = scan_request.selection.row_count(row_count);

            // Apply the limit.
            let row_count = if let Some(limit) = scan_request.limit {
                row_count.min(limit)
            } else {
                row_count
            };

            return Ok(Arc::new(Empty { dtype, row_count }));
        }

        // Check file-level pruning: if the filter can be proven false for the entire row range
        // using file-level statistics (e.g. via FileStatsLayoutReader), skip the scan entirely.
        if let Some(filter) = &scan_request.filter {
            let mask = Mask::new_true(
                usize::try_from(row_range.end - row_range.start).unwrap_or(usize::MAX),
            );
            let pruning_result = self
                .reader
                .pruning_evaluation(&row_range, filter, mask)?
                .now_or_never();
            if let Some(Ok(result_mask)) = pruning_result
                && result_mask.all_false()
            {
                return Ok(Arc::new(Empty {
                    dtype,
                    row_count: 0,
                }));
            }
        }

        Ok(Arc::new(LayoutReaderScan {
            reader: Arc::clone(&self.reader),
            session: self.session.clone(),
            dtype,
            projection: scan_request.projection,
            filter: scan_request.filter,
            limit: scan_request.limit,
            selection: scan_request.selection,
            ordered: scan_request.ordered,
            metrics_registry: self.metrics_registry.clone(),
            start_row: row_range.start,
            end_row: row_range.end,
            split_size: self.split_max_row_count,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

struct LayoutReaderScan {
    reader: LayoutReaderRef,
    session: VortexSession,
    dtype: DType,
    projection: Expression,
    filter: Option<Expression>,
    limit: Option<u64>,
    ordered: bool,
    selection: Selection,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
    start_row: u64,
    end_row: u64,
    split_size: u64,
}

impl DataSourceScan for LayoutReaderScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        if self.start_row >= self.end_row {
            return Precision::exact(0usize);
        }

        if self.filter.is_none() && self.limit.is_some_and(|limit| limit == 0) {
            return Precision::exact(0usize);
        }

        let remaining_rows = self.end_row - self.start_row;
        let rows_to_scan = if self.filter.is_none() {
            self.limit
                .map_or(remaining_rows, |limit| remaining_rows.min(limit))
        } else {
            remaining_rows
        };
        let splits = rows_to_scan.div_ceil(self.split_size);
        Precision::exact(usize::try_from(splits).unwrap_or(usize::MAX))
    }

    fn partition(self: Arc<Self>, partition_idx: usize) -> VortexResult<Option<PartitionRef>> {
        let row_offset = (partition_idx as u64).saturating_mul(self.split_size);
        let split_start = self.start_row.saturating_add(row_offset);
        if split_start >= self.end_row {
            vortex_bail!(
                "layout reader scan partition {partition_idx} is outside 0..{}",
                self.partition_count().as_exact().unwrap_or(0)
            );
        }

        if self.filter.is_none() && self.limit.is_some_and(|limit| row_offset >= limit) {
            return Ok(None);
        }

        let split_end = split_start
            .saturating_add(self.split_size)
            .min(self.end_row);
        let row_range = split_start..split_end;

        let split_limit = if self.filter.is_none() {
            self.limit.map(|limit| limit.saturating_sub(row_offset))
        } else {
            self.limit
        };
        // With a filter, output cardinality is unknown, so each split receives the full limit.

        Ok(Some(Box::new(LayoutReaderSplit {
            reader: Arc::clone(&self.reader),
            session: self.session.clone(),
            projection: self.projection.clone(),
            filter: self.filter.clone(),
            limit: split_limit,
            ordered: self.ordered,
            row_range,
            selection: self.selection.clone(),
            metrics_registry: self.metrics_registry.clone(),
        })))
    }
}

struct LayoutReaderSplit {
    reader: LayoutReaderRef,
    session: VortexSession,
    projection: Expression,
    filter: Option<Expression>,
    limit: Option<u64>,
    ordered: bool,
    row_range: Range<u64>,
    selection: Selection,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

impl Partition for LayoutReaderSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[expect(clippy::cast_possible_truncation)]
    fn index(&self) -> usize {
        // Row range is unique per split
        self.row_range.start as usize
    }

    fn row_count(&self) -> Precision<u64> {
        let row_count = self.row_range.end - self.row_range.start;
        let row_count = self.selection.row_count(row_count);
        let row_count = self.limit.map_or(row_count, |limit| row_count.min(limit));

        if self.filter.is_some() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let builder = ScanBuilder::new(self.session, self.reader)
            .with_row_range(self.row_range)
            .with_selection(self.selection)
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_some_limit(self.limit)
            .with_some_metrics_registry(self.metrics_registry)
            .with_ordered(self.ordered);

        let dtype = builder.dtype()?;
        // Use into_stream() which creates a LazyScanStream that spawns individual I/O
        // tasks onto the runtime, enabling parallel execution across executor threads.
        let stream = builder.into_stream()?;

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

/// A scan that produces no data, only empty arrays with the correct row count.
struct Empty {
    dtype: DType,
    row_count: u64,
}

impl DataSourceScan for Empty {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partition(self: Arc<Self>, partition_idx: usize) -> VortexResult<Option<PartitionRef>> {
        if partition_idx != 0 {
            vortex_bail!("empty scan partition {partition_idx} is outside 0..1");
        }

        Ok(Some(Box::new(Empty {
            dtype: self.dtype.clone(),
            row_count: self.row_count,
        })))
    }
}

impl Partition for Empty {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        0
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.row_count)
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::exact(0u64)
    }

    fn execute(mut self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let scalar = Scalar::default_value(&self.dtype);
        let dtype = self.dtype.clone();

        // Create an iterator of arrays with the correct row count, respecting u64::MAX limits.
        let iter = std::iter::from_fn(move || {
            if self.row_count == 0 {
                return None;
            }
            let chunk_size = usize::try_from(self.row_count).unwrap_or(usize::MAX);
            self.row_count -= chunk_size as u64;
            Some(VortexResult::Ok(
                ConstantArray::new(scalar.clone(), chunk_size).into_array(),
            ))
        });

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            stream::iter(iter),
        )))
    }
}
