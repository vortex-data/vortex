// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;
use futures::FutureExt;
use futures::Stream;
use futures::stream;
use futures::stream::StreamExt;
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
use vortex_scan::PartitionStream;
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

    fn row_count(&self) -> Option<Precision<u64>> {
        Some(Precision::exact(self.reader.row_count()))
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        None
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

            return Ok(Box::new(Empty { dtype, row_count }));
        }

        // Check file-level pruning: if the filter can be proven false for the entire row range
        // using file-level statistics (e.g. via FileStatsLayoutReader), skip the scan entirely.
        if let Some(ref filter) = scan_request.filter {
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
                return Ok(Box::new(Empty {
                    dtype,
                    row_count: 0,
                }));
            }
        }

        Ok(Box::new(LayoutReaderScan {
            reader: Arc::clone(&self.reader),
            session: self.session.clone(),
            dtype,
            projection: scan_request.projection,
            filter: scan_request.filter,
            limit: scan_request.limit,
            selection: scan_request.selection,
            ordered: scan_request.ordered,
            metrics_registry: self.metrics_registry.clone(),
            next_row: row_range.start,
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
    next_row: u64,
    end_row: u64,
    split_size: u64,
}

impl DataSourceScan for LayoutReaderScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Option<Precision<usize>> {
        let (lower, upper) = self.size_hint();
        match upper {
            Some(u) if u == lower => Some(Precision::exact(lower)),
            Some(u) => Some(Precision::inexact(u)),
            None => Some(Precision::inexact(lower)),
        }
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        (*self).boxed()
    }
}

impl Stream for LayoutReaderScan {
    type Item = VortexResult<PartitionRef>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.next_row >= this.end_row {
            return Poll::Ready(None);
        }

        if this.limit.is_some_and(|limit| limit == 0) {
            return Poll::Ready(None);
        }

        let split_end = this
            .next_row
            .saturating_add(this.split_size)
            .min(this.end_row);
        let row_range = this.next_row..split_end;
        let split_rows = split_end - this.next_row;

        let split_limit = this.limit;
        // Only decrement the remaining limit when there is no filter. With a filter,
        // the actual output row count is unknown (could be anywhere from 0 to split_rows),
        // so decrementing by split_rows would be too aggressive and could stop producing
        // splits before the limit is reached. Instead, pass the full remaining limit to
        // each split and let the engine enforce the exact limit at the stream level.
        if this.filter.is_none()
            && let Some(ref mut limit) = this.limit
        {
            *limit = limit.saturating_sub(split_rows);
        }

        let split = Box::new(LayoutReaderSplit {
            reader: Arc::clone(&this.reader),
            session: this.session.clone(),
            projection: this.projection.clone(),
            filter: this.filter.clone(),
            limit: split_limit,
            ordered: this.ordered,
            row_range,
            selection: this.selection.clone(),
            metrics_registry: this.metrics_registry.clone(),
        }) as PartitionRef;

        this.next_row = split_end;

        Poll::Ready(Some(Ok(split)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.next_row >= self.end_row {
            return (0, Some(0));
        }
        let remaining_rows = self.end_row - self.next_row;
        let splits = remaining_rows.div_ceil(self.split_size);
        (0, Some(usize::try_from(splits).unwrap_or(usize::MAX)))
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

    fn row_count(&self) -> Option<Precision<u64>> {
        let row_count = self.row_range.end - self.row_range.start;
        let row_count = self.selection.row_count(row_count);
        let row_count = self.limit.map_or(row_count, |limit| row_count.min(limit));

        Some(if self.filter.is_some() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        })
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        None
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

    fn partition_count(&self) -> Option<Precision<usize>> {
        Some(Precision::exact(1usize))
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter([Ok(self as _)]).boxed()
    }
}

impl Partition for Empty {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn row_count(&self) -> Option<Precision<u64>> {
        Some(Precision::exact(self.row_count))
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        Some(Precision::exact(0u64))
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
