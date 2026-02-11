// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use futures::stream::StreamExt;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_layout::LayoutReaderRef;
use vortex_metrics::MetricsRegistry;
use vortex_session::VortexSession;

use crate::ScanBuilder;
use crate::Selection;
use crate::api::DataSource;
use crate::api::DataSourceScan;
use crate::api::DataSourceScanRef;
use crate::api::Estimate;
use crate::api::ScanRequest;
use crate::api::Split;
use crate::api::SplitRef;
use crate::api::SplitStream;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
    split_size: u64,
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
            split_size: u64::MAX,
            metrics_registry: None,
        }
    }

    /// Sets the target number of rows per Scan API split.
    ///
    /// Each split drives a [`ScanBuilder`] over its row range, which internally handles
    /// physical layout alignment and I/O pipelining. This controls the engine-level
    /// parallelism granularity, not the I/O granularity.
    pub fn with_split_size(mut self, split_size: u64) -> Self {
        self.split_size = split_size;
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

    fn row_count_estimate(&self) -> Estimate<u64> {
        Estimate::exact(self.reader.row_count())
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let total_rows = self.reader.row_count();
        let row_range = scan_request.row_range.unwrap_or(0..total_rows);

        let dtype = if let Some(proj) = &scan_request.projection {
            proj.return_dtype(self.reader.dtype())?
        } else {
            self.reader.dtype().clone()
        };

        Ok(Box::new(LayoutReaderScan {
            reader: self.reader.clone(),
            session: self.session.clone(),
            dtype,
            projection: scan_request.projection,
            filter: scan_request.filter,
            limit: scan_request.limit,
            selection: scan_request.selection,
            metrics_registry: self.metrics_registry.clone(),
            next_row: row_range.start,
            end_row: row_range.end,
            split_size: self.split_size,
        }))
    }

    fn deserialize_split(&self, _data: &[u8], _session: &VortexSession) -> VortexResult<SplitRef> {
        vortex_bail!("LayoutReader splits are not yet serializable");
    }
}

struct LayoutReaderScan {
    reader: LayoutReaderRef,
    session: VortexSession,
    dtype: DType,
    projection: Option<Expression>,
    filter: Option<Expression>,
    limit: Option<u64>,
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

    fn splits_estimate(&self) -> Estimate<usize> {
        if self.next_row >= self.end_row {
            return Estimate::exact(0);
        }
        let remaining_rows = self.end_row - self.next_row;
        let splits = remaining_rows.div_ceil(self.split_size);
        Estimate {
            lower: 0,
            upper: Some(usize::try_from(splits).unwrap_or(usize::MAX)),
        }
    }

    fn splits(self: Box<Self>) -> SplitStream {
        stream::unfold(*self, |mut state| async move {
            if state.next_row >= state.end_row {
                return None;
            }

            if state.limit.is_some_and(|limit| limit == 0) {
                return None;
            }

            let split_end = state
                .next_row
                .saturating_add(state.split_size)
                .min(state.end_row);
            let row_range = state.next_row..split_end;
            let split_rows = split_end - state.next_row;

            let split_limit = state.limit;
            if let Some(ref mut limit) = state.limit {
                *limit = limit.saturating_sub(split_rows);
            }

            let split = Box::new(LayoutReaderSplit {
                reader: state.reader.clone(),
                session: state.session.clone(),
                projection: state.projection.clone(),
                filter: state.filter.clone(),
                limit: split_limit,
                row_range,
                selection: state.selection.clone(),
                metrics_registry: state.metrics_registry.clone(),
            }) as SplitRef;

            state.next_row = split_end;

            Some((Ok(split), state))
        })
        .boxed()
    }
}

struct LayoutReaderSplit {
    reader: LayoutReaderRef,
    session: VortexSession,
    projection: Option<Expression>,
    filter: Option<Expression>,
    limit: Option<u64>,
    row_range: Range<u64>,
    selection: Selection,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

impl Split for LayoutReaderSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let mut builder = ScanBuilder::new(self.session, self.reader)
            .with_row_range(self.row_range)
            .with_selection(self.selection);

        if let Some(proj) = self.projection {
            builder = builder.with_projection(proj);
        }
        if let Some(filter) = self.filter {
            builder = builder.with_filter(filter);
        }
        if let Some(limit) = self.limit {
            builder = builder.with_limit(limit);
        }
        if let Some(metrics) = self.metrics_registry {
            builder = builder.with_metrics_registry(metrics);
        }

        Ok(Box::pin(builder.into_array_stream()?))
    }

    fn row_count_estimate(&self) -> Estimate<u64> {
        Estimate {
            lower: 0,
            upper: Some(self.row_range.end - self.row_range.start),
        }
    }

    fn byte_size_estimate(&self) -> Estimate<u64> {
        Estimate::default()
    }
}
