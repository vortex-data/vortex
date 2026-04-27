// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`DataSource`] that combines multiple [`LayoutReaderRef`]s into a single scannable source.
//!
//! Readers may be pre-opened or deferred via [`LayoutReaderFactory`]. Deferred readers are opened
//! concurrently during scanning using `buffer_unordered`: up to `concurrency` file opens run in
//! parallel as spawned tasks on the session runtime. Once opened, each reader yields a single
//! partition covering its full row range; internal I/O pipelining and chunking are handled by
//! [`ScanBuilder`].
//!
//! # Schema Resolution
//!
//! Currently, all children must share the exact same [`DType`]. A dtype
//! mismatch produces an error.
//!
//! # Future Work
//!
//! - **Schema union**: Allow missing columns (filled with nulls) and compatible type upcasts
//!   across sources instead of requiring exact dtype matches.
//! - **Hive-style partitioning**: Extract partition values from file paths (e.g. `year=2024/month=01/`)
//!   and expose them as virtual columns.
//! - **Virtual columns**: `filename`, `file_row_number`, `file_index`.
//! - **Per-file statistics**: Merge column statistics across sources for planner hints.
//! - **Error resilience**: Skip failed sources instead of aborting the entire scan.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::stream;
use tracing::Instrument;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::scan_builder::ScanBuilder;

/// Default concurrency for opening deferred readers.
const DEFAULT_CONCURRENCY: usize = 8;

/// An async factory that produces a [`LayoutReaderRef`].
///
/// Implementations handle file opening, footer caching, and statistics-based pruning.
/// Returns `None` if the source should be skipped (e.g., pruned based on file-level
/// statistics before the reader is fully constructed).
#[async_trait]
pub trait LayoutReaderFactory: 'static + Send + Sync {
    /// Opens the layout reader, or returns `None` if it should be skipped.
    async fn open(&self) -> VortexResult<Option<LayoutReaderRef>>;
}

/// A [`DataSource`] that combines multiple [`LayoutReaderRef`]s into a single scannable source.
///
/// Readers may be pre-opened or deferred via [`LayoutReaderFactory`]. Deferred readers are opened
/// concurrently during scanning using `buffer_unordered`, mirroring the DuckDB scan pattern: up
/// to `concurrency` file opens run in parallel as spawned tasks on the session runtime. Once
/// opened, each reader yields a single partition covering its full row range; internal I/O
/// pipelining and chunking are handled by [`ScanBuilder`].
pub struct MultiLayoutDataSource {
    dtype: DType,
    session: VortexSession,
    children: Vec<MultiLayoutChild>,
    concurrency: usize,
}

pub enum MultiLayoutChild {
    Opened(LayoutReaderRef),
    Deferred(Arc<dyn LayoutReaderFactory>),
}

impl MultiLayoutDataSource {
    /// Creates a multi-layout data source with the first reader pre-opened.
    ///
    /// The first reader determines the dtype. Remaining readers are opened lazily during
    /// scanning via their factories.
    pub fn new_with_first(
        first: LayoutReaderRef,
        remaining: Vec<Arc<dyn LayoutReaderFactory>>,
        session: &VortexSession,
    ) -> Self {
        let dtype = first.dtype().clone();
        let concurrency = get_available_parallelism().unwrap_or(DEFAULT_CONCURRENCY);

        let mut children = Vec::with_capacity(1 + remaining.len());
        children.push(MultiLayoutChild::Opened(first));
        children.extend(remaining.into_iter().map(MultiLayoutChild::Deferred));

        Self {
            dtype,
            session: session.clone(),
            children,
            concurrency,
        }
    }

    /// Creates a multi-layout data source where all children are deferred.
    ///
    /// The dtype must be provided externally since there is no pre-opened reader to infer it
    /// from. This avoids eagerly opening any file when the schema is already known (e.g. from
    /// a catalog or a prior scan).
    pub fn new_deferred(
        dtype: DType,
        factories: Vec<Arc<dyn LayoutReaderFactory>>,
        session: &VortexSession,
    ) -> Self {
        let concurrency = get_available_parallelism().unwrap_or(DEFAULT_CONCURRENCY);

        Self {
            dtype,
            session: session.clone(),
            children: factories
                .into_iter()
                .map(MultiLayoutChild::Deferred)
                .collect(),
            concurrency,
        }
    }

    pub fn children(&self) -> &Vec<MultiLayoutChild> {
        &self.children
    }

    /// Sets the concurrency for opening deferred readers.
    ///
    /// Controls how many file opens run in parallel via `buffer_unordered`.
    /// Defaults to the number of available CPU cores.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }
}

#[async_trait]
impl DataSource for MultiLayoutDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Option<Precision<u64>> {
        let mut sum: u64 = 0;
        let mut opened_count: u64 = 0;
        let mut deferred_count: u64 = 0;

        for child in &self.children {
            match child {
                MultiLayoutChild::Opened(reader) => {
                    opened_count += 1;
                    sum = sum.saturating_add(reader.row_count());
                }
                MultiLayoutChild::Deferred(_) => {
                    deferred_count += 1;
                }
            }
        }

        let total_count = opened_count + deferred_count;
        if total_count == 0 {
            return Some(Precision::exact(0u64));
        }

        if deferred_count == 0 {
            Some(Precision::exact(sum))
        } else if opened_count > 0 {
            let avg = sum / opened_count;
            let extrapolated = avg.saturating_mul(total_count);
            Some(Precision::inexact(extrapolated))
        } else {
            None
        }
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("MultiLayoutDataSource partitions are not yet serializable")
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let mut ready = VecDeque::new();
        let mut deferred = VecDeque::new();

        for child in &self.children {
            match child {
                MultiLayoutChild::Opened(reader) => ready.push_back(Arc::clone(reader)),
                MultiLayoutChild::Deferred(factory) => deferred.push_back(Arc::clone(factory)),
            }
        }

        let dtype = scan_request.projection.return_dtype(&self.dtype)?;

        Ok(Box::new(MultiLayoutScan {
            session: self.session.clone(),
            dtype,
            request: scan_request,
            ready,
            deferred,
            handle: self.session.handle(),
            concurrency: self.concurrency,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

struct MultiLayoutScan {
    session: VortexSession,
    dtype: DType,
    request: ScanRequest,
    ready: VecDeque<LayoutReaderRef>,
    deferred: VecDeque<Arc<dyn LayoutReaderFactory>>,
    handle: vortex_io::runtime::Handle,
    concurrency: usize,
}

impl DataSourceScan for MultiLayoutScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Option<Precision<usize>> {
        let count = self.ready.len() + self.deferred.len();
        if self.deferred.is_empty() {
            Some(Precision::exact(count))
        } else {
            Some(Precision::inexact(count))
        }
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        let Self {
            session,
            dtype: _,
            request,
            ready,
            deferred,
            handle,
            concurrency,
        } = *self;

        let ordered = request.ordered;

        // Pre-opened readers are immediately available.
        let ready_stream = stream::iter(ready).map(Ok);

        // Deferred readers are opened concurrently via spawned tasks.
        // When ordered, we use `buffered` to preserve the original partition order.
        // When unordered, we use `buffer_unordered` to yield partitions as they open.
        let spawned = stream::iter(deferred).map(move |factory| {
            handle.spawn(async move {
                factory
                    .open()
                    .instrument(tracing::info_span!("LayoutReaderFactory::open"))
                    .await
            })
        });

        let deferred_stream = if ordered {
            spawned
                .buffered(concurrency)
                .filter_map(|result| async move {
                    match result {
                        Ok(Some(reader)) => Some(Ok(reader)),
                        Ok(None) => None,
                        Err(e) => Some(Err(e)),
                    }
                })
                .boxed()
        } else {
            spawned
                .buffer_unordered(concurrency)
                .filter_map(|result| async move {
                    match result {
                        Ok(Some(reader)) => Some(Ok(reader)),
                        Ok(None) => None,
                        Err(e) => Some(Err(e)),
                    }
                })
                .boxed()
        };

        // For each reader (ready or just-opened), generate a partition.
        // Partition generation is synchronous (just creates structs with row ranges), so
        // `flat_map` is appropriate here. The real I/O work happens when `execute()` is called.
        ready_stream
            .chain(deferred_stream)
            .flat_map(move |reader_result| match reader_result {
                Ok(reader) => reader_partition(reader, session.clone(), request.clone()),
                Err(e) => stream::once(async move { Err(e) }).boxed(),
            })
            .boxed()
    }
}

/// Generates a partition stream for a single layout reader.
///
/// Checks file-level pruning first (via `pruning_evaluation`). If the filter proves no rows
/// can match, returns an empty stream. Otherwise, yields a single partition covering the
/// reader's full row range.
fn reader_partition(
    reader: LayoutReaderRef,
    session: VortexSession,
    request: ScanRequest,
) -> PartitionStream {
    let row_count = reader.row_count();
    let row_range = request.row_range.clone().unwrap_or(0..row_count);

    // Check file-level pruning: if the filter can be proven false for the entire row range
    // using file-level statistics, skip this reader entirely.
    if let Some(ref filter) = request.filter {
        let mask_len = usize::try_from(row_range.end - row_range.start).unwrap_or(usize::MAX);
        let mask = Mask::new_true(mask_len);
        if let Ok(pruning_future) = reader.pruning_evaluation(&row_range, filter, mask)
            && let Some(Ok(result_mask)) = pruning_future.now_or_never()
            && result_mask.all_false()
        {
            return stream::empty().boxed();
        }
    }

    stream::once(async move {
        Ok(Box::new(MultiLayoutPartition {
            reader,
            session,
            request: ScanRequest {
                row_range: Some(row_range),
                ..request
            },
        }) as PartitionRef)
    })
    .boxed()
}

/// A partition backed by a single [`LayoutReaderRef`] and a row range.
///
/// On `execute()`, creates a [`ScanBuilder`][crate::ScanBuilder] over the row range, enabling
/// internal I/O pipelining and split-level parallelism within the reader.
struct MultiLayoutPartition {
    reader: LayoutReaderRef,
    session: VortexSession,
    request: ScanRequest,
}

impl Partition for MultiLayoutPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn row_count(&self) -> Option<Precision<u64>> {
        let row_range = self.request.row_range.as_ref()?;
        let row_count = row_range.end - row_range.start;
        let row_count = self.request.selection.row_count(row_count);
        let row_count = self
            .request
            .limit
            .map_or(row_count, |limit| row_count.min(limit));

        Some(if self.request.filter.is_some() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        })
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        None
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let request = self.request;
        let mut builder = ScanBuilder::new(self.session, self.reader)
            .with_selection(request.selection)
            .with_projection(request.projection)
            .with_some_filter(request.filter)
            .with_some_limit(request.limit)
            .with_ordered(request.ordered);

        if let Some(row_range) = request.row_range {
            builder = builder.with_row_range(row_range);
        }

        let dtype = builder.dtype()?;
        let stream = builder.into_stream()?;

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}
