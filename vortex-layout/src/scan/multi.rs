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
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::stream;
use itertools::Itertools;
use tracing::Instrument;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_scan::pool::PartitionPool;
use vortex_scan::pool::ScanPoolRef;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::layout::Empty;
use crate::scan::layout::group_contiguous;
use crate::scan::limit::RowLimit;
use crate::scan::pool::FileWork;
use crate::scan::pool::PiecePartition;
use crate::scan::pool::PlannedSplits;
use crate::scan::pool::PoolFile;
use crate::scan::pool::PoolOptions;
use crate::scan::pool::PrepareFile;
use crate::scan::pool::PreparedScan;
use crate::scan::pool::open_pool;
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
#[derive(Clone)]
pub struct MultiLayoutDataSource {
    dtype: DType,
    session: VortexSession,
    children: Arc<[MultiLayoutChild]>,
    concurrency: usize,
}

pub enum MultiLayoutChild {
    Opened {
        reader: LayoutReaderRef,
        /// On-storage file size in bytes, if known from the listing metadata.
        byte_size: Option<u64>,
    },
    Deferred {
        factory: Arc<dyn LayoutReaderFactory>,
        /// On-storage file size in bytes, if known from the listing metadata.
        byte_size: Option<u64>,
    },
}

impl MultiLayoutChild {
    /// On-storage file size in bytes for this child, if known.
    pub fn byte_size(&self) -> Option<u64> {
        match self {
            MultiLayoutChild::Opened { byte_size, .. } => *byte_size,
            MultiLayoutChild::Deferred { byte_size, .. } => *byte_size,
        }
    }
}

impl MultiLayoutDataSource {
    /// Creates a multi-layout data source with the first reader pre-opened.
    ///
    /// The first reader determines the dtype. Remaining readers are opened lazily during
    /// scanning via their factories. `byte_sizes` carries the on-storage file size in bytes for
    /// each child (first followed by remaining); pass `None` for entries where the size is
    /// unknown. Must be empty or have length `1 + remaining.len()`.
    pub fn new_with_first(
        first: LayoutReaderRef,
        remaining: Vec<Arc<dyn LayoutReaderFactory>>,
        byte_sizes: Vec<Option<u64>>,
        session: &VortexSession,
    ) -> Self {
        let dtype = first.dtype().clone();
        let concurrency = get_available_parallelism().unwrap_or(DEFAULT_CONCURRENCY);

        let total = 1 + remaining.len();
        let mut sizes = byte_sizes;
        if sizes.is_empty() {
            sizes = vec![None; total];
        }
        debug_assert_eq!(
            sizes.len(),
            total,
            "byte_sizes length must match the number of children"
        );

        let mut children = Vec::with_capacity(total);
        let mut sizes_iter = sizes.into_iter();
        let first_size = sizes_iter.next().unwrap_or(None);
        children.push(MultiLayoutChild::Opened {
            reader: first,
            byte_size: first_size,
        });
        children.extend(
            remaining
                .into_iter()
                .zip_eq(sizes_iter)
                .map(|(factory, byte_size)| MultiLayoutChild::Deferred { factory, byte_size }),
        );

        Self {
            dtype,
            session: session.clone(),
            children: children.into(),
            concurrency,
        }
    }

    /// Creates a multi-layout data source where all children are deferred.
    ///
    /// The dtype must be provided externally since there is no pre-opened reader to infer it
    /// from. This avoids eagerly opening any file when the schema is already known (e.g. from
    /// a catalog or a prior scan). `byte_sizes` carries the on-storage file size in bytes for
    /// each factory; pass `None` for entries where the size is unknown. Must be empty or have
    /// the same length as `factories`.
    pub fn new_deferred(
        dtype: DType,
        factories: Vec<Arc<dyn LayoutReaderFactory>>,
        byte_sizes: Vec<Option<u64>>,
        session: &VortexSession,
    ) -> Self {
        let concurrency = get_available_parallelism().unwrap_or(DEFAULT_CONCURRENCY);

        let mut sizes = byte_sizes;
        if sizes.is_empty() {
            sizes = vec![None; factories.len()];
        }
        debug_assert_eq!(
            sizes.len(),
            factories.len(),
            "byte_sizes length must match the number of factories"
        );

        Self {
            dtype,
            session: session.clone(),
            children: factories
                .into_iter()
                .zip_eq(sizes)
                .map(|(factory, byte_size)| MultiLayoutChild::Deferred { factory, byte_size })
                .collect(),
            concurrency,
        }
    }

    pub fn children(&self) -> &[MultiLayoutChild] {
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

    fn row_count(&self) -> Precision<u64> {
        let mut sum: u64 = 0;
        let mut opened_count: u64 = 0;
        let mut deferred_count: u64 = 0;

        for child in self.children.iter() {
            match child {
                MultiLayoutChild::Opened { reader, .. } => {
                    opened_count += 1;
                    sum = sum.saturating_add(reader.row_count());
                }
                MultiLayoutChild::Deferred { .. } => {
                    deferred_count += 1;
                }
            }
        }

        let total_count = opened_count + deferred_count;
        if total_count == 0 {
            return Precision::exact(0u64);
        }

        if deferred_count == 0 {
            Precision::exact(sum)
        } else if let Some(avg) = sum.checked_div(opened_count) {
            let extrapolated = avg.saturating_mul(total_count);
            Precision::inexact(extrapolated)
        } else {
            Precision::Absent
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        let total_count = self.children.len() as u64;
        if total_count == 0 {
            return Precision::exact(0u64);
        }

        let mut sum: u64 = 0;
        let mut known_count: u64 = 0;
        for child in self.children.iter() {
            if let Some(size) = child.byte_size() {
                sum = sum.saturating_add(size);
                known_count += 1;
            }
        }

        if known_count == 0 {
            return Precision::Absent;
        }

        if known_count == total_count {
            Precision::exact(sum)
        } else {
            let avg = sum / known_count;
            let extrapolated = avg.saturating_mul(total_count);
            Precision::inexact(extrapolated)
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

        for child in self.children.iter() {
            match child {
                MultiLayoutChild::Opened { reader, .. } => ready.push_back(Arc::clone(reader)),
                MultiLayoutChild::Deferred { factory, .. } => {
                    deferred.push_back(Arc::clone(factory))
                }
            }
        }

        let request = BoundScanRequest::try_new(scan_request, &self.dtype)?;
        let dtype = request.projection.dtype().clone();

        // Only unordered scans share a limit across external partitions: reservation order is
        // completion order, which an ordered scan cannot accept. Ordered partitions each apply the
        // limit locally and the engine trims the concatenated result.
        let row_limit = (!request.ordered)
            .then(|| request.limit.map(RowLimit::new))
            .flatten();

        Ok(Box::new(MultiLayoutScan {
            session: self.session.clone(),
            source_dtype: self.dtype.clone(),
            dtype,
            request,
            row_limit,
            ready,
            deferred,
            concurrency: self.concurrency,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

#[derive(Clone)]
struct BoundScanRequest {
    projection: BoundExpression,
    filter: SharedVortexResult<Option<BoundExpression>>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    partition_selection: Selection,
    partition_range: Option<Range<u64>>,
    ordered: bool,
    limit: Option<u64>,
    split_concurrency: Option<NonZeroUsize>,
}

impl BoundScanRequest {
    fn try_new(request: ScanRequest, dtype: &DType) -> VortexResult<Self> {
        let ScanRequest {
            projection,
            filter,
            row_range,
            selection,
            partition_selection,
            partition_range,
            ordered,
            limit,
            split_concurrency,
        } = request;

        Ok(Self {
            projection: projection.bind(dtype)?.optimize_recursive()?,
            filter: filter
                .map(|expr| expr.bind(dtype)?.optimize_recursive())
                .transpose()
                .map_err(Arc::new),
            row_range,
            selection,
            partition_selection,
            partition_range,
            ordered,
            limit,
            split_concurrency,
        })
    }

    /// Whether the scan filters rows, so its output row count is unknown up front. A filter
    /// that failed to bind counts as one.
    fn has_filter(&self) -> bool {
        !matches!(self.filter, Ok(None))
    }

    /// Whether the request selects partitions, which only the partitions path applies.
    fn selects_partitions(&self) -> bool {
        self.partition_range.is_some() || !matches!(self.partition_selection, Selection::All)
    }
}

struct MultiLayoutScan {
    session: VortexSession,
    source_dtype: DType,
    dtype: DType,
    request: BoundScanRequest,
    row_limit: Option<RowLimit>,
    ready: VecDeque<LayoutReaderRef>,
    deferred: VecDeque<Arc<dyn LayoutReaderFactory>>,
    concurrency: usize,
}

impl MultiLayoutScan {
    /// The scan as files to split or open. Files keep their partition order.
    fn into_split_scan(self) -> MultiLayoutSplitScan {
        let children = self
            .ready
            .into_iter()
            .map(ScanChild::Opened)
            .chain(self.deferred.into_iter().map(ScanChild::Deferred))
            .collect();
        MultiLayoutSplitScan {
            context: Arc::new(ScanContext {
                session: self.session,
                source_dtype: self.source_dtype,
                dtype: self.dtype,
                request: self.request,
            }),
            children,
            index: 0,
        }
    }
}

impl DataSourceScan for MultiLayoutScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        let count = self.ready.len() + self.deferred.len();
        if self.deferred.is_empty() {
            Precision::exact(count)
        } else {
            Precision::inexact(count)
        }
    }

    fn row_count(&self) -> Precision<u64> {
        let estimate = estimate_row_count(
            &self.request,
            self.ready
                .iter()
                .map(|reader| file_row_count(&self.request, reader)),
            self.ready.len() + self.deferred.len(),
        );
        if self.request.selects_partitions() {
            estimate.into_inexact()
        } else {
            estimate
        }
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        if self.request.selects_partitions() {
            return Ok(vec![self]);
        }
        Box::new(self.into_split_scan()).split(n)
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        if self.request.selects_partitions() {
            return Ok(Arc::new(PartitionPool::new(self)));
        }
        Box::new(self.into_split_scan()).open()
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        let Self {
            session,
            source_dtype,
            dtype: _,
            request,
            row_limit,
            ready,
            deferred,
            concurrency,
        } = *self;
        let handle = session.handle();

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
            .enumerate()
            .flat_map(move |(i, reader_result)| match reader_result {
                Ok(reader) => reader_partition(
                    i,
                    reader,
                    session.clone(),
                    &source_dtype,
                    request.clone(),
                    row_limit.clone(),
                ),
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
    partition_idx: usize,
    reader: LayoutReaderRef,
    session: VortexSession,
    source_dtype: &DType,
    request: BoundScanRequest,
    row_limit: Option<RowLimit>,
) -> PartitionStream {
    if reader.dtype() != source_dtype {
        let error = vortex_err!(
            "Multi-layout reader dtype mismatch: expected {}, got {}",
            source_dtype,
            reader.dtype()
        );
        return stream::once(async move { Err(error) }).boxed();
    }

    let row_count = reader.row_count();
    let row_range = request.row_range.clone().unwrap_or(0..row_count);

    let partition_idx_u64: u64 = partition_idx as u64;
    if let Some(range) = &request.partition_range
        && !range.contains(&partition_idx_u64)
    {
        return stream::empty().boxed();
    };
    match &request.partition_selection {
        Selection::IncludeByIndex(buffer)
            if buffer.as_slice().binary_search(&partition_idx_u64).is_err() =>
        {
            return stream::empty().boxed();
        }
        Selection::ExcludeByIndex(buffer)
            if buffer.as_slice().binary_search(&partition_idx_u64).is_ok() =>
        {
            return stream::empty().boxed();
        }
        _ => {}
    };

    // Check file-level pruning: if the filter can be proven false for the entire row range
    // using file-level statistics, skip this reader entirely.
    if let Ok(Some(filter)) = &request.filter {
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
            request: BoundScanRequest {
                row_range: Some(row_range),
                ..request
            },
            row_limit,
            index: partition_idx,
        }) as PartitionRef)
    })
    .boxed()
}

/// A partition backed by a single [`LayoutReaderRef`] and a row range.
///
/// On `execute()`, creates a [`ScanBuilder`] over the row range, enabling
/// internal I/O pipelining and split-level parallelism within the reader.
struct MultiLayoutPartition {
    reader: LayoutReaderRef,
    session: VortexSession,
    request: BoundScanRequest,
    row_limit: Option<RowLimit>,
    index: usize,
}

impl Partition for MultiLayoutPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        let Some(row_range) = self.request.row_range.as_ref() else {
            return Precision::Absent;
        };
        let row_count = self.request.selection.row_count_in_range(row_range);
        let row_count = self
            .request
            .limit
            .map_or(row_count, |limit| row_count.min(limit));

        let has_filter = match &self.request.filter {
            Ok(filter) => filter.is_some(),
            Err(_) => true,
        };
        if has_filter || self.row_limit.is_some() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let request = self.request;
        let filter = request.filter?;
        let mut builder = ScanBuilder::new(self.session, self.reader)
            .with_selection(request.selection)
            .with_projection(request.projection)
            .with_some_filter(filter)
            .with_ordered(request.ordered)
            .with_some_max_splits_in_flight(request.split_concurrency);
        builder = match self.row_limit {
            Some(row_limit) => builder.with_shared_limit(row_limit),
            None => builder.with_some_limit(request.limit),
        };

        if let Some(row_range) = request.row_range {
            builder = builder.with_row_range(row_range);
        }

        let dtype = builder.dtype()?;
        let stream = builder.into_stream()?.boxed();

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

/// The rows `request` selects from an opened file, before any filter or limit.
fn file_row_count(request: &BoundScanRequest, reader: &LayoutReaderRef) -> u64 {
    let row_range = request
        .row_range
        .clone()
        .unwrap_or_else(|| 0..reader.row_count());
    request.selection.row_count_in_range(&row_range)
}

/// Estimates the rows a scan of `files` files returns from the rows of the files whose rows are
/// known, extrapolating to the rest.
fn estimate_row_count(
    request: &BoundScanRequest,
    known: impl Iterator<Item = u64>,
    files: usize,
) -> Precision<u64> {
    let (rows, known_files) = known.fold((0u64, 0u64), |(rows, count), file_rows| {
        (rows.saturating_add(file_rows), count + 1)
    });
    let files = files as u64;
    let (rows, is_exact) = if known_files == files {
        (rows, !request.has_filter())
    } else if let Some(average) = rows.checked_div(known_files) {
        (average.saturating_mul(files), false)
    } else {
        return Precision::Absent;
    };
    let rows = request.limit.map_or(rows, |limit| rows.min(limit));
    if is_exact {
        Precision::exact(rows)
    } else {
        Precision::inexact(rows)
    }
}

/// What every piece of a split or opened multi-layout scan shares.
struct ScanContext {
    session: VortexSession,
    source_dtype: DType,
    dtype: DType,
    request: BoundScanRequest,
}

impl ScanContext {
    /// Plans the scan of one opened file: checks its dtype, prunes it on statistics, and
    /// prepares its splits. Returns `None` when statistics reject the file. Reads metadata only.
    fn plan_file(&self, reader: LayoutReaderRef) -> VortexResult<Option<PlannedSplits>> {
        if reader.dtype() != &self.source_dtype {
            return Err(vortex_err!(
                "Multi-layout reader dtype mismatch: expected {}, got {}",
                self.source_dtype,
                reader.dtype()
            ));
        }
        let filter = self.request.filter.clone()?;
        let row_range = self
            .request
            .row_range
            .clone()
            .unwrap_or_else(|| 0..reader.row_count());

        if let Some(filter) = &filter {
            let mask_len = usize::try_from(row_range.end - row_range.start).unwrap_or(usize::MAX);
            if let Ok(pruning) =
                reader.pruning_evaluation(&row_range, filter, Mask::new_true(mask_len))
                && let Some(Ok(mask)) = pruning.now_or_never()
                && mask.all_false()
            {
                return Ok(None);
            }
        }

        let scan = ScanBuilder::new(self.session.clone(), reader)
            .with_row_range(row_range)
            .with_selection(self.request.selection.clone())
            .with_projection(self.request.projection.clone())
            .with_some_filter(filter)
            .with_ordered(self.request.ordered)
            .prepare()?;
        let ranges = scan.split_ranges(None);
        Ok(Some((Arc::new(scan), ranges)))
    }

    /// Starts reading an opened file for one execution.
    fn start_file(&self, reader: LayoutReaderRef) -> VortexResult<FileWork> {
        match self.plan_file(reader)? {
            Some((scan, ranges)) => FileWork::start(scan, ranges),
            None => Ok(FileWork::empty()),
        }
    }
}

/// One file of a split or opened multi-layout scan.
enum ScanChild {
    Opened(LayoutReaderRef),
    /// An opened file whose splits are known: all of them, or the share of one piece.
    Planned {
        scan: Arc<PreparedScan>,
        ranges: Vec<Range<u64>>,
    },
    Deferred(Arc<dyn LayoutReaderFactory>),
}

/// A multi-layout scan that was split or opened, or a piece cut from one. Its units are files,
/// then each file's splits.
struct MultiLayoutSplitScan {
    context: Arc<ScanContext>,
    children: Vec<ScanChild>,
    /// The index of the piece's single partition.
    index: usize,
}

impl MultiLayoutSplitScan {
    /// Splits a planned file's ranges into `pieces` contiguous groups.
    fn split_child(child: ScanChild, pieces: usize) -> Vec<ScanChild> {
        match child {
            ScanChild::Planned { scan, ranges } if pieces > 1 => group_contiguous(
                ranges,
                NonZeroUsize::new(pieces).unwrap_or(NonZeroUsize::MIN),
            )
            .into_iter()
            .map(|ranges| ScanChild::Planned {
                scan: Arc::clone(&scan),
                ranges,
            })
            .collect(),
            child => vec![child],
        }
    }
}

impl DataSourceScan for MultiLayoutSplitScan {
    fn dtype(&self) -> &DType {
        &self.context.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        let index = self.index;
        PiecePartition::stream(self, index)
    }

    fn row_count(&self) -> Precision<u64> {
        let request = &self.context.request;
        let known = self.children.iter().filter_map(|child| match child {
            ScanChild::Opened(reader) => Some(file_row_count(request, reader)),
            ScanChild::Planned { scan, ranges } => Some(
                ranges
                    .iter()
                    .map(|range| scan.selected_row_count(range))
                    .sum(),
            ),
            ScanChild::Deferred(_) => None,
        });
        estimate_row_count(request, known, self.children.len())
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        if self.context.request.limit.is_some() || n.get() == 1 || self.children.is_empty() {
            return Ok(vec![self]);
        }
        let MultiLayoutSplitScan {
            context, children, ..
        } = *self;

        // Plan the opened files so they can be cut on their splits. Deferred files stay whole:
        // splitting never opens a file.
        let mut planned = Vec::with_capacity(children.len());
        for child in children {
            match child {
                ScanChild::Opened(reader) => {
                    if let Some((scan, ranges)) = context.plan_file(reader)? {
                        planned.push(ScanChild::Planned { scan, ranges });
                    }
                }
                child => planned.push(child),
            }
        }
        if planned.is_empty() {
            return Ok(vec![Box::new(MultiLayoutSplitScan {
                context,
                children: planned,
                index: 0,
            })]);
        }

        let n = n.get();
        let pieces: Vec<Vec<ScanChild>> = if n <= planned.len() {
            // Group whole files into contiguous pieces.
            let files = planned.len();
            let mut pieces: Vec<Vec<ScanChild>> = Vec::new();
            let mut current = None;
            for (index, child) in planned.into_iter().enumerate() {
                let piece = index * n / files;
                if current != Some(piece) {
                    current = Some(piece);
                    pieces.push(Vec::new());
                }
                if let Some(last) = pieces.last_mut() {
                    last.push(child);
                }
            }
            pieces
        } else {
            // One piece per file, and the pieces left over cut planned files on their splits, in
            // proportion to their rows.
            let extra = (n - planned.len()) as u128;
            let rows = |child: &ScanChild| match child {
                ScanChild::Planned { ranges, .. } => ranges
                    .iter()
                    .map(|range| u128::from(range.end - range.start))
                    .sum::<u128>(),
                _ => 0,
            };
            let total = planned.iter().map(rows).sum::<u128>().max(1);
            planned
                .into_iter()
                .flat_map(|child| {
                    let pieces = 1 + usize::try_from(extra * rows(&child) / total).unwrap_or(0);
                    Self::split_child(child, pieces)
                })
                .map(|child| vec![child])
                .collect()
        };

        Ok(pieces
            .into_iter()
            .enumerate()
            .map(|(index, children)| {
                Box::new(MultiLayoutSplitScan {
                    context: Arc::clone(&context),
                    children,
                    index,
                }) as DataSourceScanRef
            })
            .collect())
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        let MultiLayoutSplitScan {
            context, children, ..
        } = *self;
        let request = &context.request;
        if request.limit == Some(0) {
            return Box::new(Empty::new(context.dtype.clone(), 0)).open();
        }

        let mut files = Vec::with_capacity(children.len());
        for child in children {
            files.push(match child {
                ScanChild::Opened(reader) => PoolFile::Opened(context.start_file(reader)?),
                ScanChild::Planned { scan, ranges } => {
                    PoolFile::Opened(FileWork::start(scan, ranges)?)
                }
                ScanChild::Deferred(factory) => PoolFile::Deferred(factory),
            });
        }

        let options = PoolOptions {
            dtype: context.dtype.clone(),
            ordered: request.ordered,
            limit: request.limit,
            has_filter: request.has_filter(),
            splits_in_flight: request.split_concurrency,
        };
        let prepare: PrepareFile = Arc::new(move |reader| context.start_file(reader));
        Ok(open_pool(options, files, Some(prepare)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::TryStreamExt;
    use rstest::rstest;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_scan::DataSource;
    use vortex_scan::ScanRequest;

    use super::*;
    use crate::scan::test::InFlight;
    use crate::scan::test::TestLayoutReader;
    use crate::scan::test::collect_scan_values;
    use crate::scan::test::new_session;
    use crate::scan::test::session_with_handle;

    struct NeverOpened;

    #[async_trait]
    impl LayoutReaderFactory for NeverOpened {
        async fn open(&self) -> VortexResult<Option<LayoutReaderRef>> {
            unreachable!("byte_size must not open readers")
        }
    }

    fn deferred_source(byte_sizes: Vec<Option<u64>>) -> MultiLayoutDataSource {
        let factories: Vec<Arc<dyn LayoutReaderFactory>> = byte_sizes
            .iter()
            .map(|_| Arc::new(NeverOpened) as _)
            .collect();
        MultiLayoutDataSource::new_deferred(
            DType::Bool(Nullability::NonNullable),
            factories,
            byte_sizes,
            &new_session(),
        )
    }

    #[rstest]
    #[case::all_known(vec![Some(10), Some(20), Some(30)], Precision::exact(60u64))]
    #[case::some_known_extrapolates(vec![Some(10), None, Some(30)], Precision::inexact(60u64))]
    #[case::none_known(vec![None, None], Precision::Absent)]
    #[case::no_children(vec![], Precision::exact(0u64))]
    fn byte_size_precision(#[case] sizes: Vec<Option<u64>>, #[case] expected: Precision<u64>) {
        assert_eq!(deferred_source(sizes).byte_size(), expected);
    }

    #[test]
    fn filter_binding_errors_are_deferred() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let request = ScanRequest {
            filter: Some(eq(root(), lit(67_i32))),
            ..ScanRequest::default()
        };

        let request = BoundScanRequest::try_new(request, &dtype)?;

        assert_eq!(request.projection.dtype(), &dtype);
        assert!(request.filter.is_err());
        Ok(())
    }

    struct StaticReaderFactory {
        reader: LayoutReaderRef,
    }

    #[async_trait]
    impl LayoutReaderFactory for StaticReaderFactory {
        async fn open(&self) -> VortexResult<Option<LayoutReaderRef>> {
            Ok(Some(Arc::clone(&self.reader)))
        }
    }

    /// An unordered limit is shared by every file of the scan, so the files together return no
    /// more than the limit even though each one is scanned independently.
    #[test]
    fn unordered_limit_is_shared_across_readers() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let first: LayoutReaderRef = Arc::new(TestLayoutReader::new(2));
        let second: LayoutReaderRef = Arc::new(TestLayoutReader::new(2).with_base(10));
        let source = MultiLayoutDataSource::new_with_first(
            first,
            vec![Arc::new(StaticReaderFactory { reader: second })],
            vec![],
            &session,
        );

        let scan = runtime.block_on(source.scan(ScanRequest {
            filter: Some(root()),
            limit: Some(3),
            ordered: false,
            ..Default::default()
        }))?;
        let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
        assert_eq!(partitions.len(), 2);

        let mut values = Vec::new();
        for partition in partitions {
            values.extend(collect_scan_values(
                runtime.block_on_stream(partition.execute()?),
            )?);
        }

        // Three of the four rows, whichever partition reserved them first.
        assert_eq!(values.len(), 3);
        Ok(())
    }

    /// Scans 64 one-row splits, returning the rows and the most splits read at once.
    fn scan_with_split_concurrency(
        split_concurrency: Option<NonZeroUsize>,
    ) -> VortexResult<(Vec<i32>, usize)> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let in_flight = Arc::new(InFlight::default());
        let reader: LayoutReaderRef = Arc::new(
            TestLayoutReader::new(64)
                .with_split_size(1)
                .with_in_flight(Arc::clone(&in_flight)),
        );
        let source = MultiLayoutDataSource::new_with_first(reader, vec![], vec![], &session);

        let scan = runtime.block_on(source.scan(ScanRequest {
            ordered: true,
            split_concurrency,
            ..Default::default()
        }))?;
        let mut values = Vec::new();
        for partition in runtime.block_on(scan.partitions().try_collect::<Vec<_>>())? {
            values.extend(collect_scan_values(
                runtime.block_on_stream(partition.execute()?),
            )?);
        }
        Ok((values, in_flight.peak()))
    }

    /// A per-worker concurrency set after the bound leaves the bound in place.
    #[test]
    fn a_later_per_worker_concurrency_keeps_the_bound() {
        let reader: LayoutReaderRef = Arc::new(TestLayoutReader::new(1));
        let builder = ScanBuilder::new(new_session(), reader)
            .with_some_max_splits_in_flight(NonZeroUsize::new(2))
            .with_concurrency(8);

        assert_eq!(builder.concurrency(), 8);
        assert_eq!(builder.max_splits_in_flight(), NonZeroUsize::new(2));
    }

    /// The scan reads at most `split_concurrency` splits at once, however many cores there are.
    #[rstest]
    #[case(1)]
    #[case(2)]
    fn split_concurrency_bounds_the_splits_read_at_once(#[case] max: usize) -> VortexResult<()> {
        let (values, peak) = scan_with_split_concurrency(NonZeroUsize::new(max))?;
        let (default_values, default_peak) = scan_with_split_concurrency(None)?;

        assert_eq!(values, (0..64).collect::<Vec<_>>());
        assert_eq!(values, default_values);
        assert_eq!(peak, max, "splits read at once");
        assert!(
            default_peak > max,
            "the default read {default_peak} at once"
        );
        Ok(())
    }
}

#[cfg(test)]
mod pool_tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;

    use async_trait::async_trait;
    use futures::StreamExt;
    use futures::future::poll_fn;
    use futures::task::ArcWake;
    use futures::task::AtomicWaker;
    use futures::task::noop_waker_ref;
    use futures::task::waker;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::stream::SendableArrayStream;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_scan::DataSource;
    use vortex_scan::ScanRequest;
    use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

    use super::*;
    use crate::scan::test::InFlight;
    use crate::scan::test::TestLayoutReader;
    use crate::scan::test::collect_scan_values;
    use crate::scan::test::new_session;
    use crate::scan::test::read_with_one_worker;
    use crate::scan::test::read_with_workers;
    use crate::scan::test::session_with_handle;

    fn plan(
        source: &MultiLayoutDataSource,
        request: ScanRequest,
    ) -> VortexResult<DataSourceScanRef> {
        futures::executor::block_on(source.scan(request))
    }

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    /// A factory whose open completes only once its gate is released.
    #[derive(Default)]
    struct Gate {
        is_open: AtomicBool,
        waker: AtomicWaker,
    }

    impl Gate {
        fn release(&self) {
            self.is_open.store(true, Ordering::SeqCst);
            self.waker.wake();
        }
    }

    struct GatedFactory {
        reader: LayoutReaderRef,
        gate: Arc<Gate>,
        opens: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LayoutReaderFactory for GatedFactory {
        async fn open(&self) -> VortexResult<Option<LayoutReaderRef>> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            poll_fn(|cx| {
                self.gate.waker.register(cx.waker());
                if self.gate.is_open.load(Ordering::SeqCst) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            })
            .await;
            Ok(Some(Arc::clone(&self.reader)))
        }
    }

    fn gated_factory(reader: TestLayoutReader, gate: &Arc<Gate>) -> Arc<dyn LayoutReaderFactory> {
        Arc::new(GatedFactory {
            reader: Arc::new(reader),
            gate: Arc::clone(gate),
            opens: Arc::default(),
        })
    }

    /// Deferred files of `rows` one-row splits, the `i`th producing `100 * i + row`.
    fn gated_source(files: usize, rows: u64) -> (MultiLayoutDataSource, Vec<Arc<Gate>>) {
        let gates: Vec<_> = (0..files).map(|_| Arc::new(Gate::default())).collect();
        let factories = gates
            .iter()
            .enumerate()
            .map(|(index, gate)| {
                let base = 100 * i32::try_from(index).expect("few files");
                gated_factory(
                    TestLayoutReader::new(rows)
                        .with_split_size(1)
                        .with_base(base),
                    gate,
                )
            })
            .collect();
        let source =
            MultiLayoutDataSource::new_deferred(i32_dtype(), factories, vec![], &new_session());
        (source, gates)
    }

    #[derive(Default)]
    struct WakeFlag(AtomicBool);

    impl ArcWake for WakeFlag {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.store(true, Ordering::SeqCst);
        }
    }

    fn poll_worker(
        worker: &mut SendableArrayStream,
        cx: &mut Context<'_>,
    ) -> Poll<Option<VortexResult<ArrayRef>>> {
        worker.poll_next_unpin(cx)
    }

    /// A deferred file is opened by a worker, and its splits are shared with other workers. A
    /// worker that finds only an open in progress waits, and is woken when the open adds splits.
    #[test]
    fn deferred_file_splits_are_shared_with_waiting_workers() -> VortexResult<()> {
        let (source, gates) = gated_source(1, 16);
        let pool = plan(&source, ScanRequest::default())?.open()?;
        let mut opener = Arc::clone(&pool).worker();
        let mut waiter = Arc::clone(&pool).worker();
        let mut noop = Context::from_waker(noop_waker_ref());
        let flag = Arc::new(WakeFlag::default());
        let flag_waker = waker(Arc::clone(&flag));
        let mut flagged = Context::from_waker(&flag_waker);

        // The opener takes the only file and blocks on its open; the waiter finds nothing to take
        // but an open in progress, so it waits.
        assert!(poll_worker(&mut opener, &mut noop).is_pending());
        assert!(poll_worker(&mut waiter, &mut flagged).is_pending());

        gates[0].release();
        let Poll::Ready(Some(first)) = poll_worker(&mut opener, &mut noop) else {
            panic!("the opener reads a split of the file it opened");
        };
        assert!(flag.0.load(Ordering::SeqCst), "the waiter is woken");

        let waiter_values = collect_scan_values(futures::executor::block_on_stream(waiter))?;
        let opener_values = collect_scan_values(
            std::iter::once(first).chain(futures::executor::block_on_stream(opener)),
        )?;
        assert!(
            !waiter_values.is_empty(),
            "the waiter reads the opened file's splits"
        );
        let mut values = [opener_values, waiter_values].concat();
        values.sort_unstable();
        assert_eq!(values, (0..16).collect::<Vec<_>>());
        Ok(())
    }

    /// An ordered worker emits files in file order, even when a later file opens first.
    #[test]
    fn ordered_worker_emits_files_in_order() -> VortexResult<()> {
        let (source, gates) = gated_source(3, 3);
        let pool = plan(
            &source,
            ScanRequest {
                ordered: true,
                ..Default::default()
            },
        )?
        .open()?;
        let mut worker = pool.worker();
        let mut noop = Context::from_waker(noop_waker_ref());
        assert!(poll_worker(&mut worker, &mut noop).is_pending());
        for gate in gates.iter().rev() {
            gate.release();
            if !Arc::ptr_eq(gate, &gates[0]) {
                assert!(poll_worker(&mut worker, &mut noop).is_pending());
            }
        }
        let values = collect_scan_values(futures::executor::block_on_stream(worker))?;
        assert_eq!(values, [0, 1, 2, 100, 101, 102, 200, 201, 202]);
        Ok(())
    }

    /// A limit is shared by every worker and file of an unordered pool.
    #[test]
    fn pool_limit_is_shared_across_readers() -> VortexResult<()> {
        let first: LayoutReaderRef = Arc::new(TestLayoutReader::new(4).with_split_size(1));
        let gate = Arc::new(Gate::default());
        gate.release();
        let source = MultiLayoutDataSource::new_with_first(
            first,
            vec![gated_factory(
                TestLayoutReader::new(4).with_split_size(1).with_base(10),
                &gate,
            )],
            vec![],
            &new_session(),
        );

        let pool = plan(
            &source,
            ScanRequest {
                filter: Some(root()),
                limit: Some(3),
                ..Default::default()
            },
        )?
        .open()?;
        let values = read_with_workers(&pool, 3)?.concat();

        assert_eq!(values.len(), 3);
        Ok(())
    }

    /// Splitting cuts opened files on their splits and keeps deferred files whole, in order.
    #[test]
    fn split_cuts_opened_files_and_keeps_deferred_files_whole() -> VortexResult<()> {
        let first: LayoutReaderRef = Arc::new(TestLayoutReader::new(6).with_split_size(2));
        let opens = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Gate::default());
        gate.release();
        let source = MultiLayoutDataSource::new_with_first(
            first,
            vec![Arc::new(GatedFactory {
                reader: Arc::new(TestLayoutReader::new(4).with_split_size(2).with_base(100)),
                gate,
                opens: Arc::clone(&opens),
            })],
            vec![],
            &new_session(),
        );
        let scan = plan(
            &source,
            ScanRequest {
                ordered: true,
                ..Default::default()
            },
        )?;
        let pieces = scan.split(NonZeroUsize::new(4).expect("non-zero"))?;
        assert_eq!(opens.load(Ordering::SeqCst), 0, "splitting opens no file");
        // The opened file's three splits take three pieces; the deferred file stays whole.
        assert_eq!(pieces.len(), 4);

        let mut values = Vec::new();
        for piece in pieces {
            values.extend(read_with_one_worker(piece.open()?)?);
        }
        assert_eq!(values, [0, 1, 2, 3, 4, 5, 100, 101, 102, 103]);
        Ok(())
    }

    /// A request that selects partitions is split and opened by its partitions, so both paths
    /// read the same files.
    #[rstest]
    #[case::range(Some(1..2), None)]
    #[case::selection(None, Some(1))]
    fn selected_partitions_open_by_partition(
        #[case] partition_range: Option<Range<u64>>,
        #[case] selected_partition: Option<u64>,
    ) -> VortexResult<()> {
        let partition_selection = match selected_partition {
            Some(index) => {
                Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter([index]))?)
            }
            None => Selection::All,
        };
        // The partitions path spawns file opens onto the session's runtime.
        let runtime = SingleThreadRuntime::default();
        let gate = Arc::new(Gate::default());
        gate.release();
        let source = MultiLayoutDataSource::new_with_first(
            Arc::new(TestLayoutReader::new(3)),
            vec![gated_factory(TestLayoutReader::new(3).with_base(10), &gate)],
            vec![],
            &session_with_handle(runtime.handle()),
        );
        let request = ScanRequest {
            partition_range,
            partition_selection,
            ordered: true,
            ..Default::default()
        };
        let pieces = plan(&source, request.clone())?.split(NonZeroUsize::new(4).expect("4"))?;
        assert_eq!(pieces.len(), 1);
        let pool = plan(&source, request)?.open()?;
        let values = collect_scan_values(runtime.block_on_stream(pool.worker()))?;
        assert_eq!(values, [10, 11, 12]);
        Ok(())
    }

    /// A worker reads at most `split_concurrency` splits at once.
    #[rstest]
    #[case(1)]
    #[case(2)]
    fn split_concurrency_bounds_each_worker(#[case] max: usize) -> VortexResult<()> {
        let in_flight = Arc::new(InFlight::default());
        let reader: LayoutReaderRef = Arc::new(
            TestLayoutReader::new(64)
                .with_split_size(1)
                .with_in_flight(Arc::clone(&in_flight)),
        );
        let source = MultiLayoutDataSource::new_with_first(reader, vec![], vec![], &new_session());
        let pool = plan(
            &source,
            ScanRequest {
                ordered: true,
                split_concurrency: NonZeroUsize::new(max),
                ..Default::default()
            },
        )?
        .open()?;
        assert_eq!(read_with_one_worker(pool)?, (0..64).collect::<Vec<_>>());
        assert!(in_flight.peak() <= max, "peak {}", in_flight.peak());
        Ok(())
    }
}
