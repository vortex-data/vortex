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
use vortex_scan::pool::ScanStream;
use vortex_scan::pool::split_shared;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::layout::group_contiguous;
use crate::scan::limit::RowLimit;
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
        let mut ready_sizes = VecDeque::new();
        let mut deferred_sizes = VecDeque::new();

        for child in self.children.iter() {
            match child {
                MultiLayoutChild::Opened { reader, byte_size } => {
                    ready.push_back(Arc::clone(reader));
                    ready_sizes.push_back(*byte_size);
                }
                MultiLayoutChild::Deferred { factory, byte_size } => {
                    deferred.push_back(Arc::clone(factory));
                    deferred_sizes.push_back(*byte_size);
                }
            }
        }
        // The scan reads its opened files first, so its sizes follow that order too.
        let byte_sizes = ready_sizes.into_iter().chain(deferred_sizes).collect();

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
            byte_sizes,
            handle: self.session.handle(),
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
}

struct MultiLayoutScan {
    session: VortexSession,
    source_dtype: DType,
    dtype: DType,
    request: BoundScanRequest,
    row_limit: Option<RowLimit>,
    ready: VecDeque<LayoutReaderRef>,
    deferred: VecDeque<Arc<dyn LayoutReaderFactory>>,
    /// The on-storage size of each file, if known: the `ready` files, then the `deferred` ones.
    byte_sizes: VecDeque<Option<u64>>,
    handle: vortex_io::runtime::Handle,
    concurrency: usize,
}

impl DataSourceScan for MultiLayoutScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Splits an unordered scan into `n` clones of its opened partition stream, which share its
    /// files. Cuts an ordered scan into at most `n` contiguous pieces in file order, reading
    /// metadata only.
    ///
    /// With at least `n` files, each ordered piece is a run of whole files. The runs balance the
    /// files' byte sizes when every size is known, and their counts otherwise. With fewer files,
    /// each file gets a share of the `n` pieces in proportion to its size, at least one, and no
    /// piece holds parts of two files. A file that is open is cut at its natural split
    /// boundaries. A file that is not open stays one piece, because cutting it would mean
    /// opening it, so the scan may return fewer than `n` pieces.
    ///
    /// A scan with a limit stays whole, because the limit is one budget across all its files.
    /// So does an ordered one that selects partitions, because partition indices count from its
    /// first file.
    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        let files = self.ready.len() + self.deferred.len();
        if n.get() == 1 || files == 0 || self.request.limit.is_some() {
            return Ok(vec![self]);
        }
        if !self.request.ordered {
            return split_shared(self, n);
        }
        if self.request.partition_range.is_some()
            || !matches!(self.request.partition_selection, Selection::All)
        {
            return Ok(vec![self]);
        }
        let Self {
            session,
            source_dtype,
            dtype,
            request,
            row_limit,
            mut ready,
            mut deferred,
            mut byte_sizes,
            handle,
            concurrency,
        } = *self;

        let weights: Vec<u128> = byte_sizes
            .iter()
            .map(|size| size.map(|size| u128::from(size.max(1))))
            .collect::<Option<_>>()
            .unwrap_or_else(|| vec![1; files]);
        let total = weights.iter().sum::<u128>();
        let n = n.get();

        let piece = |ready: VecDeque<LayoutReaderRef>,
                     deferred: VecDeque<Arc<dyn LayoutReaderFactory>>,
                     byte_sizes: VecDeque<Option<u64>>,
                     request: BoundScanRequest| {
            Box::new(MultiLayoutScan {
                session: session.clone(),
                source_dtype: source_dtype.clone(),
                dtype: dtype.clone(),
                request,
                row_limit: row_limit.clone(),
                ready,
                deferred,
                byte_sizes,
                handle: handle.clone(),
                concurrency,
            }) as DataSourceScanRef
        };

        if files >= n {
            // Each file joins the piece in which its first byte falls.
            let mut runs: Vec<usize> = Vec::with_capacity(n);
            let mut last_group = None;
            let mut before = 0u128;
            for weight in &weights {
                let group = before * n as u128 / total;
                before += weight;
                match runs.last_mut() {
                    Some(run) if last_group == Some(group) => *run += 1,
                    _ => runs.push(1),
                }
                last_group = Some(group);
            }
            return Ok(runs
                .into_iter()
                .map(|take| {
                    let from_ready = take.min(ready.len());
                    piece(
                        ready.drain(..from_ready).collect(),
                        deferred.drain(..take - from_ready).collect(),
                        byte_sizes.drain(..take).collect(),
                        request.clone(),
                    )
                })
                .collect());
        }

        let spare = (n - files) as u128;
        let shares = weights
            .iter()
            .map(|weight| 1 + u64::try_from(weight * spare / total).unwrap_or(0));
        let mut pieces = Vec::with_capacity(n);
        let mut sizes = byte_sizes.into_iter();
        let ready_len = ready.len();
        for (file, share) in shares.enumerate() {
            let size = sizes.next().flatten();
            if file < ready_len {
                let Some(reader) = ready.pop_front() else {
                    break;
                };
                let cuts = (share > 1)
                    .then(|| opened_file_cuts(&session, &reader, &request, share))
                    .flatten();
                match cuts {
                    Some(cuts) => {
                        for rows in cuts {
                            let mut request = request.clone();
                            request.row_range = Some(rows);
                            pieces.push(piece(
                                VecDeque::from([Arc::clone(&reader)]),
                                VecDeque::new(),
                                VecDeque::from([size]),
                                request,
                            ));
                        }
                    }
                    None => pieces.push(piece(
                        VecDeque::from([reader]),
                        VecDeque::new(),
                        VecDeque::from([size]),
                        request.clone(),
                    )),
                }
            } else {
                let Some(factory) = deferred.pop_front() else {
                    break;
                };
                pieces.push(piece(
                    VecDeque::new(),
                    VecDeque::from([factory]),
                    VecDeque::from([size]),
                    request.clone(),
                ));
            }
        }
        Ok(pieces)
    }

    /// Opens the scan into its partition pipeline, which opens up to `concurrency` files ahead
    /// and reads each file as one partition, handing the partitions to workers.
    fn open(self: Box<Self>) -> VortexResult<ScanStream> {
        Ok(ScanStream::from_partitions(self))
    }

    fn partition_count(&self) -> Precision<usize> {
        let count = self.ready.len() + self.deferred.len();
        if self.deferred.is_empty() {
            Precision::exact(count)
        } else {
            Precision::inexact(count)
        }
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
            byte_sizes: _,
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

/// The row ranges of `reader`'s natural splits within `rows` that `request` reads, or `None` if
/// they cannot be planned, such as when the request does not bind to the file. Reads metadata
/// only.
fn file_split_ranges(
    session: &VortexSession,
    reader: &LayoutReaderRef,
    request: &BoundScanRequest,
    rows: Range<u64>,
) -> Option<Vec<Range<u64>>> {
    let filter = request.filter.clone().ok()?;
    let scan = ScanBuilder::new(session.clone(), Arc::clone(reader))
        .with_selection(request.selection.clone())
        .with_projection(request.projection.clone())
        .with_some_filter(filter)
        .prepare()
        .ok()?;
    Some(scan.split_ranges(Some(rows)))
}

/// Cuts the rows of an opened file that `request` reads into at most `share` ranges that cover
/// them in order, at the file's natural split boundaries. `None` if the file cannot be cut.
fn opened_file_cuts(
    session: &VortexSession,
    reader: &LayoutReaderRef,
    request: &BoundScanRequest,
    share: u64,
) -> Option<Vec<Range<u64>>> {
    let row_count = reader.row_count();
    let rows = request.row_range.clone().map_or(0..row_count, |rows| {
        rows.start.min(row_count)..rows.end.min(row_count)
    });
    let share = NonZeroUsize::new(usize::try_from(share).ok()?)?;
    let splits = file_split_ranges(session, reader, request, rows.clone())?;
    let groups = group_contiguous(splits, share);
    if groups.len() <= 1 {
        return None;
    }
    let last = groups.len() - 1;
    Some(
        groups
            .iter()
            .enumerate()
            .filter_map(|(i, group)| {
                // The cuts start and end where `rows` does, so no row is lost at either edge.
                let start = if i == 0 {
                    rows.start
                } else {
                    group.first()?.start
                };
                let end = if i == last {
                    rows.end
                } else {
                    group.last()?.end
                };
                Some(start..end)
            })
            .collect(),
    )
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
            unreachable!("byte_size and split must not open readers")
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

    /// Reads every partition of `piece`, in order.
    fn piece_values(
        runtime: &SingleThreadRuntime,
        piece: DataSourceScanRef,
    ) -> VortexResult<Vec<i32>> {
        let mut values = Vec::new();
        for partition in runtime.block_on(piece.partitions().try_collect::<Vec<_>>())? {
            values.extend(collect_scan_values(
                runtime.block_on_stream(partition.execute()?),
            )?);
        }
        Ok(values)
    }

    /// A source of files that are never opened, whose scans run on `runtime`.
    fn unopened_files(
        runtime: &SingleThreadRuntime,
        byte_sizes: Vec<Option<u64>>,
    ) -> MultiLayoutDataSource {
        let factories = byte_sizes
            .iter()
            .map(|_| Arc::new(NeverOpened) as Arc<dyn LayoutReaderFactory>)
            .collect();
        MultiLayoutDataSource::new_deferred(
            DType::Bool(Nullability::NonNullable),
            factories,
            byte_sizes,
            &session_with_handle(runtime.handle()),
        )
    }

    /// With at least `n` files, splitting hands out runs of whole files in file order, without
    /// opening any.
    #[test]
    fn split_groups_whole_files() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let first: LayoutReaderRef = Arc::new(TestLayoutReader::new(2));
        let rest = (1..5)
            .map(|file| {
                Arc::new(StaticReaderFactory {
                    reader: Arc::new(TestLayoutReader::new(2).with_base(10 * file)),
                }) as Arc<dyn LayoutReaderFactory>
            })
            .collect();
        let source = MultiLayoutDataSource::new_with_first(first, rest, vec![], &session);

        let scan = runtime.block_on(source.scan(ScanRequest {
            ordered: true,
            ..Default::default()
        }))?;
        let pieces = scan.split(NonZeroUsize::try_from(3)?)?;
        let files: Vec<_> = pieces.iter().map(|piece| piece.partition_count()).collect();
        assert_eq!(
            files,
            [
                Precision::inexact(2usize),
                Precision::inexact(2usize),
                Precision::inexact(1usize)
            ]
        );

        let mut values = Vec::new();
        for piece in pieces {
            values.extend(piece_values(&runtime, piece)?);
        }
        assert_eq!(values, [0, 1, 10, 11, 20, 21, 30, 31, 40, 41]);
        Ok(())
    }

    /// Known file sizes balance the runs by bytes rather than by file count. `NeverOpened`
    /// panics if splitting opens a file.
    #[test]
    fn split_balances_whole_files_by_size() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let scan = runtime.block_on(
            unopened_files(&runtime, vec![Some(80), Some(10), Some(10)]).scan(ScanRequest {
                ordered: true,
                ..Default::default()
            }),
        )?;
        let files: Vec<_> = scan
            .split(NonZeroUsize::try_from(2)?)?
            .iter()
            .map(|piece| piece.partition_count())
            .collect();
        assert_eq!(
            files,
            [Precision::inexact(1usize), Precision::inexact(2usize)]
        );
        Ok(())
    }

    /// An ordered split never opens a file to cut it, so an unopened file stays one piece.
    #[test]
    fn ordered_split_keeps_unopened_files_whole() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let scan = runtime.block_on(unopened_files(&runtime, vec![Some(30), Some(10)]).scan(
            ScanRequest {
                ordered: true,
                ..Default::default()
            },
        ))?;
        assert_eq!(scan.split(NonZeroUsize::try_from(4)?)?.len(), 2);
        Ok(())
    }

    /// An opened file is cut at its split boundaries.
    #[test]
    fn split_cuts_an_opened_file_at_splits() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let reader: LayoutReaderRef = Arc::new(TestLayoutReader::new(8).with_split_size(2));
        let source = MultiLayoutDataSource::new_with_first(reader, vec![], vec![], &session);

        let scan = runtime.block_on(source.scan(ScanRequest {
            ordered: true,
            ..Default::default()
        }))?;
        let pieces = scan
            .split(NonZeroUsize::try_from(4)?)?
            .into_iter()
            .map(|piece| piece_values(&runtime, piece))
            .collect::<VortexResult<Vec<_>>>()?;
        assert_eq!(pieces, [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
        Ok(())
    }

    /// The pieces of an unordered scan share its files, so together they read every row once.
    #[test]
    fn unordered_pieces_share_the_files() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let first: LayoutReaderRef = Arc::new(TestLayoutReader::new(2));
        let rest = (1..3)
            .map(|file| {
                Arc::new(StaticReaderFactory {
                    reader: Arc::new(TestLayoutReader::new(2).with_base(10 * file)),
                }) as Arc<dyn LayoutReaderFactory>
            })
            .collect();
        let source = MultiLayoutDataSource::new_with_first(first, rest, vec![], &session);

        let scan = runtime.block_on(source.scan(ScanRequest::default()))?;
        let pieces = scan.split(NonZeroUsize::try_from(5)?)?;
        assert_eq!(pieces.len(), 5);
        let mut values = Vec::new();
        for piece in pieces {
            values.extend(piece_values(&runtime, piece)?);
        }
        values.sort_unstable();
        assert_eq!(values, [0, 1, 10, 11, 20, 21]);
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
