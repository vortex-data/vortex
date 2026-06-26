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
use itertools::Itertools;
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
use vortex_scan::selection::Selection;
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
            children,
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

    fn row_count(&self) -> Precision<u64> {
        let mut sum: u64 = 0;
        let mut opened_count: u64 = 0;
        let mut deferred_count: u64 = 0;

        for child in &self.children {
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
        } else if opened_count > 0 {
            let avg = sum / opened_count;
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
        for child in &self.children {
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

        for child in &self.children {
            match child {
                MultiLayoutChild::Opened { reader, .. } => ready.push_back(Arc::clone(reader)),
                MultiLayoutChild::Deferred { factory, .. } => {
                    deferred.push_back(Arc::clone(factory))
                }
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
            .enumerate()
            .flat_map(move |(i, reader_result)| match reader_result {
                Ok(reader) => reader_partition(i, reader, session.clone(), request.clone()),
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
    request: ScanRequest,
) -> PartitionStream {
    let row_count = reader.row_count();
    let row_range = request.row_range.clone().unwrap_or(0..row_count);

    let partition_idx_u64: u64 = partition_idx as u64;
    if let Some(range) = &request.partition_range
        && !range.contains(&partition_idx_u64)
    {
        return stream::empty().boxed();
    };
    match &request.partition_selection {
        Selection::IncludeByIndex(buffer) => {
            if buffer.as_slice().binary_search(&partition_idx_u64).is_err() {
                return stream::empty().boxed();
            }
        }
        Selection::ExcludeByIndex(buffer) => {
            if buffer.as_slice().binary_search(&partition_idx_u64).is_ok() {
                return stream::empty().boxed();
            }
        }
        _ => {}
    };

    // Check file-level pruning: if the filter can be proven false for the entire row range
    // using file-level statistics, skip this reader entirely.
    if let Some(filter) = &request.filter {
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
    request: ScanRequest,
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
        let row_count = row_range.end - row_range.start;
        let row_count = self.request.selection.row_count(row_count);
        let row_count = self
            .request
            .limit
            .map_or(row_count, |limit| row_count.min(limit));

        if self.request.filter.is_some() {
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
        let mut builder = ScanBuilder::new(self.session, self.reader)
            .with_selection(request.selection)
            .with_projection(request.projection)
            .with_some_filter(request.filter)
            .with_some_limit(request.limit)
            .with_attach_aggregate_stats(request.attach_aggregate_stats)
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::Nullability;

    use super::*;
    use crate::scan::test::new_session;

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
}
