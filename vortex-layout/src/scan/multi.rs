// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`DataSource`] that combines multiple [`LayoutReaderRef`]s into a single scannable source.
//!
//! Readers may be pre-opened or deferred via [`LayoutReaderFactory`]. A scan's units are its files,
//! then each file's layout splits. The workers of an opened scan open deferred files themselves,
//! at most `concurrency` at once, and hand each opened file's splits back to the pool so that any
//! worker can read them. The request's row range and selection apply to each file.
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

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use itertools::Itertools;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_scan::DataSource;
use vortex_scan::Scan;
use vortex_scan::ScanPoolRef;
use vortex_scan::ScanRef;
use vortex_scan::ScanRequest;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::layout::ConstantPool;
use crate::scan::layout::group_contiguous;
use crate::scan::pool::FileWork;
use crate::scan::pool::PlannedSplits;
use crate::scan::pool::PoolOptions;
use crate::scan::pool::PrepareFile;
use crate::scan::pool::PreparedScan;
use crate::scan::pool::SplitPool;
use crate::scan::pool::default_splits_in_flight;
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
/// Readers may be pre-opened or deferred via [`LayoutReaderFactory`]. The workers of an opened
/// scan open deferred readers lazily, at most `concurrency` at once, and share each opened
/// reader's splits.
#[derive(Clone)]
pub struct MultiLayoutDataSource {
    dtype: DType,
    session: VortexSession,
    children: Arc<[MultiLayoutChild]>,
    concurrency: usize,
    max_splits_in_flight: NonZeroUsize,
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
            max_splits_in_flight: default_splits_in_flight(),
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
            max_splits_in_flight: default_splits_in_flight(),
        }
    }

    pub fn children(&self) -> &[MultiLayoutChild] {
        &self.children
    }

    /// Sets how many deferred readers one opened scan opens at once.
    ///
    /// Defaults to the number of available CPU cores.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Bounds the splits in flight at once across all workers of one opened scan; see
    /// [`LayoutReaderDataSource::with_max_splits_in_flight`].
    ///
    /// [`LayoutReaderDataSource::with_max_splits_in_flight`]: crate::scan::layout::LayoutReaderDataSource::with_max_splits_in_flight
    pub fn with_max_splits_in_flight(mut self, max: NonZeroUsize) -> Self {
        self.max_splits_in_flight = max;
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

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<ScanRef> {
        let request = BoundScanRequest::try_new(scan_request, &self.dtype)?;
        let children = self
            .children
            .iter()
            .map(|child| match child {
                MultiLayoutChild::Opened { reader, .. } => ScanChild::Opened(Arc::clone(reader)),
                MultiLayoutChild::Deferred { factory, .. } => {
                    ScanChild::Deferred(Arc::clone(factory))
                }
            })
            .collect();

        Ok(Box::new(MultiLayoutScan {
            context: Arc::new(ScanContext {
                session: self.session.clone(),
                source_dtype: self.dtype.clone(),
                dtype: request.projection.dtype().clone(),
                request,
                concurrency: self.concurrency,
                max_splits_in_flight: self.max_splits_in_flight,
            }),
            children,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

#[derive(Clone)]
struct BoundScanRequest {
    projection: BoundExpression,
    /// A filter that fails to bind fails the scan only once a file is read.
    filter: SharedVortexResult<Option<BoundExpression>>,
    /// The rows of each file to read.
    row_range: Option<Range<u64>>,
    /// The rows of each file to read, by index within the file.
    selection: Selection,
    ordered: bool,
    limit: Option<u64>,
}

impl BoundScanRequest {
    fn try_new(request: ScanRequest, dtype: &DType) -> VortexResult<Self> {
        let ScanRequest {
            projection,
            filter,
            row_range,
            selection,
            ordered,
            limit,
        } = request;

        Ok(Self {
            projection: projection.bind(dtype)?.optimize_recursive()?,
            filter: filter
                .map(|expr| expr.bind(dtype)?.optimize_recursive())
                .transpose()
                .map_err(Arc::new),
            row_range,
            selection,
            ordered,
            limit,
        })
    }

    fn has_filter(&self) -> bool {
        !matches!(self.filter, Ok(None))
    }
}

/// What every piece of a multi-layout scan shares.
struct ScanContext {
    session: VortexSession,
    source_dtype: DType,
    dtype: DType,
    request: BoundScanRequest,
    concurrency: usize,
    max_splits_in_flight: NonZeroUsize,
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

/// One file of a multi-layout scan.
enum ScanChild {
    Opened(LayoutReaderRef),
    /// An opened file whose splits are known: all of them, or the share of one piece.
    Planned {
        scan: Arc<PreparedScan>,
        ranges: Vec<Range<u64>>,
    },
    Deferred(Arc<dyn LayoutReaderFactory>),
}

/// A planned scan over several layout readers. Its units are files, then each file's splits.
struct MultiLayoutScan {
    context: Arc<ScanContext>,
    children: Vec<ScanChild>,
}

impl MultiLayoutScan {
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

impl Scan for MultiLayoutScan {
    fn dtype(&self) -> &DType {
        &self.context.dtype
    }

    fn ordered(&self) -> bool {
        self.context.request.ordered
    }

    fn row_count(&self) -> Precision<u64> {
        let request = &self.context.request;
        let mut known = 0u64;
        let mut known_files = 0u64;
        for child in &self.children {
            let rows = match child {
                ScanChild::Opened(reader) => {
                    let row_range = request
                        .row_range
                        .clone()
                        .unwrap_or_else(|| 0..reader.row_count());
                    request.selection.row_count_in_range(&row_range)
                }
                ScanChild::Planned { scan, ranges } => ranges
                    .iter()
                    .map(|range| scan.selected_row_count(range))
                    .sum(),
                ScanChild::Deferred(_) => continue,
            };
            known = known.saturating_add(rows);
            known_files += 1;
        }
        let files = self.children.len() as u64;
        let (rows, is_exact) = if known_files == files {
            (known, !request.has_filter())
        } else if let Some(average) = known.checked_div(known_files) {
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

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<ScanRef>> {
        let MultiLayoutScan { context, children } = *self;
        if context.request.limit.is_some() || n.get() == 1 || children.is_empty() {
            return Ok(vec![Box::new(MultiLayoutScan { context, children })]);
        }

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
            return Ok(vec![Box::new(MultiLayoutScan {
                context,
                children: planned,
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
            .map(|children| {
                Box::new(MultiLayoutScan {
                    context: Arc::clone(&context),
                    children,
                }) as ScanRef
            })
            .collect())
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        let MultiLayoutScan { context, children } = *self;
        let request = &context.request;
        if request.limit == Some(0) {
            return Ok(Arc::new(ConstantPool::new(context.dtype.clone(), 0)));
        }

        let file_count = children.len();
        let mut opened = Vec::new();
        let mut deferred = Vec::new();
        for (index, child) in children.into_iter().enumerate() {
            match child {
                ScanChild::Opened(reader) => opened.push((index, context.start_file(reader)?)),
                ScanChild::Planned { scan, ranges } => {
                    opened.push((index, FileWork::start(scan, ranges)?))
                }
                ScanChild::Deferred(factory) => deferred.push((index, factory)),
            }
        }

        let options = PoolOptions {
            dtype: context.dtype.clone(),
            ordered: request.ordered,
            limit: request.limit,
            has_filter: request.has_filter(),
            max_splits_in_flight: context.max_splits_in_flight,
        };
        let concurrency = context.concurrency.max(1);
        let prepare: PrepareFile = Arc::new(move |reader| context.start_file(reader));
        Ok(Arc::new(SplitPool::new(
            options,
            opened,
            deferred,
            file_count,
            Some((prepare, concurrency)),
        )))
    }
}

#[cfg(test)]
mod tests {
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
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexResult;
    use vortex_scan::DataSource;
    use vortex_scan::ScanRequest;

    use super::*;
    use crate::scan::test::InFlight;
    use crate::scan::test::TestLayoutReader;
    use crate::scan::test::collect_scan_values;
    use crate::scan::test::new_session;
    use crate::scan::test::read_with_one_worker;
    use crate::scan::test::read_with_workers;

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

    fn plan(source: &MultiLayoutDataSource, request: ScanRequest) -> VortexResult<ScanRef> {
        futures::executor::block_on(source.scan(request))
    }

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
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

    /// Deferred files of `rows` one-row splits, the `i`th producing `100 * i + row`.
    fn gated_source(files: usize, rows: u64) -> (MultiLayoutDataSource, Vec<Arc<Gate>>) {
        let gates: Vec<_> = (0..files).map(|_| Arc::new(Gate::default())).collect();
        let factories = gates
            .iter()
            .enumerate()
            .map(|(index, gate)| {
                let base = 100 * i32::try_from(index).expect("few files");
                Arc::new(GatedFactory {
                    reader: Arc::new(
                        TestLayoutReader::new(rows)
                            .with_split_size(1)
                            .with_base(base),
                    ),
                    gate: Arc::clone(gate),
                    opens: Arc::default(),
                }) as Arc<dyn LayoutReaderFactory>
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

    /// Acceptance: a deferred file is opened by a worker, and its splits are shared with other
    /// workers. A worker that finds only an open in progress waits, and is woken when the open
    /// adds splits.
    #[test]
    fn deferred_file_splits_are_shared_with_waiting_workers() -> VortexResult<()> {
        let (source, gates) = gated_source(1, 8);
        let pool = plan(
            &source.with_max_splits_in_flight(NonZeroUsize::MIN),
            ScanRequest::default(),
        )?
        .open()?;
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
        assert!(!pool.is_exhausted());

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
        assert_eq!(values, (0..8).collect::<Vec<_>>());
        Ok(())
    }

    /// A worker dropped while opening a file hands the file back, so another worker opens it.
    #[test]
    fn dropped_worker_hands_back_its_open() -> VortexResult<()> {
        let (source, gates) = gated_source(1, 4);
        let pool = plan(&source, ScanRequest::default())?.open()?;
        let mut abandoned = Arc::clone(&pool).worker();
        let mut noop = Context::from_waker(noop_waker_ref());
        assert!(poll_worker(&mut abandoned, &mut noop).is_pending());
        drop(abandoned);

        gates[0].release();
        assert_eq!(read_with_one_worker(pool)?, [0, 1, 2, 3]);
        Ok(())
    }

    /// A single ordered worker emits files in file order, even when a later file opens first.
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

    /// An unordered limit is shared by every worker and file of the pool.
    #[test]
    fn unordered_limit_is_shared_across_readers() -> VortexResult<()> {
        let first: LayoutReaderRef = Arc::new(TestLayoutReader::new(4).with_split_size(1));
        let second: LayoutReaderRef =
            Arc::new(TestLayoutReader::new(4).with_split_size(1).with_base(10));
        let source = MultiLayoutDataSource::new_with_first(
            first,
            vec![Arc::new(StaticReaderFactory { reader: second })],
            vec![],
            &new_session(),
        )
        .with_max_splits_in_flight(NonZeroUsize::MIN);

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
        let second: LayoutReaderRef =
            Arc::new(TestLayoutReader::new(4).with_split_size(2).with_base(100));
        let opens = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Gate::default());
        gate.release();
        let source = MultiLayoutDataSource::new_with_first(
            first,
            vec![Arc::new(GatedFactory {
                reader: second,
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

    /// Reads 64 one-row splits with one worker, returning the rows and the most splits read at
    /// once.
    fn read_with_splits_in_flight(max: Option<NonZeroUsize>) -> VortexResult<(Vec<i32>, usize)> {
        let in_flight = Arc::new(InFlight::default());
        let reader: LayoutReaderRef = Arc::new(
            TestLayoutReader::new(64)
                .with_split_size(1)
                .with_in_flight(Arc::clone(&in_flight)),
        );
        let mut source =
            MultiLayoutDataSource::new_with_first(reader, vec![], vec![], &new_session());
        if let Some(max) = max {
            source = source.with_max_splits_in_flight(max);
        }

        let pool = plan(
            &source,
            ScanRequest {
                ordered: true,
                ..Default::default()
            },
        )?
        .open()?;
        let values = read_with_one_worker(pool)?;
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

    /// A worker reads at most `max_splits_in_flight` splits at once, however many cores there
    /// are, and by default overlaps more.
    #[rstest]
    #[case(1)]
    #[case(2)]
    fn max_splits_in_flight_bounds_the_splits_read_at_once(#[case] max: usize) -> VortexResult<()> {
        let (values, peak) = read_with_splits_in_flight(NonZeroUsize::new(max))?;
        let (default_values, default_peak) = read_with_splits_in_flight(None)?;

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
