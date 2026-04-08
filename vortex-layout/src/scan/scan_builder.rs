// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::immediate_access::immediate_scope_access;
use vortex_array::expr::root;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_metrics::MetricsRegistry;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::layouts::row_idx::RowIdxLayoutReader;
use crate::scan::repeated_scan::RepeatedScan;
use crate::scan::split_by::SplitBy;
use crate::scan::splits::Splits;
use crate::scan::splits::attempt_split_ranges;

/// A struct for building a scan operation.
pub struct ScanBuilder<A> {
    session: VortexSession,
    layout_reader: LayoutReaderRef,
    projection: Expression,
    filter: Option<Expression>,
    /// Whether the scan needs to return splits in the order they appear in the file.
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    // TODO(joe): replace this is usage of row_id selection, see
    selection: Selection,
    /// How to split the file for concurrent processing.
    split_by: SplitBy,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
    /// Should we try to prune the file (using stats) on open.
    file_stats: Option<Arc<[StatsSet]>>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The row-offset assigned to the first row of the file. Used by the `row_idx` expression,
    /// but not by the scan [`Selection`] which remains relative.
    row_offset: u64,
}

impl ScanBuilder<ArrayRef> {
    pub fn new(session: VortexSession, layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            session,
            layout_reader,
            projection: root(),
            filter: None,
            ordered: true,
            row_range: None,
            selection: Default::default(),
            split_by: SplitBy::Layout,
            // We default to four tasks per worker thread, which allows for some I/O lookahead
            // without too much impact on work-stealing.
            concurrency: 4,
            map_fn: Arc::new(Ok),
            metrics_registry: None,
            file_stats: None,
            limit: None,
            row_offset: 0,
        }
    }

    /// Returns an [`ArrayStream`] with tasks spawned onto the session's runtime handle.
    ///
    /// See [`ScanBuilder::into_stream`] for more details.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype()?;
        let stream = self.into_stream()?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }

    /// Returns an [`ArrayIterator`] using the given blocking runtime.
    pub fn into_array_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let stream = self.into_array_stream()?;
        let dtype = stream.dtype().clone();
        Ok(ArrayIteratorAdapter::new(
            dtype,
            runtime.block_on_stream(stream),
        ))
    }
}

impl<A: 'static + Send> ScanBuilder<A> {
    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_some_filter(mut self, filter: Option<Expression>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    pub fn ordered(&self) -> bool {
        self.ordered
    }

    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
        self
    }

    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// The number of row splits to make progress on concurrently per-thread, must
    /// be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    pub fn with_some_metrics_registry(mut self, metrics: Option<Arc<dyn MetricsRegistry>>) -> Self {
        self.metrics_registry = metrics;
        self
    }

    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics);
        self
    }

    pub fn with_some_limit(mut self, limit: Option<u64>) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// The [`DType`] returned by the scan, after applying the projection.
    pub fn dtype(&self) -> VortexResult<DType> {
        self.projection.return_dtype(self.layout_reader.dtype())
    }

    /// The session used by the scan.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Map each split of the scan. The function will be run on the spawned task.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            session: self.session,
            layout_reader: self.layout_reader,
            projection: self.projection,
            filter: self.filter,
            ordered: self.ordered,
            row_range: self.row_range,
            selection: self.selection,
            split_by: self.split_by,
            concurrency: self.concurrency,
            metrics_registry: self.metrics_registry,
            file_stats: self.file_stats,
            limit: self.limit,
            row_offset: self.row_offset,
            map_fn: Arc::new(move |a| old_map_fn(a).and_then(&map_fn)),
        }
    }

    pub fn prepare(self) -> VortexResult<RepeatedScan<A>> {
        let dtype = self.dtype()?;

        if self.filter.is_some() && self.limit.is_some() {
            vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
        }

        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let mut layout_reader = self.layout_reader;

        // Enrich the layout reader to support RowIdx expressions.
        // Note that this is applied below the filter layout reader since it can perform
        // better over individual conjunctions.
        layout_reader = Arc::new(RowIdxLayoutReader::new(
            self.row_offset,
            layout_reader,
            self.session.clone(),
        ));

        // Normalize and simplify the expressions.
        let projection = self.projection.optimize_recursive(layout_reader.dtype())?;

        let filter = self
            .filter
            .map(|f| f.optimize_recursive(layout_reader.dtype()))
            .transpose()?;

        // Construct field masks and compute the row splits of the scan.
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, filter.as_ref(), layout_reader.dtype())?;
        let field_mask: Vec<_> = [filter_mask, projection_mask].concat();

        let splits =
            if let Some(ranges) = attempt_split_ranges(&self.selection, self.row_range.as_ref()) {
                Splits::Ranges(ranges)
            } else {
                let split_range = self
                    .row_range
                    .clone()
                    .unwrap_or_else(|| 0..layout_reader.row_count());
                Splits::Natural(self.split_by.splits(
                    layout_reader.as_ref(),
                    &split_range,
                    &field_mask,
                )?)
            };

        Ok(RepeatedScan::new(
            self.session.clone(),
            layout_reader,
            projection,
            filter,
            self.ordered,
            self.row_range,
            self.selection,
            splits,
            self.concurrency,
            self.map_fn,
            self.limit,
            dtype,
        ))
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    pub fn build(self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        // The ultimate short circuit
        if self.limit.is_some_and(|l| l == 0) {
            return Ok(vec![]);
        }

        self.prepare()?.execute(None)
    }

    /// Returns a [`Stream`] with tasks spawned onto the session's runtime handle.
    pub fn into_stream(
        self,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        Ok(LazyScanStream::new(self))
    }

    /// Returns an [`Iterator`] using the session's runtime.
    pub fn into_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl Iterator<Item = VortexResult<A>> + 'static> {
        let stream = self.into_stream()?;
        Ok(runtime.block_on_stream(stream))
    }
}

enum LazyScanState<A: 'static + Send> {
    Builder(Option<Box<ScanBuilder<A>>>),
    Preparing(PreparingScan<A>),
    Stream(BoxStream<'static, VortexResult<A>>),
    Error(Option<vortex_error::VortexError>),
}

type PreparedScanTasks<A> = Vec<BoxFuture<'static, VortexResult<Option<A>>>>;

struct PreparingScan<A: 'static + Send> {
    ordered: bool,
    concurrency: usize,
    handle: Handle,
    task: Task<VortexResult<PreparedScanTasks<A>>>,
}

struct LazyScanStream<A: 'static + Send> {
    state: LazyScanState<A>,
}

impl<A: 'static + Send> LazyScanStream<A> {
    fn new(builder: ScanBuilder<A>) -> Self {
        Self {
            state: LazyScanState::Builder(Some(Box::new(builder))),
        }
    }
}

impl<A: 'static + Send> Unpin for LazyScanStream<A> {}

impl<A: 'static + Send> Stream for LazyScanStream<A> {
    type Item = VortexResult<A>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                LazyScanState::Builder(builder) => {
                    let builder = builder.take().vortex_expect("polled after completion");
                    let ordered = builder.ordered;
                    let num_workers = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    let concurrency = builder.concurrency * num_workers;
                    let handle = builder.session.handle();
                    let task = handle.spawn_blocking(move || {
                        builder.prepare().and_then(|scan| scan.execute(None))
                    });
                    self.state = LazyScanState::Preparing(PreparingScan {
                        ordered,
                        concurrency,
                        handle,
                        task,
                    });
                }
                LazyScanState::Preparing(preparing) => {
                    match ready!(Pin::new(&mut preparing.task).poll(cx)) {
                        Ok(tasks) => {
                            let ordered = preparing.ordered;
                            let concurrency = preparing.concurrency;
                            let handle = preparing.handle.clone();
                            let stream =
                                futures::stream::iter(tasks).map(move |task| handle.spawn(task));
                            let stream = if ordered {
                                stream.buffered(concurrency).boxed()
                            } else {
                                stream.buffer_unordered(concurrency).boxed()
                            };
                            let stream = stream
                                .filter_map(|chunk| async move { chunk.transpose() })
                                .boxed();
                            self.state = LazyScanState::Stream(stream);
                        }
                        Err(err) => self.state = LazyScanState::Error(Some(err)),
                    }
                }
                LazyScanState::Stream(stream) => return stream.as_mut().poll_next(cx),
                LazyScanState::Error(err) => return Poll::Ready(err.take().map(Err)),
            }
        }
    }
}

/// Compute masks of field paths referenced by the projection and filter in the scan.
///
/// Projection and filter must be pre-simplified.
pub fn filter_and_projection_masks(
    projection: &Expression,
    filter: Option<&Expression>,
    dtype: &DType,
) -> VortexResult<(Vec<FieldMask>, Vec<FieldMask>)> {
    let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
        return Ok(match filter {
            Some(_) => (vec![FieldMask::All], vec![FieldMask::All]),
            None => (Vec::new(), vec![FieldMask::All]),
        });
    };
    let projection_mask = immediate_scope_access(projection, struct_dtype);
    Ok(match filter {
        None => (
            Vec::new(),
            projection_mask.into_iter().map(to_field_mask).collect_vec(),
        ),
        Some(f) => {
            let filter_mask = immediate_scope_access(f, struct_dtype);
            let only_projection_mask = projection_mask
                .difference(&filter_mask)
                .cloned()
                .map(to_field_mask)
                .collect_vec();
            (
                filter_mask.into_iter().map(to_field_mask).collect_vec(),
                only_projection_mask,
            )
        }
    })
}

fn to_field_mask(field: FieldName) -> FieldMask {
    FieldMask::Prefix(FieldPath::from(Field::Name(field)))
}

#[cfg(test)]
mod test {
    use std::collections::BTreeSet;
    use std::ops::Range;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;
    use std::time::Duration;

    use futures::Stream;
    use futures::task::noop_waker_ref;
    use parking_lot::Mutex;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_mask::Mask;

    use super::ScanBuilder;
    use crate::ArrayFuture;
    use crate::LayoutReader;

    #[derive(Debug)]
    struct CountingLayoutReader {
        name: Arc<str>,
        dtype: DType,
        row_count: u64,
        register_splits_calls: Arc<AtomicUsize>,
    }

    impl CountingLayoutReader {
        fn new(register_splits_calls: Arc<AtomicUsize>) -> Self {
            Self {
                name: Arc::from("counting"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                row_count: 1,
                register_splits_calls,
            }
        }
    }

    impl LayoutReader for CountingLayoutReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            self.row_count
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            row_range: &Range<u64>,
            splits: &mut BTreeSet<u64>,
        ) -> VortexResult<()> {
            self.register_splits_calls.fetch_add(1, Ordering::Relaxed);
            splits.insert(row_range.end);
            Ok(())
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: Mask,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for this test");
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for this test");
        }

        fn projection_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            Ok(Box::pin(async move {
                unreachable!("scan should not be polled in this test")
            }))
        }
    }

    #[test]
    fn into_stream_is_lazy() {
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(CountingLayoutReader::new(Arc::clone(&calls)));

        let session = crate::scan::test::SCAN_SESSION.clone();

        let _stream = ScanBuilder::new(session, reader).into_stream().unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[derive(Debug)]
    struct SplittingLayoutReader {
        name: Arc<str>,
        dtype: DType,
        row_count: u64,
        register_splits_calls: Arc<AtomicUsize>,
    }

    impl SplittingLayoutReader {
        fn new(register_splits_calls: Arc<AtomicUsize>) -> Self {
            Self {
                name: Arc::from("splitting"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                row_count: 4,
                register_splits_calls,
            }
        }
    }

    impl LayoutReader for SplittingLayoutReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            self.row_count
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            row_range: &Range<u64>,
            splits: &mut BTreeSet<u64>,
        ) -> VortexResult<()> {
            self.register_splits_calls.fetch_add(1, Ordering::Relaxed);
            for split in (row_range.start + 1)..=row_range.end {
                splits.insert(split);
            }
            Ok(())
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: Mask,
        ) -> VortexResult<MaskFuture> {
            Ok(MaskFuture::ready(mask))
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            Ok(mask)
        }

        fn projection_evaluation(
            &self,
            row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            let start = usize::try_from(row_range.start)
                .map_err(|_| vortex_err!("row_range.start must fit in usize"))?;
            let end = usize::try_from(row_range.end)
                .map_err(|_| vortex_err!("row_range.end must fit in usize"))?;

            let values: VortexResult<Vec<i32>> = (start..end)
                .map(|v| i32::try_from(v).map_err(|_| vortex_err!("split value must fit in i32")))
                .collect();

            let array = PrimitiveArray::from_iter(values?).into_array();
            Ok(Box::pin(async move { Ok(array) }))
        }
    }

    #[test]
    fn into_stream_executes_after_prepare() -> VortexResult<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(SplittingLayoutReader::new(Arc::clone(&calls)));

        let runtime = SingleThreadRuntime::default();
        let session = crate::scan::test::session_with_handle(runtime.handle());

        let stream = ScanBuilder::new(session, reader).into_stream().unwrap();
        let mut iter = runtime.block_on_stream(stream);

        let mut values = Vec::new();
        for chunk in &mut iter {
            values.push(chunk?.to_primitive().into_buffer::<i32>()[0]);
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(values.as_ref(), [0, 1, 2, 3]);

        Ok(())
    }

    #[derive(Debug)]
    struct BlockingSplitsLayoutReader {
        name: Arc<str>,
        dtype: DType,
        row_count: u64,
        register_splits_calls: Arc<AtomicUsize>,
        gate: Arc<Mutex<()>>,
    }

    impl BlockingSplitsLayoutReader {
        fn new(gate: Arc<Mutex<()>>, register_splits_calls: Arc<AtomicUsize>) -> Self {
            Self {
                name: Arc::from("blocking-splits"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                row_count: 1,
                register_splits_calls,
                gate,
            }
        }
    }

    impl LayoutReader for BlockingSplitsLayoutReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            self.row_count
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            row_range: &Range<u64>,
            splits: &mut BTreeSet<u64>,
        ) -> VortexResult<()> {
            self.register_splits_calls.fetch_add(1, Ordering::Relaxed);
            let _guard = self.gate.lock();
            splits.insert(row_range.end);
            Ok(())
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: Mask,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for this test");
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for this test");
        }

        fn projection_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            Ok(Box::pin(async move {
                unreachable!("scan should not be polled in this test")
            }))
        }
    }

    #[test]
    fn into_stream_first_poll_does_not_block() {
        let gate = Arc::new(Mutex::new(()));
        let guard = gate.lock();

        let calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(BlockingSplitsLayoutReader::new(
            Arc::clone(&gate),
            Arc::clone(&calls),
        ));

        let runtime = SingleThreadRuntime::default();
        let session = crate::scan::test::session_with_handle(runtime.handle());

        let mut stream = ScanBuilder::new(session, reader).into_stream().unwrap();

        let (send, recv) = std::sync::mpsc::channel::<bool>();
        let join = std::thread::spawn(move || {
            let waker = noop_waker_ref();
            let mut cx = Context::from_waker(waker);
            let poll = Pin::new(&mut stream).poll_next(&mut cx);
            let _ = send.send(matches!(poll, Poll::Pending));
        });

        let polled_pending = recv.recv_timeout(Duration::from_secs(1)).ok();

        // Always release the gate and join the thread so failures don't hang the test process.
        drop(guard);
        drop(join.join());

        let polled_pending = polled_pending.expect("poll_next blocked; expected quick return");
        assert!(
            polled_pending,
            "expected Poll::Pending while prepare is blocked"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        drop(runtime);
    }
}
