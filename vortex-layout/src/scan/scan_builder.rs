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
use futures::stream::BoxStream;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::referenced_field_paths;
use vortex_array::expr::root;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_metrics::MetricsRegistry;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::layouts::row_idx::RowIdxLayoutReader;
use crate::scan::limit::RowLimit;
use crate::scan::repeated_scan::RepeatedScan;
use crate::scan::split_by::SplitBy;
use crate::scan::splits::Splits;
use crate::scan::splits::attempt_split_ranges;

/// Builder for scanning a [`LayoutReader`] into arrays, streams, or iterators.
///
/// A scan has three independent row restriction mechanisms:
///
/// - [`with_row_range`](Self::with_row_range) selects a contiguous range before scanning.
/// - [`with_selection`](Self::with_selection) applies a [`Selection`] inside that range.
/// - [`with_filter`](Self::with_filter) evaluates an expression predicate during execution.
///
/// Projection and filter expressions are optimized against the reader dtype during
/// [`prepare`](Self::prepare). Work is divided by the configured [`SplitBy`] strategy or by
/// explicit selection ranges.
pub struct ScanBuilder {
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
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
    /// Maximal number of rows to read after filtering.
    limit: Option<u64>,
    /// A row limit shared with sibling external partitions, when the caller owns one.
    row_limit: Option<RowLimit>,
    /// The row-offset assigned to the first row of the file. Used by the `row_idx` expression,
    /// but not by the scan [`Selection`] which remains relative.
    row_offset: u64,
}

impl ScanBuilder {
    /// Create a scan builder over `layout_reader` using `session` for runtime and execution state.
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
            metrics_registry: None,
            limit: None,
            row_limit: None,
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

impl ScanBuilder {
    /// Add a filter expression evaluated against the projected row ranges.
    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add or clear the filter expression.
    pub fn with_some_filter(mut self, filter: Option<Expression>) -> Self {
        self.filter = filter;
        self
    }

    /// Set the projection expression for returned rows.
    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    /// Returns whether output chunks are yielded in file order.
    pub fn ordered(&self) -> bool {
        self.ordered
    }

    /// Configure whether output chunks must be yielded in file order.
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Restrict scanning to a contiguous row range.
    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    /// Apply a row selection to the selected row range.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Select rows by absolute indices relative to the scan input.
    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
        self
    }

    /// Set the root row offset used by row-index expressions.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    /// Configure how natural scan work is split for concurrency.
    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// Returns the per-worker row-split concurrency.
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

    /// Add or clear the metrics registry used by scan execution.
    pub fn with_some_metrics_registry(mut self, metrics: Option<Arc<dyn MetricsRegistry>>) -> Self {
        self.metrics_registry = metrics;
        self
    }

    /// Set the metrics registry used by scan execution.
    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics);
        self
    }

    /// Add or clear the maximum number of rows returned after filtering.
    pub fn with_some_limit(mut self, limit: Option<u64>) -> Self {
        self.limit = limit;
        self
    }

    /// Set the maximum number of rows returned after filtering.
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Use a row limit supplied by the enclosing data source instead of creating a local one.
    pub(crate) fn with_some_row_limit(mut self, row_limit: Option<RowLimit>) -> Self {
        self.row_limit = row_limit;
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

    /// Optimize expressions, compute split ranges, and return an executable repeated scan.
    pub fn prepare(self) -> VortexResult<RepeatedScan> {
        let dtype = self.dtype()?;

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
        let field_mask =
            referenced_field_masks(&projection, filter.as_ref(), layout_reader.dtype())?;

        let splits =
            if let Some(ranges) = attempt_split_ranges(&self.selection, self.row_range.as_ref()) {
                Splits::Ranges(ranges)
            } else {
                let split_range = self
                    .row_range
                    .clone()
                    .unwrap_or_else(|| 0..layout_reader.row_count());
                Splits::Natural(
                    self.split_by
                        .splits(layout_reader.as_ref(), &split_range, &field_mask)?
                        .into(),
                )
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
            self.limit,
            self.row_limit,
            dtype,
        ))
    }

    /// Returns a [`Stream`] with tasks spawned onto the session's runtime handle.
    ///
    /// Preparation and initial stream construction begin on the first poll. Errors from either
    /// step are returned as the stream's next item.
    pub fn into_stream(
        self,
    ) -> VortexResult<impl Stream<Item = VortexResult<ArrayRef>> + Send + 'static> {
        Ok(LazyScanStream::new(self))
    }

    /// Returns an [`Iterator`] using the session's runtime.
    pub fn into_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl Iterator<Item = VortexResult<ArrayRef>> + 'static> {
        let stream = self.into_stream()?;
        Ok(runtime.block_on_stream(stream))
    }
}

enum LazyScanState {
    Builder(Option<Box<ScanBuilder>>),
    Preparing(PreparingScan),
    Stream(BoxStream<'static, VortexResult<ArrayRef>>),
    Error(Option<vortex_error::VortexError>),
}

struct PreparingScan {
    task: Task<VortexResult<BoxStream<'static, VortexResult<ArrayRef>>>>,
}

struct LazyScanStream {
    state: LazyScanState,
}

impl LazyScanStream {
    fn new(builder: ScanBuilder) -> Self {
        Self {
            state: LazyScanState::Builder(Some(Box::new(builder))),
        }
    }
}

impl Unpin for LazyScanStream {}

impl Stream for LazyScanStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                LazyScanState::Builder(builder) => {
                    let builder = builder.take().vortex_expect("polled after completion");
                    let handle = builder.session.handle();
                    // IMPORTANT: Building the stream can synchronously walk the layout and
                    // register I/O for every split. Keep it with preparation in this CPU
                    // task: poll_next must only wait for and poll an already-constructed stream.
                    // This also keeps construction errors on the Preparing -> Error path rather
                    // than running construction on the caller's executor.
                    let task = handle.spawn_cpu(move || {
                        let scan = builder.prepare()?;
                        Ok(scan.execute_stream(None)?.boxed())
                    });
                    self.state = LazyScanState::Preparing(PreparingScan { task });
                }
                LazyScanState::Preparing(preparing) => {
                    match ready!(Pin::new(&mut preparing.task).poll(cx)) {
                        Ok(stream) => self.state = LazyScanState::Stream(stream),
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
pub fn referenced_field_masks(
    projection: &Expression,
    filter: Option<&Expression>,
    dtype: &DType,
) -> VortexResult<Vec<FieldMask>> {
    if dtype.as_struct_fields_opt().is_none() {
        return Ok(vec![FieldMask::All]);
    }

    let mut field_paths = referenced_field_paths(projection, dtype)?;
    if let Some(filter) = filter {
        field_paths.extend(referenced_field_paths(filter, dtype)?);
    }
    Ok(field_paths.into_iter().map(FieldMask::Prefix).collect_vec())
}

#[cfg(test)]
mod test {
    use std::ops::Range;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::task::Context;
    use std::task::Poll;
    use std::time::Duration;

    use futures::Stream;
    use futures::task::noop_waker_ref;
    use parking_lot::Mutex;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_io::runtime::tokio::TokioRuntime;
    use vortex_mask::Mask;

    use super::ScanBuilder;
    use super::referenced_field_masks;
    use crate::ArrayFuture;
    use crate::LayoutReader;
    use crate::RowSplits;
    use crate::SplitRange;
    use crate::scan::test::SCAN_SESSION;
    use crate::scan::test::TestLayoutReader;
    use crate::scan::test::collect_scan_values;
    use crate::scan::test::drain_runtime;
    use crate::scan::test::keep_all;
    use crate::scan::test::keep_odd;
    use crate::scan::test::session_with_handle;

    fn nested_dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    DType::Struct(
                        StructFields::from_iter([
                            ("1", DType::Primitive(PType::I32, Nullability::NonNullable)),
                            ("2", DType::Primitive(PType::I32, Nullability::NonNullable)),
                        ]),
                        Nullability::NonNullable,
                    ),
                ),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn nested_projection_preserves_field_path_in_split_mask() -> VortexResult<()> {
        let projection = get_item("1", get_item("a", root()));
        let filter = eq(get_item("2", get_item("a", root())), lit(0_i32));

        let field_masks = referenced_field_masks(&projection, Some(&filter), &nested_dtype())?;

        assert_eq!(field_masks.len(), 2);
        assert!(field_masks.contains(&FieldMask::Prefix(FieldPath::from_name("a").push("1"))));
        assert!(field_masks.contains(&FieldMask::Prefix(FieldPath::from_name("a").push("2"))));
        Ok(())
    }

    #[test]
    fn filter_path_covers_nested_projection_path() -> VortexResult<()> {
        let projection = get_item("1", get_item("a", root()));
        let filter = is_not_null(get_item("a", root()));

        let field_masks = referenced_field_masks(&projection, Some(&filter), &nested_dtype())?;

        assert_eq!(field_masks, [FieldMask::Prefix(FieldPath::from_name("a"))]);
        Ok(())
    }

    #[test]
    fn parent_projection_path_covers_nested_filter_path() -> VortexResult<()> {
        let projection = get_item("a", root());
        let filter = is_not_null(get_item("1", get_item("a", root())));

        let field_masks = referenced_field_masks(&projection, Some(&filter), &nested_dtype())?;

        assert_eq!(field_masks, [FieldMask::Prefix(FieldPath::from_name("a"))]);
        Ok(())
    }

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
            split_range: &SplitRange,
            splits: &mut RowSplits,
        ) -> VortexResult<()> {
            self.register_splits_calls.fetch_add(1, Ordering::Relaxed);
            splits.push(split_range.root_row_range().end);
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

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn into_stream_is_lazy() {
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(CountingLayoutReader::new(Arc::clone(&calls)));

        let session = SCAN_SESSION.clone();

        let _stream = ScanBuilder::new(session, reader).into_stream().unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[derive(Debug)]
    struct SplittingLayoutReader {
        name: Arc<str>,
        dtype: DType,
        row_count: u64,
        register_splits_calls: Arc<AtomicUsize>,
        blocking_projection: Option<BlockingProjection>,
    }

    #[derive(Debug)]
    struct BlockingProjection {
        started: mpsc::Sender<()>,
        gate: Arc<Mutex<()>>,
    }

    impl SplittingLayoutReader {
        fn new(register_splits_calls: Arc<AtomicUsize>) -> Self {
            Self {
                name: Arc::from("splitting"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                row_count: 4,
                register_splits_calls,
                blocking_projection: None,
            }
        }

        fn with_blocking_projection(
            register_splits_calls: Arc<AtomicUsize>,
            gate: Arc<Mutex<()>>,
            started: mpsc::Sender<()>,
        ) -> Self {
            let mut reader = Self::new(register_splits_calls);
            reader.blocking_projection = Some(BlockingProjection { started, gate });
            reader
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
            split_range: &SplitRange,
            splits: &mut RowSplits,
        ) -> VortexResult<()> {
            self.register_splits_calls.fetch_add(1, Ordering::Relaxed);
            for split in (split_range.row_range().start + 1)..=split_range.row_range().end {
                splits.push(split_range.row_offset() + split);
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
            if let Some(blocking_projection) = &self.blocking_projection {
                blocking_projection
                    .started
                    .send(())
                    .map_err(|_| vortex_err!("test projection-start receiver dropped"))?;
                let _guard = blocking_projection.gate.lock();
            }

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

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn into_stream_executes_after_prepare() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(SplittingLayoutReader::new(Arc::clone(&calls)));

        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());

        let stream = ScanBuilder::new(session, reader).into_stream()?;
        let mut iter = runtime.block_on_stream(stream);

        let mut values = Vec::new();
        for chunk in &mut iter {
            let prim = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
            values.push(prim.into_buffer::<i32>()[0]);
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(values.as_ref(), [0, 1, 2, 3]);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn into_stream_constructs_tasks_off_the_poller() -> VortexResult<()> {
        let gate = Arc::new(Mutex::new(()));
        let guard = gate.lock();
        let calls = Arc::new(AtomicUsize::new(0));
        let (started_send, started_recv) = mpsc::channel();
        let reader = Arc::new(SplittingLayoutReader::with_blocking_projection(
            Arc::clone(&calls),
            Arc::clone(&gate),
            started_send,
        ));

        let runtime = TokioRuntime::new(tokio::runtime::Handle::current());
        let session = session_with_handle(runtime.handle());
        let mut stream = ScanBuilder::new(session, reader).into_stream()?;

        let (poll_send, poll_recv) = mpsc::channel();
        let (release_send, release_recv) = mpsc::channel();
        let join = std::thread::spawn(move || {
            let waker = noop_waker_ref();
            let mut cx = Context::from_waker(waker);
            let poll = Pin::new(&mut stream).poll_next(&mut cx);
            let _ = poll_send.send(matches!(poll, Poll::Pending));
            let _ = release_recv.recv();
        });

        let poll_result = poll_recv.recv_timeout(Duration::from_secs(1));
        let projection_started = started_recv.recv_timeout(Duration::from_secs(1));

        // Release the task and join its caller before reporting a failed assertion.
        drop(guard);
        let _ = release_send.send(());
        drop(join.join());

        assert!(
            poll_result.is_ok_and(|poll_pending| poll_pending),
            "first poll must return while scan task construction is blocked"
        );
        projection_started
            .map_err(|_| vortex_err!("stream construction did not begin in the background"))?;
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        Ok(())
    }

    #[tokio::test]
    async fn into_stream_reports_stream_construction_errors() -> VortexResult<()> {
        let range_start = i32::MAX as u64 + 1;
        let reader = Arc::new(SplittingLayoutReader::new(Arc::new(AtomicUsize::new(0))));
        let session = session_with_handle(TokioRuntime::current());
        let mut stream = ScanBuilder::new(session, reader)
            .with_row_range(range_start..range_start + 1)
            .into_stream()?;

        assert!(matches!(
            futures::StreamExt::next(&mut stream).await,
            Some(Err(_))
        ));
        assert!(futures::StreamExt::next(&mut stream).await.is_none());

        Ok(())
    }

    #[rstest]
    #[case::limit_below_matches(8, keep_all, 3, &[0, 1, 2])]
    #[case::limit_zero(8, keep_all, 0, &[])]
    #[case::limit_exceeds_matches(8, keep_odd, 100, &[1, 3, 5, 7])]
    #[case::empty_input(0, keep_all, 3, &[])]
    fn filtered_limit_yields_expected_rows(
        #[case] row_count: u64,
        #[case] keep_row: fn(u64) -> bool,
        #[case] limit: u64,
        #[case] expected: &[i32],
    ) -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let reader = Arc::new(
            TestLayoutReader::new(row_count)
                .with_split_size(2)
                .with_keep_row(keep_row),
        );

        let stream = ScanBuilder::new(session, reader)
            .with_filter(root())
            .with_limit(limit)
            .into_stream()?;
        let values = collect_scan_values(runtime.block_on_stream(stream))?;
        drain_runtime(&runtime);

        assert_eq!(values.as_slice(), expected);
        Ok(())
    }

    /// An unordered filtered limit reserves rows before projecting, so a huge split never decodes
    /// more rows than the limit can return.
    #[test]
    fn unordered_filtered_limit_limits_projection_mask_before_projection() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let projection_masks = Arc::new(Mutex::new(Vec::new()));
        let reader = Arc::new(
            TestLayoutReader::new(100_000).with_projection_masks(Arc::clone(&projection_masks)),
        );

        let stream = ScanBuilder::new(session, reader)
            .with_filter(root())
            .with_limit(1)
            .with_ordered(false)
            .into_stream()?;
        let values = collect_scan_values(runtime.block_on_stream(stream))?;

        assert_eq!(values, [0]);
        assert_eq!(projection_masks.lock().as_slice(), [1]);
        Ok(())
    }

    /// An ordered filtered limit cannot reserve per split (that would grant the budget to whichever
    /// split filters first), so it trims the in-order output instead.
    #[test]
    fn ordered_filtered_limit_trims_the_emitted_rows() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let reader = Arc::new(TestLayoutReader::new(8).with_split_size(4));

        let stream = ScanBuilder::new(session, reader)
            .with_filter(root())
            .with_limit(6)
            .into_stream()?;
        let values = collect_scan_values(runtime.block_on_stream(stream))?;
        drain_runtime(&runtime);

        assert_eq!(values, [0, 1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn filter_errors_are_stream_items_and_do_not_consume_the_limit() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let projection_masks = Arc::new(Mutex::new(Vec::new()));
        let reader = Arc::new(
            TestLayoutReader::new(2)
                .with_split_size(1)
                .with_projection_masks(Arc::clone(&projection_masks))
                .with_fail_first_filter(),
        );
        let stream = ScanBuilder::new(session, reader)
            .with_filter(root())
            .with_limit(1)
            .into_stream()?;
        let mut iter = runtime.block_on_stream(stream);

        assert!(matches!(iter.next(), Some(Err(_))));
        let Some(chunk) = iter.next() else {
            return Err(vortex_err!(
                "matching split was not polled after the filter error"
            ));
        };
        let mut ctx = array_session().create_execution_ctx();
        let primitive = chunk?.execute::<PrimitiveArray>(&mut ctx)?;

        assert_eq!(primitive.into_buffer::<i32>().as_slice(), [1]);
        assert!(iter.next().is_none());
        assert_eq!(projection_masks.lock().as_slice(), [1]);
        Ok(())
    }

    /// Rows reserved against a shared limit cannot be released back, so a projection failure after
    /// reservation must end the scan rather than let a later split spend the freed budget.
    #[test]
    fn projection_error_after_reservation_terminates_the_limited_scan() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let projection_masks = Arc::new(Mutex::new(Vec::new()));
        let reader = Arc::new(
            TestLayoutReader::new(2)
                .with_split_size(1)
                .with_projection_masks(Arc::clone(&projection_masks))
                .with_fail_first_projection(),
        );
        let stream = ScanBuilder::new(session, reader)
            .with_filter(root())
            // A budget of two leaves room for the second matching split. Continuing after the
            // first projection failure would therefore yield a second stream item.
            .with_limit(2)
            .with_ordered(false)
            .into_stream()?;
        let mut iter = runtime.block_on_stream(stream);

        assert!(matches!(iter.next(), Some(Err(_))));
        assert!(iter.next().is_none());
        assert!(projection_masks.lock().contains(&1));
        Ok(())
    }

    #[test]
    fn projection_errors_are_stream_items() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let projection_masks = Arc::new(Mutex::new(Vec::new()));
        let reader = Arc::new(
            TestLayoutReader::new(1)
                .with_projection_masks(Arc::clone(&projection_masks))
                .with_projection_error(),
        );
        let stream = ScanBuilder::new(session, reader)
            .with_filter(root())
            .with_limit(1)
            .into_stream()?;
        let mut iter = runtime.block_on_stream(stream);

        assert!(matches!(iter.next(), Some(Err(_))));
        assert!(iter.next().is_none());
        assert_eq!(projection_masks.lock().as_slice(), [1]);
        Ok(())
    }

    #[test]
    fn prepared_scan_limits_filtered_results() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let reader = Arc::new(
            TestLayoutReader::new(8)
                .with_split_size(2)
                .with_keep_row(keep_odd),
        );

        let scan = ScanBuilder::new(session, reader)
            .with_filter(root())
            .with_limit(3)
            .prepare()?;
        let values = collect_scan_values(scan.execute_array_iter(None, &runtime)?)?;
        drain_runtime(&runtime);

        assert_eq!(values, [1, 3, 5]);
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
            split_range: &SplitRange,
            splits: &mut RowSplits,
        ) -> VortexResult<()> {
            self.register_splits_calls.fetch_add(1, Ordering::Relaxed);
            let _guard = self.gate.lock();
            splits.push(split_range.root_row_range().end);
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

        fn as_any(&self) -> &dyn std::any::Any {
            self
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
        let session = session_with_handle(runtime.handle());

        let mut stream = ScanBuilder::new(session, reader).into_stream().unwrap();

        let (send, recv) = mpsc::channel::<bool>();
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

    #[test]
    fn into_stream_with_row_range() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(SplittingLayoutReader::new(Arc::clone(&calls)));

        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());

        let stream = ScanBuilder::new(session, reader)
            .with_row_range(1..3)
            .into_stream()?;
        let mut iter = runtime.block_on_stream(stream);

        let mut values = Vec::new();
        for chunk in &mut iter {
            let prim = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
            values.extend(prim.into_buffer::<i32>().iter().copied());
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(values.as_ref(), [1, 2]);

        Ok(())
    }
}
