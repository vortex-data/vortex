// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::immediate_access::immediate_scope_access;
use vortex_array::expr::root;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::Field;
use vortex_dtype::FieldMask;
use vortex_dtype::FieldName;
use vortex_dtype::FieldPath;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::BlockingRuntime;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderRef;
use vortex_layout::layouts::row_idx::RowIdxLayoutReader;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;

use crate::RepeatedScan;
use crate::selection::Selection;
use crate::split_by::SplitBy;
use crate::splits::Splits;
use crate::splits::attempt_split_ranges;

/// A struct for building a scan operation.
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
    metrics: VortexMetrics,
    /// Maximal number of rows to read (after filtering)
    limit: Option<usize>,
    /// The row-offset assigned to the first row of the file. Used by the `row_idx` expression,
    /// but not by the scan [`Selection`] which remains relative.
    row_offset: u64,
}

impl ScanBuilder {
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
            metrics: Default::default(),
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

    /// The number of row splits to make progress on concurrently per-thread, must
    /// be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
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

    pub fn prepare(self) -> VortexResult<RepeatedScan> {
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
            self.limit,
            dtype,
        ))
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    pub fn build(self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>> {
        // The ultimate short circuit
        if self.limit.is_some_and(|l| l == 0) {
            return Ok(vec![]);
        }

        self.prepare()?.execute(None)
    }

    /// Returns a [`Stream`] with tasks spawned onto the session's runtime handle.
    pub fn into_stream(self) -> VortexResult<impl Stream<Item = VortexResult<ArrayRef>>> {
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
    Stream(BoxStream<'static, VortexResult<ArrayRef>>),
    Error(Option<vortex_error::VortexError>),
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
                    match builder
                        .prepare()
                        .and_then(move |scan| scan.execute_stream(None).map(|s| s.boxed()))
                    {
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
pub(crate) fn filter_and_projection_masks(
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
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_array::MaskFuture;
    use vortex_array::expr::Expression;
    use vortex_dtype::DType;
    use vortex_dtype::FieldMask;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_layout::ArrayFuture;
    use vortex_layout::LayoutReader;
    use vortex_mask::Mask;

    use super::ScanBuilder;

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
        let reader = Arc::new(CountingLayoutReader::new(calls.clone()));

        let runtime = SingleThreadRuntime::default();
        let session = crate::test::SESSION.clone().with_handle(runtime.handle());

        let _stream = ScanBuilder::new(session, reader).into_stream().unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}
