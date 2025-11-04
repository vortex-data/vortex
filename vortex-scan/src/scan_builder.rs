// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::Stream;
use futures::future::BoxFuture;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::stats::StatsSet;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexResult, vortex_bail};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed;
use vortex_expr::{Expression, root};
use vortex_io::runtime::BlockingRuntime;
use vortex_layout::layouts::row_idx::RowIdxLayoutReader;
use vortex_layout::{LayoutReader, LayoutReaderRef};
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;

use crate::RepeatedScan;
use crate::selection::Selection;
use crate::split_by::SplitBy;
use crate::splits::{Splits, attempt_split_ranges};

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
    metrics: VortexMetrics,
    /// Should we try to prune the file (using stats) on open.
    file_stats: Option<Arc<[StatsSet]>>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<usize>,
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
            metrics: Default::default(),
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
            metrics: self.metrics,
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
        layout_reader = Arc::new(RowIdxLayoutReader::new(self.row_offset, layout_reader));

        // Normalize and simplify the expressions.
        let projection = simplify_typed(self.projection, layout_reader.dtype())?;
        let filter = self
            .filter
            .map(|f| simplify_typed(f, layout_reader.dtype()))
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
        self.prepare()?.execute_stream(None)
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
