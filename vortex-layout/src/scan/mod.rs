// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
pub use executor::*;
use futures::executor::LocalPool;
use futures::task::LocalSpawnExt;
use futures::{Stream, StreamExt, stream};
use itertools::Itertools;
pub use selection::*;
pub use split_by::*;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::stats::StatsSet;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::{ArrayRef, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExprRef, root};
use vortex_metrics::VortexMetrics;

use crate::layouts::filter::FilterLayoutReader;
use crate::layouts::row_id::RowIdLayoutReader;
use crate::scan::tasks::{TaskContext, split_exec};
use crate::{LayoutReader, LayoutReaderRef};

mod executor;
pub mod row_mask;
mod selection;
mod split_by;
mod tasks;

/// A struct for building a scan operation.
pub struct ScanBuilder<A> {
    layout_reader: LayoutReaderRef,
    projection: ExprRef,
    filter: Option<ExprRef>,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    // TODO(joe): replace this is usage of row_id selection, see
    selection: Selection,
    /// How to split the file for concurrent processing.
    split_by: SplitBy,
    /// The number of splits to make progress on concurrently.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// The executor used to spawn each split task.
    executor: Option<Arc<dyn TaskExecutor>>,
    metrics: VortexMetrics,
    /// Should we try to prune the file (using stats) on open.
    file_stats: Option<Arc<[StatsSet]>>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<usize>,
    /// Include the row and file index in the scope of the scan.
    ///
    /// See also [crate::layouts::row_id].
    row_index: bool,
}

impl<A: 'static + Send + Sync> ScanBuilder<A> {
    pub fn with_filter(mut self, filter: ExprRef) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_some_filter(mut self, filter: Option<ExprRef>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_projection(mut self, projection: ExprRef) -> Self {
        self.projection = projection;
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

    pub fn with_row_index(mut self) -> Self {
        self.row_index = true;
        self
    }

    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// The number of row splits to make progress on concurrently, must be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    /// Spawn each CPU task onto the given Tokio runtime.
    ///
    /// Note that this is an odd use of the Tokio runtime. Typically, it is used predominantly
    /// for I/O bound tasks.
    #[cfg(feature = "tokio")]
    pub fn with_tokio_executor(mut self, handle: tokio::runtime::Handle) -> Self {
        self.executor = Some(Arc::new(handle));
        self
    }

    pub fn with_executor(mut self, executor: Arc<dyn TaskExecutor>) -> Self {
        self.executor = Some(executor);
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

    /// Map each split of the scan. The function will be run on the spawned task.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            layout_reader: self.layout_reader,
            projection: self.projection,
            filter: self.filter,
            row_range: self.row_range,
            selection: self.selection,
            split_by: self.split_by,
            concurrency: self.concurrency,
            map_fn: Arc::new(move |a| map_fn(old_map_fn(a)?)),
            executor: self.executor,
            metrics: self.metrics,
            file_stats: self.file_stats,
            limit: self.limit,
            row_index: self.row_index,
        }
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    #[allow(clippy::unused_enumerate_index)]
    pub fn build(
        mut self,
    ) -> VortexResult<(DType, Vec<impl Future<Output = VortexResult<Option<A>>>>)> {
        if self.filter.is_some() && self.limit.is_some() {
            vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
        }

        // The ultimate short circuit
        if self.limit.is_some_and(|l| l == 0) {
            let dtype = self
                .projection
                .return_dtype(self.layout_reader.scope_dtype())?;
            return Ok((dtype, Vec::new()));
        }

        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let mut layout_reader = self.layout_reader;

        if self.filter.is_some() {
            layout_reader = Arc::new(FilterLayoutReader::new(layout_reader));
        }
        if self.row_index {
            layout_reader = Arc::new(RowIdLayoutReader::new(layout_reader));
        }

        let scope_dtype = layout_reader.scope_dtype();

        // Normalize and simplify the expressions.
        let projection = simplify_typed(self.projection.clone(), scope_dtype)?;
        let filter = self
            .filter
            .clone()
            .map(|f| simplify_typed(f, scope_dtype))
            .transpose()?;

        // Construct field masks and compute the row splits of the scan.
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, filter.as_ref(), layout_reader.dtype())?;
        let field_mask: Vec<_> = [filter_mask, projection_mask].concat();
        let splits = self.split_by.splits(layout_reader.as_ref(), &field_mask)?;

        // Create a task that executes the full scan pipeline for each split.
        let split_tasks = splits
            .into_iter()
            .filter_map(|split_range| {
                let ctx = Arc::new(TaskContext {
                    row_range: self.row_range.clone(),
                    selection: self.selection.clone(),
                    filter: self.filter.clone(),
                    reader: layout_reader.clone(),
                    projection: projection.clone(),
                    mapper: self.map_fn.clone(),
                    task_executor: None,
                });

                if self.limit.is_some_and(|l| l == 0) {
                    None
                } else {
                    Some(split_exec(ctx, split_range, self.limit.as_mut()))
                }
            })
            .try_collect()?;

        let dtype = self.projection.return_dtype(layout_reader.scope_dtype())?;
        Ok((dtype, split_tasks))
    }

    pub fn into_stream(self) -> VortexResult<impl Stream<Item = VortexResult<A>> + 'static> {
        let concurrency = self.concurrency;
        let (_, split_tasks) = self.build()?;
        Ok(stream::iter(split_tasks)
            .buffered(concurrency)
            .filter_map(|r| async move { r.transpose() }))
    }
}

impl ScanBuilder<ArrayRef> {
    pub fn new(layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            layout_reader,
            projection: root(),
            filter: None,
            row_range: None,
            selection: Default::default(),
            split_by: SplitBy::Layout,
            // How many row splits to make progress on concurrently (not necessarily in parallel,
            // that is decided by the TaskExecutor).
            concurrency: 16,
            map_fn: Arc::new(Ok),
            executor: None,
            metrics: Default::default(),
            file_stats: None,
            limit: None,
            row_index: false,
        }
    }

    /// Map the scan into a stream of Arrow [`RecordBatch`].
    pub fn map_to_record_batch(self, schema: SchemaRef) -> ScanBuilder<RecordBatch> {
        self.map(move |array| {
            let st = array.to_struct()?;
            st.into_record_batch_with_schema(schema.as_ref())
        })
    }

    /// Returns a stream over the scan with each CPU task polled on the current thread as per
    /// the behaviour of [`futures::stream::Buffered`].
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        let concurrency = self.concurrency;
        let (dtype, split_tasks) = self.build()?;
        Ok(ArrayStreamAdapter::new(
            dtype,
            stream::iter(split_tasks)
                .buffered(concurrency)
                .filter_map(|r| async move { r.transpose() }),
        ))
    }

    /// Returns a blocking iterator over the scan.
    ///
    /// All work will be performed on the current thread, with tasks interleaved per the
    /// configured concurrency. Any configured executor will be ignored.
    pub fn into_array_iter(self) -> VortexResult<impl ArrayIterator + 'static> {
        let concurrency = self.concurrency;

        let mut local_pool = LocalPool::new();
        let spawner = local_pool.spawner();

        let (dtype, split_tasks) = self.build()?;
        let mut stream = stream::iter(split_tasks)
            .map(move |task| {
                spawner
                    .spawn_local_with_handle(task)
                    .map_err(|e| vortex_err!("Failed to spawn task: {e}"))
                    .vortex_expect("Failed to spawn task")
            })
            .buffered(concurrency)
            .filter_map(|a| async move { a.transpose() })
            .boxed_local();

        Ok(ArrayIteratorAdapter::new(
            dtype,
            iter::from_fn(move || local_pool.run_until(stream.next())),
        ))
    }
}

/// Compute masks of field paths referenced by the projection and filter in the scan.
///
/// Projection and filter must be pre-simplified.
fn filter_and_projection_masks(
    projection: &ExprRef,
    filter: Option<&ExprRef>,
    dtype: &DType,
) -> VortexResult<(Vec<FieldMask>, Vec<FieldMask>)> {
    let Some(struct_dtype) = dtype.as_struct() else {
        return Ok(match filter {
            Some(_) => (vec![FieldMask::All], vec![FieldMask::All]),
            None => (Vec::new(), vec![FieldMask::All]),
        });
    };
    let projection_mask = immediate_scope_access(projection, struct_dtype)?;
    Ok(match filter {
        None => (
            Vec::new(),
            projection_mask.into_iter().map(to_field_mask).collect_vec(),
        ),
        Some(f) => {
            let filter_mask = immediate_scope_access(f, struct_dtype)?;
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
