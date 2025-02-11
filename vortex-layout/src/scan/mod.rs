use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use futures::stream::FuturesOrdered;
use futures::{stream, FutureExt, Stream};
use itertools::Itertools;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};
use vortex_buffer::Buffer;
use vortex_expr::{ExprRef, Identity};
mod split_by;
pub mod unified;

use futures::StreamExt;
pub use split_by::*;
use vortex_array::{Array, ContextRef};
use vortex_dtype::{DType, Field, FieldMask, FieldPath};
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_mask::Mask;
use vortex_scan::{RowMask, Scanner};

use crate::scan::unified::UnifiedDriverStream;
use crate::segments::AsyncSegmentReader;
use crate::{ExprEvaluator, Layout, LayoutReader};

pub trait ScanTask {
    fn execute(&self, segments: &dyn AsyncSegmentReader) -> BoxFuture<()>;
}

pub trait ScanDriver: 'static + Sized {
    type Options: Default;

    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader>;

    fn spawn_task(
        &self,
        task: BoxFuture<'static, VortexResult<()>>,
    ) -> BoxFuture<'static, VortexResult<()>>;

    /// Return a future that drives the I/O stream for the segment reader.
    /// The future should return when the stream is complete, and can return an error to
    /// terminate the scan early.
    ///
    /// It is recommended that I/O is spawned and processed on its own thread, with this driver
    /// serving only as a mechanism to signal completion or error. There is no guarantee around
    /// how frequently this future will be polled, so it should not be used to drive I/O.
    ///
    /// TODO(ngates): make this a future
    fn io_stream(self) -> impl Stream<Item = VortexResult<()>> + 'static;
}

/// A struct for building a scan operation.
pub struct ScanBuilder<D: ScanDriver> {
    driver: D,
    driver_options: Option<D::Options>,
    layout: Layout,
    ctx: ContextRef, // TODO(ngates): store this on larger context on Layout
    projection: ExprRef,
    filter: Option<ExprRef>,
    row_indices: Option<Buffer<u64>>,
    split_by: SplitBy,
}

impl<D: ScanDriver> ScanBuilder<D> {
    pub fn new(driver: D, layout: Layout, ctx: ContextRef) -> Self {
        Self {
            driver,
            driver_options: None,
            layout,
            ctx,
            projection: Identity::new_expr(),
            filter: None,
            row_indices: None,
            split_by: SplitBy::Layout,
        }
    }

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

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.row_indices = Some(row_indices);
        self
    }

    pub fn with_some_row_indices(mut self, row_indices: Option<Buffer<u64>>) -> Self {
        self.row_indices = row_indices;
        self
    }

    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    pub fn with_options(mut self, options: D::Options) -> Self {
        self.driver_options = Some(options);
        self
    }

    pub fn build(self) -> VortexResult<Scan<D>> {
        let projection = simplify_typed(self.projection, self.layout.dtype())?;
        let filter = self
            .filter
            .map(|f| simplify_typed(f, self.layout.dtype()))
            .transpose()?;
        let field_mask = field_mask(&projection, filter.as_ref(), self.layout.dtype())?;

        let row_indices = self.row_indices.clone();
        let splits = self.split_by.splits(&self.layout, &field_mask)?;
        let row_masks = splits
            .into_iter()
            .filter_map(move |row_range| {
                let Some(row_indices) = &row_indices else {
                    // If there is no row indices filter, then take the whole range
                    return Some(RowMask::new_valid_between(row_range.start, row_range.end));
                };

                // Otherwise, find the indices that are within the row range.
                if row_indices
                    .first()
                    .is_some_and(|&first| first >= row_range.end)
                    || row_indices
                        .last()
                        .is_some_and(|&last| row_range.start >= last)
                {
                    return None;
                }

                // For the given row range, find the indices that are within the row_indices.
                let start_idx = row_indices
                    .binary_search(&row_range.start)
                    .unwrap_or_else(|x| x);
                let end_idx = row_indices
                    .binary_search(&row_range.end)
                    .unwrap_or_else(|x| x);

                if start_idx == end_idx {
                    // No rows in range
                    return None;
                }

                // Construct a row mask for the range.
                let filter_mask = Mask::from_indices(
                    usize::try_from(row_range.end - row_range.start)
                        .vortex_expect("Split ranges are within usize"),
                    row_indices[start_idx..end_idx]
                        .iter()
                        .map(|&idx| {
                            usize::try_from(idx - row_range.start)
                                .vortex_expect("index within range")
                        })
                        .collect(),
                );
                Some(RowMask::new(filter_mask, row_range.start))
            })
            .collect_vec();

        Ok(Scan {
            driver: self.driver,
            layout: self.layout,
            ctx: self.ctx,
            projection,
            filter,
            row_masks,
        })
    }

    /// Perform the scan operation and return a stream of arrays.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        self.build()?.into_array_stream()
    }

    pub async fn into_array(self) -> VortexResult<Array> {
        self.into_array_stream()?.into_array().await
    }
}

pub struct Scan<D> {
    driver: D,
    layout: Layout,
    ctx: ContextRef,
    // Guaranteed to be simplified
    projection: ExprRef,
    // Guaranteed to be simplified
    filter: Option<ExprRef>,
    row_masks: Vec<RowMask>,
}

impl<D: ScanDriver> Scan<D> {
    /// Perform the scan operation and return a stream of arrays.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        let scanner = Arc::new(Scanner::new(
            self.layout.dtype().clone(),
            self.projection,
            self.filter,
        )?);

        let result_dtype = scanner.result_dtype().clone();

        // Create a single LayoutReader that is reused for the entire scan.
        let reader: Arc<dyn LayoutReader> = self
            .layout
            .reader(self.driver.segment_reader(), self.ctx.clone())?;

        let mut tasks = FuturesOrdered::new();
        let mut results = vec![];
        for row_mask in self.row_masks.into_iter() {
            let result: Arc<Mutex<Option<VortexResult<Array>>>> = Arc::new(Mutex::new(None));
            results.push(result.clone());

            let scanner = scanner.clone();
            let reader = reader.clone();
            let fut = async move {
                let array = {
                    scanner
                        .clone()
                        .range_scanner(row_mask)?
                        .evaluate_async(move |row_mask, expr| {
                            let reader = reader.clone();
                            async move { reader.evaluate_expr(row_mask, expr).await }
                        })
                        .await
                };
                let mut lock = result.lock().map_err(|_| vortex_err!("lock error"))?;
                *lock = array.transpose();
                Ok::<_, VortexError>(())
            };

            tasks.push_back(self.driver.spawn_task(fut.boxed()));
        }

        let array_stream =
            tasks
                .zip(stream::iter(results))
                .filter_map(|(task, result)| async move {
                    // Propagate any task errors
                    if let Err(e) = task {
                        return Some(Err(e));
                    }

                    // Otherwise, the result mutex should hold our answer
                    match result.lock().map_err(|_| vortex_err!("lock error")) {
                        Ok(mut lock) => lock.take(),
                        Err(e) => Some(Err(e)),
                    }
                });

        let io_stream = self.driver.io_stream();
        let unified = UnifiedDriverStream {
            exec_stream: array_stream,
            io_stream,
        };

        Ok(ArrayStreamAdapter::new(result_dtype, unified))
    }

    pub async fn into_array(self) -> VortexResult<Array> {
        self.into_array_stream()?.into_array().await
    }
}

/// Compute a mask of field paths referenced by this scan.
///
/// Projection and filter must be pre-simplified.
fn field_mask(
    projection: &ExprRef,
    filter: Option<&ExprRef>,
    scope_dtype: &DType,
) -> VortexResult<Vec<FieldMask>> {
    let Some(struct_dtype) = scope_dtype.as_struct() else {
        return Ok(vec![FieldMask::All]);
    };

    let projection_mask = immediate_scope_access(projection, struct_dtype)?;
    let filter_mask = filter
        .map(|f| immediate_scope_access(f, struct_dtype))
        .transpose()?
        .unwrap_or_default();

    Ok(projection_mask
        .union(&filter_mask)
        .cloned()
        .map(|c| FieldMask::Prefix(FieldPath::from(Field::Name(c))))
        .collect_vec())
}
