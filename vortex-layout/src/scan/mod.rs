use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::{stream, FutureExt, Stream};
use itertools::Itertools;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_buffer::Buffer;
use vortex_expr::{ExprRef, Identity};
mod arc_iter;
mod split_by;
use futures::StreamExt;
pub use split_by::*;
use vortex_array::{Array, ContextRef};
use vortex_dtype::{DType, Field, FieldMask, FieldPath};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_mask::Mask;
use vortex_scan::{RowMask, Scanner};

use crate::scan::arc_iter::ArcIter;
use crate::segments::AsyncSegmentReader;
use crate::{Layout, LayoutReader};

pub trait ScanDriver: 'static + Sized {
    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader>;

    fn drive(
        self,
        stream: impl Stream<Item = BoxFuture<'static, VortexResult<Option<Array>>>> + 'static,
        _concurrency: usize,
    ) -> VortexResult<impl Stream<Item = VortexResult<Array>> + 'static> {
        // The default driver implementation simply wraps the stream up in an ArrayStreamAdapter.
        Ok(stream
            //.buffered(concurrency)
            .filter_map(|result| async { result.await.transpose() }))
    }
}

/// A struct for building a scan operation.
pub struct Scan<D> {
    driver: D,
    layout: Layout,
    ctx: ContextRef, // TODO(ngates): store this on larger context on Layout
    projection: ExprRef,
    filter: Option<ExprRef>,
    row_indices: Option<Buffer<u64>>,
    split_by: SplitBy,
    concurrency: usize,
}

impl<D> Scan<D> {
    pub fn new(driver: D, layout: Layout, ctx: ContextRef) -> Self {
        Self {
            driver,
            layout,
            ctx,
            projection: Identity::new_expr(),
            filter: None,
            row_indices: None,
            split_by: SplitBy::Layout,
            concurrency: 10,
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

    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Compute a mask of field paths referenced by this scan.
    fn field_mask(&self, scope_dtype: &DType) -> VortexResult<Vec<FieldMask>> {
        // TODO(joe): simplify this expr once
        let projection = simplify_typed(self.projection.clone(), scope_dtype)?;
        let filter = self
            .filter
            .clone()
            .map(|f| simplify_typed(f, scope_dtype))
            .transpose()?;

        let Some(struct_dtype) = scope_dtype.as_struct() else {
            return Ok(vec![FieldMask::All]);
        };

        let projection_mask = immediate_scope_access(&projection, struct_dtype)?;
        let filter_mask = filter
            .map(|f| immediate_scope_access(&f, struct_dtype))
            .transpose()?
            .unwrap_or_default();

        Ok(projection_mask
            .union(&filter_mask)
            .cloned()
            .map(|c| FieldMask::Prefix(FieldPath::from(Field::Name(c))))
            .collect_vec())
    }
}

impl<D: ScanDriver> Scan<D> {
    /// Perform the scan operation and return a stream of arrays.
    pub fn into_stream(self) -> VortexResult<impl ArrayStream + 'static> {
        let field_mask = self.field_mask(self.layout.dtype())?;
        let splits: Arc<[Range<u64>]> = self
            .split_by
            .splits(&self.layout, &field_mask)?
            .into_iter()
            .collect_vec()
            .into();

        let row_indices = self.row_indices.clone();

        let row_masks = ArcIter::new(splits).filter_map(move |row_range| {
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
                        usize::try_from(idx - row_range.start).vortex_expect("index within range")
                    })
                    .collect(),
            );
            Some(RowMask::new(filter_mask, row_range.start))
        });

        self.into_stream_with_masks(row_masks)
    }

    fn into_stream_with_masks<R>(
        self,
        row_masks: R,
    ) -> VortexResult<impl ArrayStream + 'static + use<D, R>>
    where
        R: Iterator<Item = RowMask> + Send + 'static,
    {
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

        let exec_stream = stream::iter(row_masks).map(move |row_mask| {
            match scanner.clone().range_scanner(row_mask) {
                Ok(range_scan) => {
                    let reader = reader.clone();
                    async move {
                        range_scan
                            .evaluate_async(|row_mask, expr| reader.evaluate_expr(row_mask, expr))
                            .await
                    }
                    .boxed()
                }
                Err(e) => futures::future::ready(Err(e)).boxed(),
            }
        });

        let stream = self.driver.drive(exec_stream, self.concurrency)?;

        Ok(ArrayStreamAdapter::new(result_dtype, stream))
    }
}
