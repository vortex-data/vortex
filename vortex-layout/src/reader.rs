use std::sync::Arc;

use vortex_array::ArrayData;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::operations::Operation;
use crate::{LayoutData, RowMask};

pub type EvalOp = Box<dyn Operation<Output = ArrayData>>;

/// A [`LayoutReader`] is an instance of a [`LayoutData`] that can cache state across multiple
/// operations.
pub trait LayoutReader {
    /// Returns the [`LayoutData`] of this reader.
    fn layout(&self) -> &LayoutData;

    /// Creates a new evaluator for the layout. It is expected that the evaluator makes use of
    /// shared state from the [`LayoutReader`] for caching and other optimisations.
    //
    // NOTE(ngates): we have chosen a general "run this expression" API instead of  separate
    //  `filter(row_mask, expr) -> row_mask` + `project(row_mask, field_mask)` APIs.
    //  The reason for this is so we can eventually support cell-level push-down.
    //  If we only projected using a field mask, then it means we need to download all the data
    //  for the rows of field present in the row mask. When I say cell-level push-down, I mean
    //  we can slice the cell directly out of storage using an API like
    //  `SegmentReader::read(segment_id, byte_range: Range<usize>)`. This is a highly advanced
    //  use-case, but can prove invaluable for large cell values such as images and video.
    //  If instead we make the projection API `project(row_mask, expr)`, then identical to the
    //  filter API and there's now no point having two. Hence: `evaluate(row_mask, expr)`.
    fn create_evaluator(self: Arc<Self>, row_mask: RowMask, expr: ExprRef) -> VortexResult<EvalOp>;
}

pub trait LayoutScanExt: LayoutReader {
    /// Box the layout scan.
    fn into_arc(self) -> Arc<dyn LayoutReader>
    where
        Self: Sized + 'static,
    {
        Arc::new(self) as _
    }

    /// Returns the DType of the layout.
    fn dtype(&self) -> &DType {
        self.layout().dtype()
    }
}

impl<L: LayoutReader> LayoutScanExt for L {}

impl dyn LayoutReader + 'static {
    /// Perform a scan over a row-range of the layout.
    #[cfg(feature = "vortex-scan")]
    pub fn range_scan(
        self: Arc<dyn LayoutReader>,
        range_scan: vortex_scan::RangeScan,
    ) -> impl Operation<Output = ArrayData> {
        crate::scan::LayoutRangeScan::new(self, range_scan)
    }
}
