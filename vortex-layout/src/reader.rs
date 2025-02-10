use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::Layout;

/// A [`LayoutReader`] is an instance of a [`Layout`] that can cache state across multiple
/// operations.
pub trait LayoutReader {
    /// Returns the [`Layout`] of this reader.
    fn layout(&self) -> &Layout;

    /// Returns a [`LayoutRangeReader`] for the given row range and field mask.
    fn range_reader(&self, row_range: Range<u64>) -> Arc<dyn LayoutRangeReader>;
}

impl LayoutReader for Arc<dyn LayoutReader> {
    fn layout(&self) -> &Layout {
        self.as_ref().layout()
    }

    fn range_reader(&self, row_range: Range<u64>) -> Arc<dyn LayoutRangeReader> {
        self.as_ref().range_reader(row_range)
    }
}

/// A trait for reading a range of rows and specific field mask from a [`LayoutReader`].
#[async_trait]
pub trait LayoutRangeReader: 'static + Send + Sync {
    fn row_range(&self) -> &Range<u64>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        usize::try_from(self.row_range().end - self.row_range().start)
            .vortex_expect("row range must fit within usize")
    }

    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array>;
}

pub trait LayoutReaderExt: LayoutReader {
    /// Box the layout scan.
    fn into_arc(self) -> Arc<dyn LayoutReader>
    where
        Self: Sized + 'static,
    {
        Arc::new(self) as _
    }

    /// Returns the row count of the layout.
    fn row_count(&self) -> u64 {
        self.layout().row_count()
    }

    /// Returns the DType of the layout.
    fn dtype(&self) -> &DType {
        self.layout().dtype()
    }
}

impl<L: LayoutReader> LayoutReaderExt for L {}
