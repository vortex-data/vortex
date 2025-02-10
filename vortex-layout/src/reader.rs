use std::iter::FilterMap;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::Array;
use vortex_dtype::{DType, FieldMask, FieldPath};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;
use vortex_scan::RowMask;

use crate::Layout;

/// A [`LayoutReader`] is an instance of a [`Layout`] that can cache state across multiple
/// operations.
///
/// Since different row ranges of the reader may be evaluated by different threads, it is required
/// to be both `Send` and `Sync`.
pub trait LayoutReader: 'static + Send + Sync {
    /// Returns the [`Layout`] of this reader.
    fn layout(&self) -> &Layout;

    /// Returns a [`LayoutRangeReader`] for the given row range and field mask.
    fn range_reader(
        &self,
        row_range: Range<u64>,
        field_mask: Arc<[FieldMask]>,
    ) -> Arc<dyn LayoutRangeReader>;
}

impl LayoutReader for Arc<dyn LayoutReader> {
    fn layout(&self) -> &Layout {
        self.as_ref().layout()
    }

    fn range_reader(
        &self,
        row_range: Range<u64>,
        field_mask: Arc<[FieldMask]>,
    ) -> Arc<dyn LayoutRangeReader> {
        self.as_ref().range_reader(row_range, field_mask)
    }
}

/// A trait for reading a range of rows and specific field mask from a [`LayoutReader`].
#[async_trait]
pub trait LayoutRangeReader: 'static + Send + Sync {
    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array>;
}

/// A trait for evaluating expressions against a [`LayoutReader`].
#[async_trait]
pub trait ExprEvaluator {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array>;
}

#[async_trait]
impl ExprEvaluator for Arc<dyn LayoutReader + 'static> {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        todo!()
    }
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
