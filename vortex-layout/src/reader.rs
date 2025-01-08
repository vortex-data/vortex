use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::scanner::EvalOp;
use crate::{LayoutData, RowMask};

/// A [`LayoutReader`] is an instance of a [`LayoutData`] that can cache state across multiple
/// operations.
pub trait LayoutReader {
    /// Returns the [`LayoutData`] of this reader.
    fn layout(&self) -> &LayoutData;

    /// Creates a new evaluator for the layout. It is expected that the evaluator makes use of
    /// shared state from the [`LayoutReader`] for caching and other optimisations.
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
