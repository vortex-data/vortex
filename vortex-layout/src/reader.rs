use std::sync::Arc;

use vortex_dtype::DType;
use vortex_scan::AsyncEvaluator;

use crate::LayoutData;

/// A [`LayoutReader`] is an instance of a [`LayoutData`] that can cache state across multiple
/// operations.
///
/// Since different row ranges of the reader may be evaluated by different threads, it is required
/// to be both `Send` and `Sync`.
pub trait LayoutReader: Send + Sync + AsyncEvaluator {
    /// Returns the [`LayoutData`] of this reader.
    fn layout(&self) -> &LayoutData;

    /// Returns the [`AsyncEvaluator`] for this reader.
    fn evaluator(&self) -> &dyn AsyncEvaluator;
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
