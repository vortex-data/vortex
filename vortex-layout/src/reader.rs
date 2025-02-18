use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::{Layout, RowMask};

/// A [`LayoutReader`] is an instance of a [`Layout`] that can cache state across multiple
/// operations.
///
/// Since different row ranges of the reader may be evaluated by different threads, it is required
/// to be both `Send` and `Sync`.
pub trait LayoutReader: Send + Sync + ExprEvaluator {
    /// Returns the [`Layout`] of this reader.
    fn layout(&self) -> &Layout;
}

impl LayoutReader for Arc<dyn LayoutReader + 'static> {
    fn layout(&self) -> &Layout {
        self.as_ref().layout()
    }
}

/// A trait for evaluating expressions against a [`LayoutReader`].
///
/// FIXME(ngates): what if this was evaluating_predicate(mask, expr) -> mask,
///  evaluate_filter(mask, scan) -> Array, and evaluate_projection(mask, expr) -> Array?
#[async_trait]
pub trait ExprEvaluator: Send + Sync {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array>;

    /// Refine the row mask by evaluating any pruning. This should be relatively cheap, statistics
    /// based evaluation, and returns an approximate result.
    async fn prune_mask(&self, row_mask: RowMask, _expr: ExprRef) -> VortexResult<RowMask> {
        Ok(row_mask)
    }
}

#[async_trait]
impl ExprEvaluator for Arc<dyn LayoutReader + 'static> {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        self.as_ref().evaluate_expr(row_mask, expr).await
    }

    async fn prune_mask(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<RowMask> {
        self.as_ref().prune_mask(row_mask, expr).await
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
