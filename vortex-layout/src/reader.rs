use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{SharedVortexResult, VortexError, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::Layout;

/// A [`LayoutReader`] is an instance of a [`Layout`] that can cache state across multiple
/// operations.
///
/// Since different row ranges of the reader may be evaluated by different threads, it is required
/// to be both `Send` and `Sync`.
pub trait LayoutReader: 'static + Send + Sync + ExprEvaluator {
    /// Returns the [`Layout`] of this reader.
    fn layout(&self) -> &Layout;

    /// Returns the row count of the layout.
    fn row_count(&self) -> u64 {
        self.layout().row_count()
    }

    /// Returns the DType of the layout.
    fn dtype(&self) -> &DType {
        self.layout().dtype()
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>>;
}

impl LayoutReader for Arc<dyn LayoutReader> {
    fn layout(&self) -> &Layout {
        self.as_ref().layout()
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        self.as_ref().children()
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
}

impl<L: LayoutReader> LayoutReaderExt for L {}

pub type MaskFuture = Shared<BoxFuture<'static, SharedVortexResult<Mask>>>;

/// Create a resolved [`MaskFuture`] from a [`Mask`].
pub fn mask_future_ready(mask: Mask) -> MaskFuture {
    async move { Ok::<_, Arc<VortexError>>(mask) }
        .boxed()
        .shared()
}

/// A trait for evaluating expressions against a [`LayoutReader`].
///
/// FIXME(ngates): what if this was evaluating_predicate(mask, expr) -> mask,
///  evaluate_filter(mask, scan) -> Array, and evaluate_projection(mask, expr) -> Array?
#[async_trait]
pub trait ExprEvaluator: Send + Sync {
    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>>;

    fn projection_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>>;
}

#[async_trait]
impl ExprEvaluator for Arc<dyn LayoutReader> {
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        self.as_ref().filter_evaluation(row_range, expr)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        self.as_ref().projection_evaluation(row_range, expr)
    }
}

/// Refines the given mask, returning a mask equal in length to the input mask.
#[async_trait]
pub trait MaskEvaluation: 'static + Send + Sync {
    /// Returns an approximate refinement of the mask with relatively cheap computation.
    async fn invoke_approx(&self, mask: Mask) -> VortexResult<Mask>;

    async fn invoke(&self, mask: Mask) -> VortexResult<Mask>;
}

/// Evaluates an expression against an array, returning an array equal in length to the true count
/// of the input mask.
#[async_trait]
pub trait ArrayEvaluation: 'static + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef>;
}
