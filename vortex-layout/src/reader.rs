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
    /// Construct an expression evaluation future for the given row range, expression, and mask.
    ///
    /// The row range is relative to the start of the layout.
    ///
    /// Note: this function returns a future with a static lifetime. It is recommended that
    /// after producing evaluation futures for each desired row range, that the original
    /// [`LayoutReader`] is dropped. This does two things:
    ///  * Any caches will be automatically cleaned up at the earliest opportunity.
    ///  * Any segments that were requested at creation of the future, but are not longer needed
    ///    (for example, those that are pruned away with statistics), will be dropped. Enabling
    ///    the segment reader to cancel any in-flight or upcoming requests.
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>;

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        todo!()
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        todo!()
    }

    fn projection_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        todo!()
    }
}

#[async_trait]
impl ExprEvaluator for Arc<dyn LayoutReader> {
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>> {
        self.as_ref().evaluate_expr2(row_range, expr, mask)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        self.as_ref().pruning_evaluation(row_range, expr)
    }

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

#[async_trait]
pub trait MaskEvaluation: 'static + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask>;
}

#[async_trait]
pub trait ArrayEvaluation: 'static + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef>;
}
