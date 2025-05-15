use std::ops::Range;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_assert};
use vortex_layout::{
    ArrayEvaluation, ExprEvaluator, Layout, LayoutReader, MaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
};

/// Layout Reader
pub struct BBoxReader {
    /// The layout definition provided from file or from client.
    layout: Layout,
}

impl BBoxReader {
    pub fn try_new(layout: Layout) -> VortexResult<Self> {
        vortex_assert!(layout.id() == crate::layout::ID, "Invalid layout ID");

        Ok(Self { layout })
    }
}

impl ExprEvaluator for BBoxReader {
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &vortex_expr::ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        // TODO(aduffy): any expressions that touch the geometry.
        // Could be function application, could be something else.
        // Can we implement our own expressions externally that plugin?
        Ok(Box::new(NoOpPruningEvaluation))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &vortex_expr::ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        todo!()
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &vortex_expr::ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        todo!()
    }
}

impl LayoutReader for BBoxReader {
    fn layout(&self) -> &Layout {
        todo!()
    }

    fn row_count(&self) -> u64 {
        todo!()
    }

    fn dtype(&self) -> &DType {
        todo!()
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        // Children will be whichever children were inherited from the layouts.
        todo!()
    }
}
