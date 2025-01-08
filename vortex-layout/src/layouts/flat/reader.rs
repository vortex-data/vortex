use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::{vortex_err, vortex_panic, VortexResult};
use vortex_expr::ExprRef;

use crate::layouts::flat::evaluator::FlatEvaluator;
use crate::layouts::flat::FlatLayout;
use crate::operations::OperationExt;
use crate::reader::{EvalOp, LayoutReader};
use crate::segments::SegmentId;
use crate::{LayoutData, LayoutEncoding, RowMask};

#[derive(Debug)]
pub struct FlatReader {
    layout: LayoutData,
    ctx: ContextRef,
    // The segment ID of the array in this FlatLayout.
    // NOTE(ngates): we don't cache the ArrayData here since the cache lives for as long as the
    //  reader does, which means likely holding a strong reference to the array for much longer
    //  than necessary, and potentially causing a memory leak.
    segment_id: SegmentId,
}

impl FlatReader {
    pub(crate) fn try_new(layout: LayoutData, ctx: ContextRef) -> VortexResult<Self> {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let segment_id = layout
            .segment_id(0)
            .ok_or_else(|| vortex_err!("FlatLayout missing SegmentID"))?;

        Ok(Self {
            layout,
            ctx,
            segment_id,
        })
    }

    pub(crate) fn ctx(&self) -> ContextRef {
        self.ctx.clone()
    }

    pub(crate) fn segment_id(&self) -> SegmentId {
        self.segment_id
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn create_evaluator(self: Arc<Self>, row_mask: RowMask, expr: ExprRef) -> VortexResult<EvalOp> {
        let filter_mask = row_mask.into_filter_mask()?;
        Ok(FlatEvaluator::new(self, filter_mask, expr).boxed())
    }
}
