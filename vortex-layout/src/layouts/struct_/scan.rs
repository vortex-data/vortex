use std::sync::Arc;

use vortex_array::{ArrayData, ContextRef};
use vortex_error::{vortex_panic, VortexResult};
use vortex_expr::ExprRef;

use crate::layouts::struct_::StructLayout;
use crate::operations::{Operation, Poll};
use crate::reader::{EvalOp, LayoutReader};
use crate::segments::SegmentReader;
use crate::{LayoutData, LayoutEncoding, RowMask};

#[derive(Debug)]
pub struct StructScan {
    layout: LayoutData,
}

impl StructScan {
    pub(super) fn try_new(layout: LayoutData, _ctx: ContextRef) -> VortexResult<Self> {
        if layout.encoding().id() != StructLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self { layout })
    }
}

impl LayoutReader for StructScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn create_evaluator(
        self: Arc<Self>,
        _row_mask: RowMask,
        _expr: ExprRef,
    ) -> VortexResult<EvalOp> {
        todo!()
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct StructScanner {
    layout: LayoutData,
    mask: RowMask,
}

impl Operation for StructScanner {
    type Output = ArrayData;

    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        todo!()
    }
}
