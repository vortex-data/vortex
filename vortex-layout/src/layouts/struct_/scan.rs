use async_trait::async_trait;
use vortex_array::{ArrayData, ContextRef};
use vortex_error::{vortex_panic, VortexResult};
use vortex_expr::ExprRef;
use vortex_scan::{AsyncEvaluator, RowMask};

use crate::layouts::struct_::StructLayout;
use crate::reader::LayoutReader;
use crate::{LayoutData, LayoutEncoding};

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

#[async_trait(?Send)]
impl AsyncEvaluator for StructScan {
    async fn evaluate(self: &Self, _row_mask: RowMask, _expr: ExprRef) -> VortexResult<ArrayData> {
        todo!()
    }
}

impl LayoutReader for StructScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn evaluator(&self) -> &dyn AsyncEvaluator {
        self
    }
}
