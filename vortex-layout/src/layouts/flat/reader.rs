use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::{vortex_panic, VortexResult};

use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::scan::ScanExecutor;
use crate::{Layout, LayoutVTable};

pub struct FlatReader {
    layout: Layout,
    ctx: ContextRef,
    executor: Arc<dyn ScanExecutor>,
}

impl FlatReader {
    pub(crate) fn try_new(
        layout: Layout,
        ctx: ContextRef,
        executor: Arc<dyn ScanExecutor>,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        Ok(Self {
            layout,
            ctx,
            executor,
        })
    }

    pub(crate) fn ctx(&self) -> ContextRef {
        self.ctx.clone()
    }

    pub(crate) fn executor(&self) -> &dyn ScanExecutor {
        self.executor.as_ref()
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}
