use std::sync::Arc;

use vortex_array::{Array, ContextRef};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{Layout, LayoutReaderExt, LayoutVTable};

#[derive(Debug)]
pub struct FlatReader {
    layout: Layout,
    ctx: ContextRef,
    segments: Arc<dyn AsyncSegmentReader>,
}

impl FlatReader {
    pub(crate) fn try_new(
        layout: Layout,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        Ok(Self {
            layout,
            ctx,
            segments,
        })
    }

    pub(crate) fn ctx(&self) -> ContextRef {
        self.ctx.clone()
    }

    pub(crate) fn segments(&self) -> &dyn AsyncSegmentReader {
        self.segments.as_ref()
    }

    pub(crate) async fn array(&self) -> VortexResult<Array> {
        log::debug!("Fetching segment for FlatLayout {}", self.layout().name());
        let buffer = self
            .segments()
            .get(
                self.layout()
                    .segment_id(0)
                    .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?,
            )
            .await?;
        let row_count = usize::try_from(self.layout().row_count())
            .vortex_expect("FlatLayout row count does not fit within usize");

        Array::deserialize(buffer, self.ctx(), self.dtype().clone(), row_count)
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}
