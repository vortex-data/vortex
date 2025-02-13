use std::sync::Arc;

use async_once_cell::OnceCell;
use vortex_array::{Array, ContextRef};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::scan::ScanExecutor;
use crate::{Layout, LayoutReaderExt, LayoutVTable};

pub struct FlatReader {
    layout: Layout,
    ctx: ContextRef,
    executor: Arc<ScanExecutor>,
    // TODO(ngates): we need to add an invalidate_row_range function to evict these from the
    //  cache.
    array: Arc<OnceCell<Array>>,
}

impl FlatReader {
    pub(crate) fn try_new(
        layout: Layout,
        ctx: ContextRef,
        executor: Arc<ScanExecutor>,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        Ok(Self {
            layout,
            ctx,
            executor,
            array: Arc::new(Default::default()),
        })
    }

    pub(crate) fn ctx(&self) -> ContextRef {
        self.ctx.clone()
    }

    pub(crate) fn executor(&self) -> &ScanExecutor {
        self.executor.as_ref()
    }

    pub(crate) async fn array(&self) -> VortexResult<&Array> {
        self.array
            .get_or_try_init(async move {
                let segment_id = self
                    .layout()
                    .segment_id(0)
                    .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;

                log::debug!(
                    "Requesting segment {} for flat layout {} expr",
                    segment_id,
                    self.layout().name(),
                );

                // Fetch all the array segment.
                let buffer = self.executor().get_segment(segment_id).await?;
                let row_count = usize::try_from(self.layout().row_count())
                    .vortex_expect("FlatLayout row count does not fit within usize");

                Array::deserialize(buffer, self.ctx(), self.dtype().clone(), row_count)
            })
            .await
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}
