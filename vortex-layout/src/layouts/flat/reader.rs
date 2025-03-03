use std::sync::Arc;

use async_once_cell::OnceCell;
use vortex_array::serde::ArrayParts;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};

use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{Layout, LayoutReaderExt, LayoutVTable, instrument};

pub struct FlatReader {
    layout: Layout,
    ctx: ArrayContext,
    segment_reader: Arc<dyn AsyncSegmentReader>,
    // TODO(ngates): we need to add an invalidate_row_range function to evict these from the
    //  cache.
    array: Arc<OnceCell<ArrayRef>>,
}

impl FlatReader {
    pub(crate) fn try_new(
        layout: Layout,
        ctx: ArrayContext,
        segment_reader: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Self> {
        if layout.vtable().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        Ok(Self {
            layout,
            ctx,
            segment_reader,
            array: Arc::new(Default::default()),
        })
    }

    pub(crate) fn ctx(&self) -> &ArrayContext {
        &self.ctx
    }

    pub(crate) async fn array(&self) -> VortexResult<&ArrayRef> {
        self.array
            .get_or_try_init(instrument!(
                "flat_read",
                { name = self.layout().name() },
                async move {
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
                    let buffer = self.segment_reader.get(segment_id).await?;
                    let row_count = usize::try_from(self.layout().row_count())
                        .vortex_expect("FlatLayout row count does not fit within usize");

                    ArrayParts::try_from(buffer)?.decode(
                        self.ctx(),
                        self.dtype().clone(),
                        row_count,
                    )
                }
            ))
            .await
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}
