use std::sync::{Arc, OnceLock};

use futures::FutureExt;
use vortex_array::ArrayContext;
use vortex_array::serde::ArrayParts;
use vortex_error::{VortexResult, VortexUnwrap as _, vortex_err, vortex_panic};

use crate::layouts::SharedArrayFuture;
use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;
use crate::{Layout, LayoutVTable};

pub struct FlatReader {
    pub(crate) layout: Layout,
    pub(crate) segment_source: Arc<dyn SegmentSource>,
    pub(crate) ctx: ArrayContext,
    pub(crate) array: OnceLock<SharedArrayFuture>,
}

impl FlatReader {
    pub(crate) fn try_new(
        layout: Layout,
        segment_source: Arc<dyn SegmentSource>,
        ctx: ArrayContext,
    ) -> VortexResult<Self> {
        if layout.vtable().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        Ok(Self {
            layout,
            segment_source,
            ctx,
            array: Default::default(),
        })
    }

    /// Returns a cached future that resolves this array.
    ///
    /// This method is idempotent, and returns a cached future on subsequent calls, all of which
    /// will use the original segment reader.
    // TODO(ngates): caching this and ignoring SegmentReaders may be a terrible idea... we may
    //  instead want to store all segment futures and race them, so if a layout requests a
    //  projection future before a pruning future, the pruning isn't blocked.
    pub(crate) fn array_future(&self) -> VortexResult<SharedArrayFuture> {
        let segment_id = self
            .layout
            .segment_id(0)
            .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;
        let row_count = usize::try_from(self.layout.row_count()).vortex_unwrap();

        // We create the segment_fut here to ensure we give the segment reader visibility into
        // how to prioritize this segment, even if the `array` future has already been initialized.
        // This is gross... see the function's TODO for a maybe better solution?
        let segment_fut = self.segment_source.request(segment_id, self.layout.name());

        Ok(self
            .array
            .get_or_init(|| {
                let ctx = self.ctx.clone();
                let dtype = self.layout.dtype().clone();
                async move {
                    let segment = segment_fut.await?;
                    ArrayParts::try_from(segment)?
                        .decode(&ctx, &dtype, row_count)
                        .map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone())
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        Ok(vec![])
    }
}
