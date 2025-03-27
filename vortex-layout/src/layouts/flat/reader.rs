use std::sync::{Arc, OnceLock};

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use vortex_array::serde::ArrayParts;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_error::{SharedVortexResult, VortexResult, vortex_err, vortex_panic};

use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::segments::SegmentReader;
use crate::{Layout, LayoutVTable};

pub(crate) type SharedArray = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;

pub struct FlatReader {
    pub(crate) layout: Layout,
    pub(crate) ctx: ArrayContext,
    pub(crate) array: OnceLock<SharedArray>,
}

impl FlatReader {
    pub(crate) fn try_new(layout: Layout, ctx: ArrayContext) -> VortexResult<Self> {
        if layout.vtable().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        Ok(Self {
            layout,
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
    pub(crate) fn array_future(
        &self,
        segment_reader: &dyn SegmentReader,
    ) -> VortexResult<SharedArray> {
        let segment_id = self
            .layout
            .segment_id(0)
            .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;
        let row_count = usize::try_from(self.layout.row_count())?;
        let segment_fut = segment_reader.get(segment_id, self.layout.name());

        Ok(self
            .array
            .get_or_init(|| {
                let ctx = self.ctx.clone();
                let dtype = self.layout.dtype().clone();
                async move {
                    let segment = segment_fut.await?;
                    ArrayParts::try_from(segment)?
                        .decode(&ctx, dtype.clone(), row_count)
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
