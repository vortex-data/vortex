use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::{vortex_err, vortex_panic, VortexResult};
use vortex_scan::AsyncEvaluator;

use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::segments::{AsyncSegmentReader, SegmentId};
use crate::{LayoutData, LayoutEncoding};

pub struct FlatReader {
    layout: LayoutData,
    ctx: ContextRef,
    segments: Arc<dyn AsyncSegmentReader>,
    // The segment ID of the array in this FlatLayout.
    // NOTE(ngates): we don't cache the ArrayData here since the cache lives for as long as the
    //  reader does, which means likely holding a strong reference to the array for much longer
    //  than necessary, and potentially causing a memory leak.
    segment_id: SegmentId,
}

impl FlatReader {
    pub(crate) fn try_new(
        layout: LayoutData,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let segment_id = layout
            .segment_id(0)
            .ok_or_else(|| vortex_err!("FlatLayout missing SegmentID"))?;

        Ok(Self {
            layout,
            ctx,
            segments,
            segment_id,
        })
    }

    pub(crate) fn ctx(&self) -> ContextRef {
        self.ctx.clone()
    }

    pub(crate) fn segments(&self) -> &dyn AsyncSegmentReader {
        self.segments.as_ref()
    }

    pub(crate) fn segment_id(&self) -> SegmentId {
        self.segment_id
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn evaluator(&self) -> &dyn AsyncEvaluator {
        self
    }
}
