use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use bytes::Bytes;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::Context;
use vortex_error::{vortex_err, VortexResult};
use vortex_flatbuffers::footer as fb;

use crate::layouts::{ChunkedLayout, ColumnarLayout, FlatLayout, InlineDTypeLayout};
use crate::{LayoutReader, RelativeLayoutCache, Scan};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayoutId(pub u16);

impl Display for LayoutId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

pub trait Layout: Debug + Send + Sync {
    fn id(&self) -> LayoutId;

    fn reader(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_serde: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>>;
}

pub type LayoutRef = &'static dyn Layout;

#[derive(Debug, Clone)]
pub struct LayoutContext {
    layout_refs: HashMap<LayoutId, LayoutRef>,
}

impl LayoutContext {
    pub fn new(layout_refs: HashMap<LayoutId, LayoutRef>) -> Self {
        Self { layout_refs }
    }

    pub fn lookup_layout(&self, id: &LayoutId) -> Option<LayoutRef> {
        self.layout_refs.get(id).cloned()
    }
}

impl Default for LayoutContext {
    fn default() -> Self {
        Self::new(
            [
                &ColumnarLayout as LayoutRef,
                &ChunkedLayout,
                &InlineDTypeLayout,
                &FlatLayout,
            ]
            .into_iter()
            .map(|l| (l.id(), l))
            .collect(),
        )
    }
}

#[derive(Default, Debug, Clone)]
pub struct LayoutDeserializer {
    ctx: Arc<Context>,
    layout_ctx: Arc<LayoutContext>,
}

impl LayoutDeserializer {
    pub fn new(ctx: Arc<Context>, layout_ctx: Arc<LayoutContext>) -> Self {
        Self { ctx, layout_ctx }
    }

    pub fn read_layout(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        let fb_layout = unsafe {
            let tab = flatbuffers::Table::new(&fb_bytes, fb_loc);
            fb::Layout::init_from_table(tab)
        };
        let layout_id = LayoutId(fb_layout.encoding());
        self.layout_ctx
            .lookup_layout(&layout_id)
            .ok_or_else(|| vortex_err!("Unknown layout definition {layout_id}"))?
            .reader(fb_bytes, fb_loc, scan, self.clone(), message_cache)
    }

    pub(crate) fn ctx(&self) -> Arc<Context> {
        self.ctx.clone()
    }
}
