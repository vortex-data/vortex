use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use bytes::Bytes;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::Context;
use vortex_error::{vortex_err, VortexResult};
use vortex_flatbuffers::footer as fb;

use crate::read::cache::RelativeLayoutCache;
use crate::read::layouts::{
    ChunkedLayoutSpec, ColumnarLayoutSpec, FlatLayoutSpec, InlineDTypeLayoutSpec,
};
use crate::read::{LayoutReader, Scan};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayoutId(pub u16);

impl Display for LayoutId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

pub trait LayoutSpec: Debug + Send + Sync {
    fn id(&self) -> LayoutId;

    fn layout_reader(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_serde: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>>;
}

pub type LayoutSpecRef = &'static dyn LayoutSpec;

#[derive(Debug, Clone)]
pub struct LayoutContext {
    layout_refs: HashMap<LayoutId, LayoutSpecRef>,
}

impl LayoutContext {
    pub fn new(layout_refs: HashMap<LayoutId, LayoutSpecRef>) -> Self {
        Self { layout_refs }
    }

    pub fn lookup_layout(&self, id: &LayoutId) -> Option<LayoutSpecRef> {
        self.layout_refs.get(id).cloned()
    }
}

impl Default for LayoutContext {
    fn default() -> Self {
        Self::new(
            [
                &ColumnarLayoutSpec as LayoutSpecRef,
                &ChunkedLayoutSpec,
                &InlineDTypeLayoutSpec,
                &FlatLayoutSpec,
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
            .layout_reader(fb_bytes, fb_loc, scan, self.clone(), message_cache)
    }

    pub(crate) fn ctx(&self) -> Arc<Context> {
        self.ctx.clone()
    }
}
