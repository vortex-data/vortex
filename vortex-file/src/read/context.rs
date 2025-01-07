use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use vortex_array::aliases::hash_map::HashMap;
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};
use vortex_flatbuffers::footer as fb;

use crate::layouts::{ChunkedLayout, ColumnarLayout, FlatLayout};
use crate::{LayoutPath, LayoutReader, Scan};

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
        path: LayoutPath,
        layout: fb::Layout,
        dtype: Arc<DType>,
        scan: Scan,
        layout_serde: LayoutDeserializer,
    ) -> VortexResult<Arc<dyn LayoutReader>>;
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
            [&ColumnarLayout as LayoutRef, &ChunkedLayout, &FlatLayout]
                .into_iter()
                .map(|l| (l.id(), l))
                .collect(),
        )
    }
}

#[derive(Default, Debug, Clone)]
pub struct LayoutDeserializer {
    ctx: ContextRef,
    layout_ctx: Arc<LayoutContext>,
}

impl LayoutDeserializer {
    pub fn new(ctx: ContextRef, layout_ctx: Arc<LayoutContext>) -> Self {
        Self { ctx, layout_ctx }
    }

    pub fn read_layout(
        &self,
        path: LayoutPath,
        layout: fb::Layout,
        scan: Scan,
        dtype: Arc<DType>,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        let layout_id = LayoutId(layout.encoding());
        self.layout_ctx
            .lookup_layout(&layout_id)
            .ok_or_else(|| vortex_err!("Unknown layout definition {layout_id}"))?
            .reader(path, layout, dtype, scan, self.clone())
    }

    pub(crate) fn ctx(&self) -> ContextRef {
        self.ctx.clone()
    }
}
