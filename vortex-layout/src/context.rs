use vortex_array::aliases::hash_map::HashMap;

use crate::encoding::{LayoutEncodingRef, LayoutId};
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::struct_::StructLayout;

#[derive(Debug, Clone)]
pub struct LayoutContext {
    layout_refs: HashMap<LayoutId, LayoutEncodingRef>,
}

pub type LayoutContextRef = std::sync::Arc<LayoutContext>;

impl LayoutContext {
    pub fn new(layout_refs: HashMap<LayoutId, LayoutEncodingRef>) -> Self {
        Self { layout_refs }
    }

    pub fn with_layout(mut self, layout: LayoutEncodingRef) -> Self {
        self.layout_refs.insert(layout.id(), layout);
        self
    }

    pub fn with_layouts<E: IntoIterator<Item = LayoutEncodingRef>>(mut self, layouts: E) -> Self {
        self.layout_refs
            .extend(layouts.into_iter().map(|e| (e.id(), e)));
        self
    }

    pub fn lookup_layout(&self, id: LayoutId) -> Option<LayoutEncodingRef> {
        self.layout_refs.get(&id).cloned()
    }
}

impl Default for LayoutContext {
    fn default() -> Self {
        Self::new(
            [
                &ChunkedLayout as LayoutEncodingRef,
                &FlatLayout,
                &StructLayout,
            ]
            .into_iter()
            .map(|l| (l.id(), l))
            .collect(),
        )
    }
}
