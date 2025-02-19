use vortex_array::aliases::hash_map::HashMap;

use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::stats::StatsLayout;
use crate::layouts::struct_::StructLayout;
use crate::vtable::LayoutVTableRef;
use crate::LayoutId;

#[derive(Debug, Clone)]
pub struct LayoutContext {
    layout_refs: HashMap<LayoutId, LayoutVTableRef>,
}

pub type LayoutContextRef = std::sync::Arc<LayoutContext>;

impl LayoutContext {
    pub fn new(layout_refs: HashMap<LayoutId, LayoutVTableRef>) -> Self {
        Self { layout_refs }
    }

    pub fn with_layout(mut self, layout: LayoutVTableRef) -> Self {
        self.layout_refs.insert(layout.id(), layout);
        self
    }

    pub fn with_layouts<E: IntoIterator<Item = LayoutVTableRef>>(mut self, layouts: E) -> Self {
        self.layout_refs
            .extend(layouts.into_iter().map(|e| (e.id(), e)));
        self
    }

    pub fn lookup_layout(&self, id: LayoutId) -> Option<LayoutVTableRef> {
        self.layout_refs.get(&id).cloned()
    }
}

impl Default for LayoutContext {
    fn default() -> Self {
        Self::new(
            [
                LayoutVTableRef::from_static(&ChunkedLayout),
                LayoutVTableRef::from_static(&FlatLayout),
                LayoutVTableRef::from_static(&StructLayout),
                LayoutVTableRef::from_static(&StatsLayout),
            ]
            .into_iter()
            .map(|l| (l.id(), l))
            .collect(),
        )
    }
}
