use vortex_array::aliases::hash_map::HashMap;

use crate::encoding::{LayoutEncodingRef, LayoutId};
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::struct_::StructLayout;

#[derive(Debug, Clone)]
pub struct LayoutContext {
    layout_refs: HashMap<LayoutId, LayoutEncodingRef>,
}

impl LayoutContext {
    pub fn new(layout_refs: HashMap<LayoutId, LayoutEncodingRef>) -> Self {
        Self { layout_refs }
    }

    pub fn lookup_layout(&self, id: &LayoutId) -> Option<LayoutEncodingRef> {
        self.layout_refs.get(id).cloned()
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
