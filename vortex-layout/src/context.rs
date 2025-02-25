use vortex_error::{VortexExpect, vortex_panic};

use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::stats::StatsLayout;
use crate::layouts::struct_::StructLayout;
use crate::vtable::LayoutVTableRef;

#[derive(Debug, Clone)]
pub struct LayoutContext {
    layout_refs: Vec<LayoutVTableRef>,
}

// TODO(ngates): internally Arc
pub type LayoutContextRef = std::sync::Arc<LayoutContext>;

impl LayoutContext {
    pub fn empty() -> Self {
        Self {
            layout_refs: Vec::new(),
        }
    }

    pub fn layouts(&self) -> impl Iterator<Item = &LayoutVTableRef> + '_ {
        self.layout_refs.iter()
    }

    pub fn with_layout(mut self, layout: LayoutVTableRef) -> Self {
        self.layout_refs.push(layout);
        self
    }

    pub fn with_layouts<E: IntoIterator<Item = LayoutVTableRef>>(mut self, layouts: E) -> Self {
        self.layout_refs.extend(layouts);
        self
    }

    /// Returns the index of the encoding in the context, or adds it if it doesn't exist.
    pub fn layout_idx(&mut self, encoding: &LayoutVTableRef) -> u16 {
        if let Some(idx) = self
            .layout_refs
            .iter()
            .position(|e| e.id() == encoding.id())
        {
            return u16::try_from(idx).vortex_expect("Cannot have more than u16::MAX layouts");
        }
        if self.layout_refs.len() >= u16::MAX as usize {
            vortex_panic!("Cannot have more than u16::MAX layouts");
        }
        self.layout_refs.push(encoding.clone());
        u16::try_from(self.layout_refs.len() - 1)
            .vortex_expect("Cannot have more than u16::MAX layouts")
    }

    /// Returns the encoding at the given index.
    pub fn lookup_layout(&self, idx: u16) -> Option<&LayoutVTableRef> {
        self.layout_refs.get(idx as usize)
    }
}

impl Default for LayoutContext {
    fn default() -> Self {
        Self {
            layout_refs: vec![
                LayoutVTableRef::new_ref(&ChunkedLayout),
                LayoutVTableRef::new_ref(&FlatLayout),
                LayoutVTableRef::new_ref(&StructLayout),
                LayoutVTableRef::new_ref(&StatsLayout),
            ],
        }
    }
}
