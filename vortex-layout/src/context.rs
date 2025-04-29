use vortex_array::{VTableContext, VTableRegistry};

use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::stats::StatsLayout;
use crate::layouts::struct_::StructLayout;
use crate::vtable::LayoutVTableRef;

pub type LayoutContext = VTableContext<LayoutVTableRef>;
pub type LayoutRegistry = VTableRegistry<LayoutVTableRef>;

pub trait LayoutRegistryExt {
    fn default() -> Self;
}

impl LayoutRegistryExt for LayoutRegistry {
    fn default() -> Self {
        let mut this = Self::empty();
        this.register_many([
            LayoutVTableRef::new_ref(&ChunkedLayout),
            LayoutVTableRef::new_ref(&FlatLayout),
            LayoutVTableRef::new_ref(&StructLayout),
            LayoutVTableRef::new_ref(&StatsLayout),
            LayoutVTableRef::new_ref(&DictLayout),
        ]);
        this
    }
}
