use vortex_array::{VTableContext, VTableRegistry};

use crate::LayoutRef;
use crate::layouts::chunked::ChunkedLayout;

pub type LayoutContext = VTableContext<LayoutRef>;
pub type LayoutRegistry = VTableRegistry<LayoutRef>;

pub trait LayoutRegistryExt {
    fn default() -> Self;
}

impl LayoutRegistryExt for LayoutRegistry {
    fn default() -> Self {
        let mut this = Self::empty();
        this.register_many([
            LayoutRef::new_ref(ChunkedLayout.as_ref()),
            LayoutRef::new_ref(FlatLayout.as_ref()),
            LayoutRef::new_ref(StructLayout.as_ref()),
            LayoutRef::new_ref(StatsLayout.as_ref()),
            LayoutRef::new_ref(DictLayout.as_ref()),
        ]);
        this
    }
}
