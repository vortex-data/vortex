use vortex_array::{VTableContext, VTableRegistry};

use crate::LayoutEncodingRef;
use crate::layouts::chunked::ChunkedLayoutEncoding;
use crate::layouts::dict::DictLayoutEncoding;
use crate::layouts::flat::FlatLayoutEncoding;
use crate::layouts::stats::ZoneMapLayoutEncoding;
use crate::layouts::struct_::StructLayoutEncoding;

pub type LayoutContext = VTableContext<LayoutEncodingRef>;
pub type LayoutRegistry = VTableRegistry<LayoutEncodingRef>;

pub trait LayoutRegistryExt {
    fn default() -> Self;
}

impl LayoutRegistryExt for LayoutRegistry {
    fn default() -> Self {
        let mut this = Self::empty();
        this.register_many([
            LayoutEncodingRef::new_ref(ChunkedLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(FlatLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(StructLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(ZoneMapLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(DictLayoutEncoding.as_ref()),
        ]);
        this
    }
}
