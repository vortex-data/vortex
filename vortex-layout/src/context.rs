use vortex_array::{Context, Registry, RegistryBuilder};

use crate::LayoutEncodingRef;
use crate::layouts::chunked::ChunkedLayoutEncoding;
use crate::layouts::dict::DictLayoutEncoding;
use crate::layouts::flat::FlatLayoutEncoding;
use crate::layouts::struct_::StructLayoutEncoding;
use crate::layouts::zoned::ZonedLayoutEncoding;

pub type LayoutContext = Context<LayoutEncodingRef>;
pub type LayoutRegistry = Registry<LayoutEncodingRef>;
pub type LayoutRegistryBuilder = RegistryBuilder<LayoutEncodingRef>;

pub trait LayoutRegistryExt {
    /// Create a new registry with all out of the box layouts shipped by Vortex pre-registered.
    fn full() -> Self;
}

impl LayoutRegistryExt for LayoutRegistryBuilder {
    fn full() -> Self {
        LayoutRegistryBuilder::new().register_many([
            LayoutEncodingRef::new_ref(ChunkedLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(FlatLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(StructLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(ZonedLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(DictLayoutEncoding.as_ref()),
        ])
    }
}

impl LayoutRegistryExt for LayoutRegistry {
    fn full() -> Self {
        LayoutRegistryBuilder::full().build()
    }
}
