use std::any::Any;

use arcref::ArcRef;

use crate::VTable;

pub type LayoutEncodingId = ArcRef<str>;
pub type LayoutEncodingRef = ArcRef<dyn LayoutEncoding>;

pub trait LayoutEncoding: 'static + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn id(&self) -> LayoutEncodingId;
}

#[repr(transparent)]
pub struct LayoutEncodingAdapter<V: VTable>(V::Encoding);

impl<V: VTable> LayoutEncoding for LayoutEncodingAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> LayoutEncodingId {
        V::id(&self.0)
    }
}
