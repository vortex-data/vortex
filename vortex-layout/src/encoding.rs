use std::any::Any;
use std::fmt::{Debug, Display, Formatter};

use arcref::ArcRef;
use vortex_error::VortexExpect;

use crate::VTable;

pub type LayoutEncodingId = ArcRef<str>;
pub type LayoutEncodingRef = ArcRef<dyn LayoutEncoding>;

pub trait LayoutEncoding: 'static + Send + Sync + Debug + private::Sealed {
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

impl<V: VTable> Debug for LayoutEncodingAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutEncoding")
            .field("id", &self.id())
            .finish()
    }
}

impl Display for dyn LayoutEncoding + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl PartialEq for dyn LayoutEncoding + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn LayoutEncoding + '_ {}

impl dyn LayoutEncoding + '_ {
    pub fn as_<V: VTable>(&self) -> &V::Encoding {
        self.as_any()
            .downcast_ref::<LayoutEncodingAdapter<V>>()
            .map(|e| &e.0)
            .vortex_expect("LayoutEncoding is not of the expected type")
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for LayoutEncodingAdapter<V> {}
}
