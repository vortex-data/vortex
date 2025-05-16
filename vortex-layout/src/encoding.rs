use std::any::Any;
use std::fmt::{Debug, Display, Formatter};

use arcref::ArcRef;
use vortex_array::DeserializeMetadata;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_panic};

use crate::segments::SegmentId;
use crate::{IntoLayout, LayoutChildren, LayoutRef, VTable};

pub type LayoutEncodingId = ArcRef<str>;
pub type LayoutEncodingRef = ArcRef<dyn LayoutEncoding>;

pub trait LayoutEncoding: 'static + Send + Sync + Debug + private::Sealed {
    fn as_any(&self) -> &dyn Any;

    fn id(&self) -> LayoutEncodingId;

    fn build(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<LayoutRef>;
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

    fn build(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<LayoutRef> {
        let metadata = <V::Metadata as DeserializeMetadata>::deserialize(metadata)?;
        let layout = V::build(&self.0, dtype, row_count, &metadata, segment_ids, children)?;

        // Validate that the builder function returned the expected values.
        if layout.row_count() != row_count {
            vortex_panic!(
                "Layout row count mismatch: {} != {}",
                layout.row_count(),
                row_count
            );
        }
        if layout.dtype() != dtype {
            vortex_panic!("Layout dtype mismatch: {} != {}", layout.dtype(), dtype);
        }

        Ok(layout.into_layout())
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
    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    pub fn as_<V: VTable>(&self) -> &V::Encoding {
        self.as_opt::<V>()
            .vortex_expect("LayoutEncoding is not of the expected type")
    }

    pub fn as_opt<V: VTable>(&self) -> Option<&V::Encoding> {
        self.as_any()
            .downcast_ref::<LayoutEncodingAdapter<V>>()
            .map(|e| &e.0)
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for LayoutEncodingAdapter<V> {}
}
