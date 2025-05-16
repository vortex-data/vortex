use std::any::Any;
use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::segments::{SegmentId, SegmentSource};
use crate::{LayoutEncodingId, LayoutEncodingRef, LayoutReaderRef, VTable};

pub type LayoutRef = Arc<dyn Layout>;

pub trait Layout: 'static + Send + Sync {
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the [`crate::LayoutEncoding`] for this layout.
    fn encoding(&self) -> LayoutEncodingRef;

    fn row_count(&self) -> u64;

    fn dtype(&self) -> &DType;

    fn nchildren(&self) -> usize;

    fn segment_ids(&self) -> Vec<SegmentId>;

    fn new_reader(
        &self,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef>;
}

pub trait IntoLayout {
    /// Converts this type into a [`LayoutRef`].
    fn into_layout(self) -> LayoutRef;
}

impl dyn Layout + '_ {
    fn encoding_id(&self) -> LayoutEncodingId {
        self.encoding().id()
    }

    /// Collect the row splits for this layout tree. Each split represents the start/end of a chunk
    /// somewhere in the layout tree, filtered by the given field mask.
    pub fn collect_splits(&self, field_mask: &[FieldMask], splits: &mut BTreeSet<u64>) {
        todo!()
    }

    /// Downcast a layout to a specific type.
    pub fn into<V: VTable>(self: Arc<Self>) -> Arc<V::Layout> {
        let layout_adapter = self
            .as_any_arc()
            .downcast::<LayoutAdapter<V>>()
            .map_err(|this| vortex_err!("Invalid layout type"))
            .vortex_expect("Invalid layout type");

        // Now we can perform a cheeky transmute since we know the adapter is transparent.
        // SAFETY: The adapter is transparent and we know the underlying type is correct.
        unsafe { std::mem::transmute::<Arc<LayoutAdapter<V>>, Arc<V::Layout>>(layout_adapter) }
    }
}

#[repr(transparent)]
pub struct LayoutAdapter<V: VTable>(V::Layout);

impl<V: VTable> Layout for LayoutAdapter<V> {
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn encoding(&self) -> LayoutEncodingRef {
        V::encoding(&self.0)
    }

    fn row_count(&self) -> u64 {
        V::row_count(&self.0)
    }

    fn dtype(&self) -> &DType {
        V::dtype(&self.0)
    }

    fn nchildren(&self) -> usize {
        V::nchildren(&self.0)
    }

    fn segment_ids(&self) -> Vec<SegmentId> {
        V::segment_ids(&self.0)
    }

    fn new_reader(
        &self,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        todo!()
    }
}
