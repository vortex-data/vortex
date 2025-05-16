use std::any::Any;
use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_dtype::{DType, FieldMask, FieldPath};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::segments::{SegmentId, SegmentSource};
use crate::{LayoutEncodingId, LayoutEncodingRef, LayoutReaderRef, LayoutVisitor, VTable};

pub type LayoutRef = Arc<dyn Layout>;

pub trait Layout: 'static + Send + Sync + private::Sealed {
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the [`crate::LayoutEncoding`] for this layout.
    fn encoding(&self) -> LayoutEncodingRef;

    fn row_count(&self) -> u64;

    fn dtype(&self) -> &DType;

    fn nchildren(&self) -> usize;

    fn visit_children(&self, field_mask: Option<&[FieldMask]>, visitor: &mut dyn LayoutVisitor);

    fn segment_ids(&self) -> Vec<SegmentId>;

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    );

    fn new_reader(
        &self,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef>;
}

pub trait IntoLayout {
    /// Converts this type into a [`LayoutRef`].
    fn into_layout(self) -> LayoutRef;
}

impl dyn Layout + '_ {
    /// The ID of the encoding for this layout.
    pub fn encoding_id(&self) -> LayoutEncodingId {
        self.encoding().id()
    }

    /// The children of this layout.
    pub fn children(&self) -> Vec<LayoutRef> {
        struct ChildrenCollector(Vec<LayoutRef>);

        impl LayoutVisitor for ChildrenCollector {
            fn visit_child(
                &mut self,
                _name: &str,
                _row_offset: u64,
                _field_path: Option<&FieldPath>,
                child: &LayoutRef,
            ) {
                self.0.push(child.clone());
            }
        }

        let mut collector = ChildrenCollector(Vec::new());
        self.visit_children(None, &mut collector);
        collector.0
    }

    /// Downcast a layout to a specific type.
    pub fn into<V: VTable>(self: Arc<Self>) -> Arc<V::Layout> {
        let layout_adapter = self
            .as_any_arc()
            .downcast::<LayoutAdapter<V>>()
            .map_err(|_| vortex_err!("Invalid layout type"))
            .vortex_expect("Invalid layout type");

        // Now we can perform a cheeky transmute since we know the adapter is transparent.
        // SAFETY: The adapter is transparent and we know the underlying type is correct.
        unsafe { std::mem::transmute::<Arc<LayoutAdapter<V>>, Arc<V::Layout>>(layout_adapter) }
    }
}

impl Debug for dyn Layout + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layout")
            .field("encoding", &self.encoding())
            .field("row_count", &self.row_count())
            .field("dtype", &self.dtype())
            .field("nchildren", &self.nchildren())
            .finish()
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

    fn visit_children(&self, field_mask: Option<&[FieldMask]>, visitor: &mut dyn LayoutVisitor) {
        V::visit_children(&self.0, field_mask, visitor)
    }

    fn segment_ids(&self) -> Vec<SegmentId> {
        V::segment_ids(&self.0)
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) {
        V::register_splits(&self.0, field_mask, row_offset, splits)
    }

    fn new_reader(
        &self,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        V::new_reader(&self.0, name, segment_source, ctx)
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for LayoutAdapter<V> {}
}
