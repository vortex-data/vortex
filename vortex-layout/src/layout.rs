use std::any::Any;
use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arcref::ArcRef;
use itertools::Itertools;
use vortex_array::{ArrayContext, SerializeMetadata};
use vortex_dtype::{DType, FieldMask, FieldName};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::segments::{SegmentId, SegmentSource};
use crate::{LayoutEncodingId, LayoutEncodingRef, LayoutReaderRef, VTable};

pub type LayoutId = ArcRef<str>;

pub type LayoutRef = Arc<dyn Layout>;

pub trait Layout: 'static + Send + Sync + Debug + private::Sealed {
    fn as_any(&self) -> &dyn Any;

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    fn to_layout(&self) -> LayoutRef;

    /// Returns the [`crate::LayoutEncoding`] for this layout.
    fn encoding(&self) -> LayoutEncodingRef;

    /// The number of rows in this layout.
    fn row_count(&self) -> u64;

    /// The dtype of this layout.
    fn dtype(&self) -> &DType;

    /// The number of children in this layout.
    fn nchildren(&self) -> usize;

    /// Get the child at the given index.
    fn child(&self, idx: usize) -> VortexResult<LayoutRef>;

    /// Get the relative row offset of the child at the given index, returning `None` for
    /// any auxilliary children, e.g. dictionary values, zone maps, etc.
    fn child_type(&self, idx: usize) -> LayoutChildType;

    /// Get the metadata for this layout.
    fn metadata(&self) -> Vec<u8>;

    /// Get the segment IDs for this layout.
    fn segment_ids(&self) -> Vec<SegmentId>;

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()>;

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

/// A type that allows us to identify how a layout child relates to its parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutChildType {
    /// A layout child that retains the same schema and row offset position in the dataset.
    Transparent(Arc<str>),
    /// A layout child that provides auxiliary data, e.g. dictionary values, zone maps, etc.
    /// Contains a human-readable name of the child.
    Auxiliary(Arc<str>),
    /// A layout child that represents a row-based chunk of data.
    /// Contains the chunk index and relative row offset of the child.
    Chunk((usize, u64)),
    /// A layout child that represents a single field of data.
    /// Contains the field name of the child.
    Field(FieldName),
    // A layout child that contains a subset of the fields of the parent layout.
    // Contains a mask over the fields of the parent layout.
    // TODO(ngates): FieldMask API needs fixing before we enable this. We also don't yet have a
    //  use-case for this.
    // Mask(Vec<FieldMask>),
}

impl LayoutChildType {
    /// Returns the name of this child.
    pub fn name(&self) -> Arc<str> {
        match self {
            LayoutChildType::Chunk((idx, _offset)) => format!("[{idx}]").into(),
            LayoutChildType::Auxiliary(name) => name.clone(),
            LayoutChildType::Transparent(name) => name.clone(),
            LayoutChildType::Field(name) => name.clone(),
        }
    }

    /// Returns the relative row offset of this child.
    /// For auxiliary children, this is `None`.
    pub fn row_offset(&self) -> Option<u64> {
        match self {
            LayoutChildType::Chunk((_idx, offset)) => Some(*offset),
            LayoutChildType::Auxiliary(_) => None,
            LayoutChildType::Transparent(_) => Some(0),
            LayoutChildType::Field(_) => Some(0),
        }
    }
}

impl dyn Layout + '_ {
    /// The ID of the encoding for this layout.
    pub fn encoding_id(&self) -> LayoutEncodingId {
        self.encoding().id()
    }

    /// The children of this layout.
    pub fn children(&self) -> VortexResult<Vec<LayoutRef>> {
        (0..self.nchildren()).map(|i| self.child(i)).try_collect()
    }

    /// The child types of this layout.
    pub fn child_types(&self) -> impl Iterator<Item = LayoutChildType> {
        (0..self.nchildren()).map(|i| self.child_type(i))
    }

    /// The names of the children of this layout.
    pub fn child_names(&self) -> impl Iterator<Item = Arc<str>> {
        self.child_types().map(|child| child.name())
    }

    /// The row offsets of the children of this layout, where `None` indicates an auxilliary child.
    pub fn child_row_offsets(&self) -> impl Iterator<Item = Option<u64>> {
        self.child_types().map(|child| child.row_offset())
    }

    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    /// Downcast a layout to a specific type.
    pub fn as_<V: VTable>(&self) -> &V::Layout {
        self.as_opt::<V>().vortex_expect("Failed to downcast")
    }

    /// Downcast a layout to a specific type.
    pub fn as_opt<V: VTable>(&self) -> Option<&V::Layout> {
        self.as_any()
            .downcast_ref::<LayoutAdapter<V>>()
            .map(|adapter| &adapter.0)
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

    /// Depth-first traversal of the layout and its children.
    pub fn depth_first_traversal(&self) -> impl Iterator<Item = VortexResult<LayoutRef>> {
        /// A depth-first pre-order iterator over a layout.
        struct ChildrenIterator {
            stack: Vec<LayoutRef>,
        }

        impl Iterator for ChildrenIterator {
            type Item = VortexResult<LayoutRef>;

            fn next(&mut self) -> Option<Self::Item> {
                let next = self.stack.pop()?;
                let Ok(children) = next.children() else {
                    return Some(Ok(next));
                };
                for child in children.into_iter().rev() {
                    self.stack.push(child);
                }
                Some(Ok(next))
            }
        }

        ChildrenIterator {
            stack: vec![self.to_layout()],
        }
    }
}

#[repr(transparent)]
pub struct LayoutAdapter<V: VTable>(V::Layout);

impl<V: VTable> Debug for LayoutAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: VTable> Layout for LayoutAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_layout(&self) -> LayoutRef {
        Arc::new(LayoutAdapter::<V>(self.0.clone()))
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

    fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        V::child(&self.0, idx)
    }

    fn child_type(&self, idx: usize) -> LayoutChildType {
        V::child_type(&self.0, idx)
    }

    fn metadata(&self) -> Vec<u8> {
        V::metadata(&self.0).serialize()
    }

    fn segment_ids(&self) -> Vec<SegmentId> {
        V::segment_ids(&self.0)
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
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
