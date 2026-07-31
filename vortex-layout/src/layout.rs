// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::Id;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::children::LayoutChildren;
use crate::display::DisplayLayoutTree;
use crate::display::display_tree_with_segment_sizes;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable::LayoutRef;
use crate::vtable::VTable;

/// A unique identifier for a layout encoding.
pub type LayoutId = Id;

/// Pieces used to construct a typed layout.
pub struct LayoutParts<V: VTable> {
    vtable: V,
    dtype: DType,
    row_count: u64,
    segment_ids: Vec<SegmentId>,
    children: Arc<dyn LayoutChildren>,
    data: V::LayoutData,
}

impl<V: VTable> LayoutParts<V> {
    /// Create layout parts from common fields and vtable-specific data.
    pub fn new(
        vtable: V,
        dtype: DType,
        row_count: u64,
        segment_ids: Vec<SegmentId>,
        children: Arc<dyn LayoutChildren>,
        data: V::LayoutData,
    ) -> Self {
        Self {
            vtable,
            dtype,
            row_count,
            segment_ids,
            children,
            data,
        }
    }

    /// Convert these parts into a typed layout.
    pub fn into_typed(self) -> Layout<V> {
        Layout::from_parts(self)
    }

    /// Erase these parts into a layout reference.
    pub fn into_layout(self) -> LayoutRef {
        self.into_typed().into_layout()
    }
}

/// A typed layout node.
pub struct Layout<V: VTable> {
    inner: Arc<LayoutInner<V>>,
}

struct LayoutInner<V: VTable> {
    vtable: V,
    dtype: DType,
    row_count: u64,
    segment_ids: Vec<SegmentId>,
    children: Arc<dyn LayoutChildren>,
    data: V::LayoutData,
}

impl<V: VTable> Layout<V> {
    /// Construct a layout from explicit parts.
    pub fn from_parts(parts: LayoutParts<V>) -> Self {
        Self {
            inner: Arc::new(LayoutInner {
                vtable: parts.vtable,
                dtype: parts.dtype,
                row_count: parts.row_count,
                segment_ids: parts.segment_ids,
                children: parts.children,
                data: parts.data,
            }),
        }
    }

    /// Returns the vtable.
    pub fn vtable(&self) -> &V {
        &self.inner.vtable
    }

    /// Returns layout-specific data.
    pub fn data(&self) -> &V::LayoutData {
        &self.inner.data
    }

    /// Returns the logical dtype.
    pub fn dtype(&self) -> &DType {
        &self.inner.dtype
    }

    /// Returns the number of rows.
    pub fn row_count(&self) -> u64 {
        self.inner.row_count
    }

    /// Returns directly referenced segment IDs.
    pub fn segment_ids(&self) -> &[SegmentId] {
        &self.inner.segment_ids
    }

    /// Returns the child adapter.
    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.inner.children
    }

    /// Returns the number of serialized (present) children.
    pub fn nchildren(&self) -> usize {
        self.inner.children.nchildren()
    }

    /// Returns the number of logical child slots, including any that are absent.
    pub fn nslots(&self) -> usize {
        V::nslots(self)
    }

    /// Maps a logical `slot` to the index of its serialized child, or `None` if absent.
    pub fn slot_to_child(&self, slot: usize) -> Option<usize> {
        V::slot_to_child(self, slot)
    }

    /// Materialize the child in logical `slot`, or `None` if the slot is absent.
    pub fn slot(&self, slot: usize) -> VortexResult<Option<LayoutRef>> {
        match V::slot_to_child(self, slot) {
            Some(idx) => self
                .inner
                .children
                .child(idx, &V::child_dtype(self, slot)?)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Returns the relationship of the child in logical `slot` to this layout, or `None` if the
    /// slot is absent.
    pub fn slot_type(&self, slot: usize) -> Option<LayoutChildType> {
        V::slot_to_child(self, slot).map(|_| V::child_type(self, slot))
    }

    /// Returns a child's serialized row count without materializing it.
    pub fn child_row_count(&self, idx: usize) -> u64 {
        self.inner.children.child_row_count(idx)
    }

    /// Erase this typed layout into a shared layout reference.
    pub fn to_layout(&self) -> LayoutRef {
        self.clone().into_layout()
    }

    /// Erase this typed layout into a shared layout reference.
    pub fn into_layout(self) -> LayoutRef {
        Arc::new(self)
    }

    /// Construct a reader for this layout.
    pub fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        V::new_reader(self, name, segment_source, session, ctx)
    }
}

impl<V: VTable> Clone for Layout<V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<V: VTable> Debug for Layout<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layout")
            .field("encoding_id", &self.vtable().id())
            .field("dtype", &self.inner.dtype)
            .field("row_count", &self.inner.row_count)
            .field("segment_ids", &self.inner.segment_ids)
            .field("data", &self.inner.data)
            .finish()
    }
}

impl<V: VTable> Deref for Layout<V> {
    type Target = V::LayoutData;

    fn deref(&self) -> &Self::Target {
        self.data()
    }
}

impl<V: VTable> From<Layout<V>> for LayoutRef {
    fn from(value: Layout<V>) -> Self {
        value.into_layout()
    }
}

/// Erased layout behavior used by [`LayoutRef`].
pub trait DynLayout: 'static + Send + Sync + Debug {
    /// Returns this layout as [`Any`] for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Clone this layout as an erased reference.
    fn dyn_to_layout(&self) -> LayoutRef;

    /// Returns the layout ID.
    fn dyn_encoding_id(&self) -> LayoutId;

    /// Returns the row count.
    fn dyn_row_count(&self) -> u64;

    /// Returns the logical dtype.
    fn dyn_dtype(&self) -> &DType;

    /// Returns the number of serialized (present) children.
    fn dyn_nchildren(&self) -> usize;

    /// Returns the number of logical child slots, including any that are absent.
    fn dyn_nslots(&self) -> usize;

    /// Materializes the child in logical `slot`, or `None` if the slot is absent.
    fn dyn_slot(&self, slot: usize) -> VortexResult<Option<LayoutRef>>;

    /// Returns the relationship of the child in logical `slot`, or `None` if the slot is absent.
    fn dyn_slot_type(&self, slot: usize) -> Option<LayoutChildType>;

    /// Serializes layout-specific metadata.
    fn dyn_metadata(&self) -> Vec<u8>;

    /// Returns directly referenced segment IDs.
    fn dyn_segment_ids(&self) -> Vec<SegmentId>;

    /// Constructs a reader.
    fn dyn_new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef>;
}

impl<V: VTable> DynLayout for Layout<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_to_layout(&self) -> LayoutRef {
        Layout::to_layout(self)
    }

    fn dyn_encoding_id(&self) -> LayoutId {
        self.vtable().id()
    }

    fn dyn_row_count(&self) -> u64 {
        Layout::row_count(self)
    }

    fn dyn_dtype(&self) -> &DType {
        Layout::dtype(self)
    }

    fn dyn_nchildren(&self) -> usize {
        Layout::nchildren(self)
    }

    fn dyn_nslots(&self) -> usize {
        Layout::nslots(self)
    }

    fn dyn_slot(&self, slot: usize) -> VortexResult<Option<LayoutRef>> {
        Layout::slot(self, slot)
    }

    fn dyn_slot_type(&self, slot: usize) -> Option<LayoutChildType> {
        Layout::slot_type(self, slot)
    }

    fn dyn_metadata(&self) -> Vec<u8> {
        V::metadata(self).serialize()
    }

    fn dyn_segment_ids(&self) -> Vec<SegmentId> {
        self.inner.segment_ids.clone()
    }

    fn dyn_new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Layout::new_reader(self, name, segment_source, session, ctx)
    }
}

/// Identifies how a layout child relates to its parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutChildType {
    /// A child retaining the parent's schema and row offset.
    Transparent(Arc<str>),
    /// Auxiliary data, such as dictionary values or zone maps.
    Auxiliary(Arc<str>),
    /// A row-based chunk with its relative row offset.
    Chunk((usize, u64)),
    /// A single field of a struct.
    Field(FieldName),
}

impl LayoutChildType {
    /// Returns the child name.
    pub fn name(&self) -> Arc<str> {
        match self {
            Self::Chunk((idx, _)) => format!("[{idx}]").into(),
            Self::Auxiliary(name) | Self::Transparent(name) => Arc::clone(name),
            Self::Field(name) => name.clone().into(),
        }
    }

    /// Returns the relative row offset, or `None` for auxiliary children.
    pub fn row_offset(&self) -> Option<u64> {
        match self {
            Self::Chunk((_, offset)) => Some(*offset),
            Self::Auxiliary(_) => None,
            Self::Transparent(_) | Self::Field(_) => Some(0),
        }
    }
}

impl dyn DynLayout + '_ {
    /// Returns a cloned erased layout reference.
    pub fn to_layout(&self) -> LayoutRef {
        self.dyn_to_layout()
    }

    /// Returns the layout ID.
    pub fn encoding_id(&self) -> LayoutId {
        self.dyn_encoding_id()
    }

    /// Returns the logical dtype.
    pub fn dtype(&self) -> &DType {
        self.dyn_dtype()
    }

    /// Returns the number of rows.
    pub fn row_count(&self) -> u64 {
        self.dyn_row_count()
    }

    /// Returns the number of serialized (present) children.
    pub fn nchildren(&self) -> usize {
        self.dyn_nchildren()
    }

    /// Returns the number of logical child slots, including any that are absent.
    pub fn nslots(&self) -> usize {
        self.dyn_nslots()
    }

    /// Materializes the child in logical `slot`, or `None` if the slot is absent.
    pub fn slot(&self, slot: usize) -> VortexResult<Option<LayoutRef>> {
        self.dyn_slot(slot)
    }

    /// Returns the relationship of the child in logical `slot`, or `None` if the slot is absent.
    pub fn slot_type(&self, slot: usize) -> Option<LayoutChildType> {
        self.dyn_slot_type(slot)
    }

    /// Returns serialized layout-specific metadata.
    pub fn metadata(&self) -> Vec<u8> {
        self.dyn_metadata()
    }

    /// Returns directly referenced segment IDs.
    pub fn segment_ids(&self) -> Vec<SegmentId> {
        self.dyn_segment_ids()
    }

    /// Constructs a reader for this layout.
    pub fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        self.dyn_new_reader(name, segment_source, session, ctx)
    }

    /// Returns all serialized (present) children, in slot order.
    pub fn children(&self) -> VortexResult<Vec<LayoutRef>> {
        (0..self.nslots())
            .filter_map(|slot| self.slot(slot).transpose())
            .try_collect()
    }

    /// Returns the types of all serialized (present) children, in slot order.
    pub fn child_types(&self) -> impl Iterator<Item = LayoutChildType> + '_ {
        (0..self.nslots()).filter_map(|slot| self.slot_type(slot))
    }

    /// Returns all child names.
    pub fn child_names(&self) -> impl Iterator<Item = Arc<str>> + '_ {
        self.child_types().map(|child| child.name())
    }

    /// Returns all child row offsets.
    pub fn child_row_offsets(&self) -> impl Iterator<Item = Option<u64>> + '_ {
        self.child_types().map(|child| child.row_offset())
    }

    /// Returns whether this layout uses vtable `V`.
    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    /// Downcasts this layout to vtable `V`.
    pub fn as_<V: VTable>(&self) -> &Layout<V> {
        self.as_opt::<V>().vortex_expect("Failed to downcast")
    }

    /// Attempts to downcast this layout to vtable `V`.
    pub fn as_opt<V: VTable>(&self) -> Option<&Layout<V>> {
        self.as_any().downcast_ref()
    }

    /// Returns a depth-first pre-order traversal.
    pub fn depth_first_traversal(&self) -> impl Iterator<Item = VortexResult<LayoutRef>> {
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
                self.stack.extend(children.into_iter().rev());
                Some(Ok(next))
            }
        }

        ChildrenIterator {
            stack: vec![self.to_layout()],
        }
    }

    /// Displays the layout as a tree.
    pub fn display_tree(&self) -> DisplayLayoutTree {
        DisplayLayoutTree::new(self.to_layout(), false)
    }

    /// Displays the layout as a tree with optional verbose metadata.
    pub fn display_tree_verbose(&self, verbose: bool) -> DisplayLayoutTree {
        DisplayLayoutTree::new(self.to_layout(), verbose)
    }

    /// Displays the tree after fetching segment sizes.
    pub async fn display_tree_with_segments(
        &self,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<DisplayLayoutTree> {
        display_tree_with_segment_sizes(self.to_layout(), segment_source).await
    }
}

impl Display for dyn DynLayout + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let segments = self.segment_ids();
        if segments.is_empty() {
            write!(
                f,
                "{}({}, rows={})",
                self.encoding_id(),
                self.dtype(),
                self.row_count()
            )
        } else {
            write!(
                f,
                "{}({}, rows={}, segments=[{}])",
                self.encoding_id(),
                self.dtype(),
                self.row_count(),
                segments.iter().map(|s| format!("{}", **s)).join(", ")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_child_type_names_and_offsets() {
        let chunk = LayoutChildType::Chunk((5, 100));
        assert_eq!(chunk.name().as_ref(), "[5]");
        assert_eq!(chunk.row_offset(), Some(100));

        let field = LayoutChildType::Field(FieldName::from("customer_id"));
        assert_eq!(field.name().as_ref(), "customer_id");
        assert_eq!(field.row_offset(), Some(0));

        let auxiliary = LayoutChildType::Auxiliary("zone_map".into());
        assert_eq!(auxiliary.row_offset(), None);
        let transparent = LayoutChildType::Transparent("compressed".into());
        assert_eq!(transparent.row_offset(), Some(0));
    }
}
