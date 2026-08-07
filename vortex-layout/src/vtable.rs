// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::DynLayout;
use crate::Layout;
use crate::LayoutBuildContext;
use crate::LayoutChildType;
use crate::LayoutChildren;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Shared, erased handle to a layout tree node.
pub type LayoutRef = Arc<dyn DynLayout>;

/// Layout-specific behavior for a typed [`Layout`].
///
/// Common serialized fields are stored by [`Layout`]. Implementations own only their
/// layout-specific data, metadata codec, child typing, and reader construction.
pub trait VTable: 'static + Clone + Send + Sync + Debug {
    /// Layout-specific data.
    type LayoutData: 'static + Clone + Send + Sync + Debug;
    /// Serialized metadata type.
    type Metadata: SerializeMetadata + vortex_array::DeserializeMetadata + Debug;

    /// Returns the globally unique layout ID.
    fn id(&self) -> LayoutId;

    /// Returns the serializable metadata for a layout.
    fn metadata(layout: &Layout<Self>) -> Self::Metadata;

    /// Deserialize and validate layout-specific data.
    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &<Self::Metadata as vortex_array::DeserializeMetadata>::Output,
    ) -> VortexResult<Self::LayoutData>;

    /// Construct a typed layout from deserialized common fields.
    fn build(
        vtable: &Self,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as vortex_array::DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Layout<Self>> {
        let args = LayoutDeserializeArgs {
            session: build_ctx.session,
            array_read_ctx: build_ctx.array_read_ctx,
            dtype,
            row_count,
            segment_ids,
            children,
        };
        let data = vtable.deserialize(&args, metadata)?;
        Ok(LayoutParts::new(
            vtable.clone(),
            dtype.clone(),
            row_count,
            args.segment_ids,
            children.to_arc(),
            data,
        )
        .into_typed())
    }

    /// Returns the number of logical child *slots* of this layout.
    ///
    /// Slots are fixed logical positions: a given child always occupies the same slot index
    /// regardless of which optional siblings are present. A slot may be absent (see
    /// [`slot_to_child`](VTable::slot_to_child)), in which case it has no corresponding serialized
    /// child. The default implementation reports one slot per serialized child, i.e. every slot is
    /// always present.
    fn nslots(layout: &Layout<Self>) -> usize {
        layout.nchildren()
    }

    /// Maps a logical `slot` to the index of its serialized (dense) child, or `None` if the slot
    /// is absent for this layout instance.
    ///
    /// Serialized children are stored densely (present-only), so an absent slot shifts the dense
    /// indices of the slots that follow it. This mapping centralizes that arithmetic; the default
    /// implementation is the identity, treating slot indices and dense child indices as equal.
    fn slot_to_child(layout: &Layout<Self>, slot: usize) -> Option<usize> {
        (slot < Self::nslots(layout)).then_some(slot)
    }

    /// Returns the expected dtype of the child in logical `slot`.
    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType>;

    /// Returns the relationship between the child in logical `slot` and its parent.
    fn child_type(layout: &Layout<Self>, slot: usize) -> LayoutChildType;

    /// Construct a reader for this layout.
    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef>;

    /// Returns `true` if this layout is indivisible: its readers never register natural split
    /// boundaries strictly inside their row range (see [`crate::LayoutReader::register_splits`]).
    ///
    /// Indivisible layouts — like flat, whose readers only ever push the end of the requested
    /// range — let parent layouts skip materializing the child entirely during split collection.
    fn is_indivisible(&self) -> bool {
        false
    }
}
